import os
import numpy as np
import bvhsdk
from tqdm import tqdm
import sys
import argparse

class GeneaDatasetBVHLoader():
    def __init__(self,
                 name, # Só para facilitar. O nome pode ser o mesmo nome da pasta
                 path, # Pasta que contém os arquivos BVHs desse dataset
                 data_rep = 'pos', # Data representation. Indica a representação retornada pelo __getitem__()
                 step=60,    # Step da sliding window
                 window=120, # Tamanho da sliding window (se step < window: a janela terá overlap)
                 fps=30,     # frames per second do BVH
                 skipjoint = 1, # IGNORAR
                 metadata = False, # IGNORAR
                 njoints = 83,      # Qtd. de juntas do BVH
                 pos_mean = 'dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_mean.npy', # Arquivos mean e std para poses 3D e rotações 3D
                 pos_std = 'dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_std.npy',
                 rot3d_mean = 'dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_mean.npy',
                 rot3d_std = 'dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_std.npy',
                 **kwargs) -> None:
        
        # Salvando os argumentos da classe
        self.step = step
        self.window = window
        self.fps = fps
        self.name = name
        self.path = path
        self.skipjoint = skipjoint
        self.data_rep = data_rep
        self.njoints = njoints
        self.report_path = os.path.join(self.path, "report")
        self.computeMeanStd = True
        # Se foi passado caminhos para mean e std, lê os caminhos
        # ATENÇÃO: Passar esses argumentos como "None" para calcular mean e std
        if pos_mean and pos_std and rot3d_mean and rot3d_std:
            self.computeMeanStd = False
            self.pos_mean = np.load(pos_mean)
            self.pos_std = np.load(pos_std)
            self.rot3d_mean = np.load(rot3d_mean)
            self.rot3d_std = np.load(rot3d_std)
        
        # Se não houver uma pasta chamada Report, cria uma
        if not os.path.isdir(self.report_path):
            os.mkdir(self.report_path)
            
        # Compose files with bvhs in path our based on a files list passed as 
        self.files = kwargs.pop('files', [file for file in os.listdir(path) if file.endswith('.bvh')])
        self.files.sort()

        # Get parents vector (skeleton hierarchy)
        aux = bvhsdk.ReadFile(os.path.join(self.path,self.files[0]))
        self.parents = aux.arrayParent()

        # If load = True, loads already processed data
        if kwargs.pop('load', False):
            #Check if path is a file ending with ".npy"
            self.pos = np.load(os.path.join(self.report_path, self.name + "_bvh_positions.npy"), allow_pickle = True)
            self.rot3d = np.load(os.path.join(self.report_path, self.name + "_bvh_3drotations.npy"), allow_pickle = True)
        else:
            self.__data2samples(**kwargs)
            # This does not actually save a np array due to different lens of each take
            np.save(file = os.path.join(self.report_path, self.name + "_bvh_positions.npy"),
                    arr = self.pos,
                    allow_pickle = True)
            np.save(file = os.path.join(self.report_path, self.name + "_bvh_3drotations.npy"),
                    arr = self.rot3d,
                    allow_pickle = True)
            
            if self.computeMeanStd:
                self.pos_mean, self.pos_std = self.__computeMeanStd(self.pos)
                self.rot3d_mean, self.rot3d_std = self.__computeMeanStd(self.rot3d)
                np.save(file = os.path.join(self.report_path, self.name + "_bvh_positions_mean.npy"),
                        arr = self.pos_mean,
                        allow_pickle = True)
                np.save(file = os.path.join(self.report_path, self.name + "_bvh_positions_std.npy"),
                        arr = self.pos_std,
                        allow_pickle = True)
                np.save(file = os.path.join(self.report_path, self.name + "_bvh_3drotations_mean.npy"),
                        arr = self.rot3d_mean,
                        allow_pickle = True)
                np.save(file = os.path.join(self.report_path, self.name + "_bvh_3drotations_std.npy"),
                        arr = self.rot3d_std,
                        allow_pickle = True)
            
            
        self.rot3d_std[self.rot3d_std==0] = 1
        self.pos_std[self.pos_std==0] = 1
        self.frames = [len(take) for take in self.pos]
        self.samples_per_take = [len( [i for i in np.arange(0, n, self.step) if i + self.window <= n] ) for n in self.frames]
        self.samples_cumulative = [np.sum(self.samples_per_take[:i+1]) for i in range(len(self.samples_per_take))]
        self.length = self.samples_cumulative[-1]
        
    def __getitem__(self, index):
        file_idx = np.searchsorted(self.samples_cumulative, index+1, side='left')
        sample = index - self.samples_cumulative[file_idx-1] if file_idx > 0 else index
        b, e = sample*self.step, sample*self.step+self.window
        if self.data_rep == 'pos':
            sample = self.norma(self.pos[file_idx][b:e, self.skipjoint:, :], self.pos_mean[self.skipjoint:], self.pos_std[self.skipjoint:]).reshape(-1, (self.njoints-self.skipjoint)*3)
        elif self.data_rep == 'rot3d':
            sample = self.norma(self.rot3d[file_idx][b:e, self.skipjoint:, :], self.rot3d_mean[self.skipjoint:], self.rot3d_std[self.skipjoint:]).reshape(-1, (self.njoints-self.skipjoint)*3)
        return sample, self.files[file_idx] + f"_{b}_{e}"
    
    def norma(self, arr_, mean, std):
        return (arr_-mean) / std
    
    def inv_norma(self, arr_, mean, std):
        return (arr_*std) + mean
    
    def __len__(self):
        return self.length
    
    def posLckHips(self):
        """Locks the root position to the origin"""
        return [pos-np.tile(pos[:,0,:][:, np.newaxis], (1,self.njoints,1)) for pos in self.pos]

    def __data2samples(self, **kwargs):
        # Converts all files (takes) to samples
        self.pos, self.rot3d = [], []
        print('Preparing samples...')
        for i, file in enumerate(tqdm(self.files)):
            anim = bvhsdk.ReadFile(os.path.join(self.path,file))
            p, r = self.__loadtake(anim)
            self.pos.append(p)
            self.rot3d.append(r)
        print('Done. Converting to numpy.')
        #psize(self.pos, "Samples np")
        
    def __loadtake(self, anim):
        # Converts a single file (take) to samples
        # Compute joint position
        joint_positions, joint_rotations = [], []
        for frame in range(anim.frames):
            joint_positions.append([joint.getPosition(frame) for joint in anim.getlistofjoints()])
            joint_rotations.append([joint.rotation[frame] for joint in anim.getlistofjoints()])
        
        #size = psize(joint_positions, "All joints")
        
        return np.asarray(joint_positions), np.asarray(joint_rotations)
        
    def __computeMeanStd(self, arr):
        window = self.window
        mean, m2, counter = 0.0, 0.0, 0
        for i, take in enumerate(arr):
            duration = take.shape[0]
            for frame in range(0, duration-duration%window, window):
                mean += np.sum(take[frame:frame+window]     , axis = 0)
                m2   += np.sum(take[frame:frame+window] ** 2, axis = 0)
            counter += np.floor(duration/window)

        mean = mean/(counter*window)
        m2   = m2  /(counter*window)
        std = np.sqrt(m2 - mean ** 2)
        return mean, std
    

from genea_numerical_evaluations.FGD.embedding_net import EmbeddingNet
from genea_numerical_evaluations.FGD.train_AE import AverageMeter, train_iter
import torch.nn.functional as F
from torch import optim

from torch.utils.data import DataLoader
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


## Computing FGD
# Definir la función para cargar el generador
def load_generator(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pose_dim = checkpoint['pose_dim']
    window = checkpoint['n_frames']
    generator = EmbeddingNet(pose_dim, window).to(device)
    generator.load_state_dict(checkpoint['gen_dict'])
    generator.eval()
    return generator


# Ruta del checkpoint guardado

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkpoint Encoder")
    parser.add_argument('--model_path', type=str, default='./evaluation_metric/output/model_checkpoint_epoch_49_90_246.bin', help='Path to the dataset')

    
    args = parser.parse_args()
    checkpoint_dir = './evaluation_metric/output/'
    checkpoint_path = args.model_path
    if not os.path.isabs(checkpoint_path):  # Check if it's a relative path
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    # Load the generator
    generator = load_generator(checkpoint_path)
    #checkpoint_path = './evaluation_metric/output/model_checkpoint_epoch_49_90_246.bin'


# Cargar el generador
generator = load_generator(checkpoint_path)

def run_samples(network, loader, device):
    """
    Passa amostras para a rede e retorna todas as features
    Adapted to work for the FGD network
    """
    network.eval()
    with torch.no_grad():
        embeddings, original_labels, samples = [], [], []
        for j, data in enumerate(loader):
            inputs, labels = data
            inputs = inputs.to(device).float()
            f, _ = network(inputs)
            embeddings.extend(f.data.cpu().numpy())
            original_labels.extend(labels)
        embeddings = np.array(embeddings)
        original_labels = np.array(original_labels)
        return embeddings, original_labels

from scipy import linalg

def frechet_distance(samples_A, samples_B):
    A_mu = np.mean(samples_A, axis=0)
    A_sigma = np.cov(samples_A, rowvar=False)
    B_mu = np.mean(samples_B, axis=0)
    B_sigma = np.cov(samples_B, rowvar=False)
    try:
        #print('Computing frechet distance')
        frechet_dist = calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
    except ValueError:
        print('Something went wrong')
        frechet_dist = 1e+10
    return frechet_dist

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)


## Computing L1 and L2
def calculate_errors(dataset1, dataset2):
    mse = 0.0
    mae = 0.0

    # Obtener el número mínimo de muestras para evitar desbordamiento de índices
    num_samples = min(len(dataset1), len(dataset2))

    for i in range(num_samples):
        # Obtener solo las posiciones
        pos1, _ = dataset1[i]
        pos2, _ = dataset2[i]

        # Asegurarse de que las dimensiones de las posiciones coincidan
        #if pos1.shape != pos2.shape:
        #    raise ValueError(f"Shape mismatch: pos1 has shape {pos1.shape}, pos2 has shape {pos2.shape}")

        mse += np.mean((pos1 - pos2) ** 2)
        mae += np.mean(np.abs(pos1 - pos2))

    mse /= num_samples
    mae /= num_samples

    return mse, mae

batch_size = 64
path = './BVH_generated/sample_model001200000/bvh_tst/'
path_1 = './BVH_generated/sample_model001200000/'
output_file = 'Metrics-results-Noisy-Environment.txt'  # Nombre del archivo donde se guardarán los resultados

# Contar el número de archivos en 'bvh_tst'
num_files_in_tst = len([file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))])
print("tst")
my_dataset1 = GeneaDatasetBVHLoader(name='tst', 
                                    path=f'{path}', 
                                    load=False,  # Cambiar a False para computar y guardar datos procesados
                                    pos_mean = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_mean.npy',  # Archivo mean para poses 3D
                                    pos_std = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_std.npy',
                                    rot3d_mean = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_mean.npy',
                                    rot3d_std = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_std.npy',
                                    step=10, 
                                    window=90)

# Crear el archivo y escribir los encabezados
with open(output_file, 'w') as f:
    f.write("Directory\tFGD\tMSE\tMAE\n")  # Encabezados de las columnas

# Listar los elementos en path_1 y filtrar aquellos que tienen el mismo número de archivos que 'bvh_tst'
directories = [item for item in os.listdir(path_1) 
               if os.path.isdir(os.path.join(path_1, item)) and item != 'bvh_tst'
               and len([file for file in os.listdir(os.path.join(path_1, item)) if os.path.isfile(os.path.join(path_1, item, file))]) == num_files_in_tst]

for index1, item in enumerate(directories):
    if os.path.isdir(os.path.join(path_1, item)):
        print(item)
        # Cargar dataset
        my_dataset2 = GeneaDatasetBVHLoader(name=f'{item}', 
                                            path=f'{path_1}{item}', 
                                            load=False,  
                                            pos_mean = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_mean.npy',
                                            pos_std = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_std.npy',
                                            rot3d_mean = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_mean.npy',
                                            rot3d_std = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_std.npy',
                                            step=10, 
                                            window=90)

        # Crear DataLoaders
        my_dataloader1 = DataLoader(dataset=my_dataset1, batch_size=batch_size, shuffle=True)
        my_dataloader2 = DataLoader(dataset=my_dataset2, batch_size=batch_size, shuffle=True)

        # Ejecutar las muestras y calcular FGD
        gt_feat1, gt_labels = run_samples(generator, my_dataloader1, device)
        gt_feat2, gt_labels = run_samples(generator, my_dataloader2, device)

        fgd = frechet_distance(gt_feat1, gt_feat2)
        print("")
        print(f'Computing tst and {item}')
        print(f"FGD: {fgd}")

        # Calcular errores MSE y MAE
        mse, mae = calculate_errors(my_dataset1, my_dataset2)
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print("")

        # Guardar resultados en archivo de texto
        with open(output_file, 'a') as f:  # Usar 'a' para añadir al archivo existente
            f.write(f"{item}\t{fgd}\t{mse}\t{mae}\n")  # Guardar directorio, FGD, MSE y MAE

    print("")



##############################################################
batch_size = 64
path = './BVH_generated/sample_model001200000/bvh_tst1/'
path_1 = './BVH_generated/sample_model001200000/'
output_file = 'Metrics-results-Unseen-Voices-VC.txt'  # Nombre del archivo donde se guardarán los resultados

# Contar el número de archivos en 'bvh_tst'
num_files_in_tst = len([file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))])

# Crear el archivo y escribir los encabezados
with open(output_file, 'w') as f:
    f.write("Directory\tFGD\tMSE\tMAE\n")  # Encabezados de las columnas

print("tst1")
my_dataset1 = GeneaDatasetBVHLoader(name='tst1', 
                                    path=f'{path}', 
                                    load=False,  # Cambiar a False para computar y guardar datos procesados
                                    pos_mean = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_mean.npy',  # Archivo mean para poses 3D
                                    pos_std = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_std.npy',
                                    rot3d_mean = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_mean.npy',
                                    rot3d_std = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_std.npy',
                                    step=10, 
                                    window=90)

# Listar los elementos en path_1 y filtrar aquellos que tienen el mismo número de archivos que 'bvh_tst'
directories = [item for item in os.listdir(path_1) 
               if os.path.isdir(os.path.join(path_1, item)) and item != 'bvh_tst1'
               and len([file for file in os.listdir(os.path.join(path_1, item)) if os.path.isfile(os.path.join(path_1, item, file))]) == num_files_in_tst]

for index1, item in enumerate(directories):
    if os.path.isdir(os.path.join(path_1, item)):
        print(item)
        # Cargar dataset
        my_dataset2 = GeneaDatasetBVHLoader(name=f'{item}', 
                                            path=f'{path_1}{item}', 
                                            load=False,  
                                            pos_mean = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_mean.npy',
                                            pos_std = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_std.npy',
                                            rot3d_mean = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_mean.npy',
                                            rot3d_std = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_std.npy',
                                            step=10, 
                                            window=90)

        # Crear DataLoaders
        my_dataloader1 = DataLoader(dataset=my_dataset1, batch_size=batch_size, shuffle=True)
        my_dataloader2 = DataLoader(dataset=my_dataset2, batch_size=batch_size, shuffle=True)

        # Ejecutar las muestras y calcular FGD
        gt_feat1, gt_labels = run_samples(generator, my_dataloader1, device)
        gt_feat2, gt_labels = run_samples(generator, my_dataloader2, device)

        fgd = frechet_distance(gt_feat1, gt_feat2)
        print("")
        print(f'Computing tst and {item}')
        print(f"FGD: {fgd}")

        # Calcular errores MSE y MAE
        mse, mae = calculate_errors(my_dataset1, my_dataset2)
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print("")

        # Guardar resultados en archivo de texto
        with open(output_file, 'a') as f:  # Usar 'a' para añadir al archivo existente
            f.write(f"{item}\t{fgd}\t{mse}\t{mae}\n")  # Guardar directorio, FGD, MSE y MAE

    print("")







