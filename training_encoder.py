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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genea Dataset BVH Loader")
    parser.add_argument('--path_bvh', type=str, default='./Dataset/Genea2023/trn/main-agent/bvh', help='Path to the dataset')
    #parser.add_argument('--path_bvh', type=str, default='./Dataset/Genea2023/trn/bvh', help='Path to the dataset')

    
    args = parser.parse_args()
    bvh_path=args.path_bvh
    
    print(f'Size dataset={len(os.listdir(args.path_bvh))}')

    my_dataset = GeneaDatasetBVHLoader(name='trn', 
                                   path=bvh_path, #bvh_path
                                   load=False,  #change to False to compute a save processed data
                                   pos_mean = None, # Leave it "None" to force computing mean and std
                                   step=10, 
                                   window=90)
    print(f'Size of the dataset position and rotation= {len(my_dataset.files)}')

###### Convertir a dataloader
from genea_numerical_evaluations.FGD.embedding_net import EmbeddingNet
from genea_numerical_evaluations.FGD.train_AE import AverageMeter, train_iter
import torch.nn.functional as F
from torch import optim

from torch.utils.data import DataLoader
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')

batch_size = 64
my_dataloader = DataLoader(dataset=my_dataset, batch_size=batch_size, shuffle=True)

###### Entrenar autoencoder
# train
loss_meters = [AverageMeter('loss')]

# interval params
print_interval = int(len(my_dataloader) / 5)

# init model and optimizer
pose_dim = my_dataset.__getitem__(0)[0].shape[1] # number of joint * 3d
generator = EmbeddingNet(pose_dim, my_dataset.window).to(device)
gen_optimizer = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))

def trainFGD(trn_loader):
    for epoch in range(50):
        for iter_idx, target in enumerate(trn_loader, 0):
            target_vec = target[0].float().to(device)
            loss = train_iter(target_vec, generator, gen_optimizer)
            
            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | '.format(epoch, iter_idx + 1)
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                print(print_summary)
                
                
        gen_state_dict = generator.state_dict()
        #save_name = f'./evaluation_metric/output/model_checkpoint_{trn_loader.dataset.window}_{pose_dim}.bin'
        save_name = f'./evaluation_metric/output/model_checkpoint_epoch_{epoch}_{trn_loader.dataset.window}_{pose_dim}.bin'
        # Asegúrate de que el directorio existe
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        torch.save({'pose_dim': pose_dim, 'n_frames': trn_loader.dataset.window, 'gen_dict': gen_state_dict}, save_name)
        print(f'Saving model checkpoint to: {save_name}')


trainFGD(my_dataloader)
