import os
import numpy as np
import sys
import argparse    
sys.path.append('.') 
from Computing_BVH_Loader import GeneaDatasetBVHLoader
from Computing_BVH_Loader import load_generator, run_samples, frechet_distance, calculate_frechet_distance, calculate_errors

from torch.utils.data import DataLoader
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkpoint Encoder")
    parser.add_argument('--model_path', type=str, default='./evaluation_metric/output/model_checkpoint_epoch_49_90_246.bin', help='Path to the dataset')
    parser.add_argument('--load', type=bool, default=False, help='Load preprocessed data if True, otherwise compute and save')
    
    args = parser.parse_args()
    checkpoint_path = args.model_path
    load_data = args.load
    
    checkpoint_dir = './evaluation_metric/output/'
    if not os.path.isabs(checkpoint_path):  # Check if it's a relative path
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    # Load the generator
    generator = load_generator(checkpoint_path)


# computing FGD, L1 and L2
path = './BVH_generated/sample_model001200000/bvh_tst/'
path_1 = './BVH_generated/sample_model001200000/'
output_file = 'Metrics-results-Noisy-Environment.txt'  # Nombre del archivo donde se guardarán los resultados

# Contar el número de archivos en 'bvh_tst'
num_files_in_tst = len([file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))])
print("BVH reference: tst")
my_dataset1 = GeneaDatasetBVHLoader(name='tst', 
                                    path=f'{path}', 
                                    load=load_data,  # Cambiar a False para computar y guardar datos procesados
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
        print(f'BVH compared: {item}')
        # Cargar dataset
        my_dataset2 = GeneaDatasetBVHLoader(name=f'{item}', 
                                            path=f'{path_1}{item}', 
                                            load=load_data,  
                                            pos_mean = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_mean.npy',
                                            pos_std = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_std.npy',
                                            rot3d_mean = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_mean.npy',
                                            rot3d_std = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_std.npy',
                                            step=10, 
                                            window=90)

        # Crear DataLoaders
        my_dataloader1 = DataLoader(dataset=my_dataset1, batch_size=64, shuffle=True)
        my_dataloader2 = DataLoader(dataset=my_dataset2, batch_size=64, shuffle=True)

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
path = './BVH_generated/sample_model001200000/bvh_tst1/'
path_1 = './BVH_generated/sample_model001200000/'
output_file = 'Metrics-results-Unseen-Voices-VC.txt'  # Nombre del archivo donde se guardarán los resultados

# Contar el número de archivos en 'bvh_tst'
num_files_in_tst = len([file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))])

# Crear el archivo y escribir los encabezados
with open(output_file, 'w') as f:
    f.write("Directory\tFGD\tMSE\tMAE\n")  # Encabezados de las columnas

print("BVH reference: tst1")
my_dataset1 = GeneaDatasetBVHLoader(name='tst1', 
                                    path=f'{path}', 
                                    load=load_data,  # Cambiar a False para computar y guardar datos procesados
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
        print(f'BVH compared: {item}')
        # Cargar dataset
        my_dataset2 = GeneaDatasetBVHLoader(name=f'{item}', 
                                            path=f'{path_1}{item}', 
                                            load=load_data,  
                                            pos_mean = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_mean.npy',
                                            pos_std = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_std.npy',
                                            rot3d_mean = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_mean.npy',
                                            rot3d_std = './Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_std.npy',
                                            step=10, 
                                            window=90)

        # Crear DataLoaders
        my_dataloader1 = DataLoader(dataset=my_dataset1, batch_size=64, shuffle=True)
        my_dataloader2 = DataLoader(dataset=my_dataset2, batch_size=64, shuffle=True)

        # Ejecutar las muestras y calcular FGD
        gt_feat1, gt_labels = run_samples(generator, my_dataloader1, device)
        gt_feat2, gt_labels = run_samples(generator, my_dataloader2, device)

        fgd = frechet_distance(gt_feat1, gt_feat2)
        print("")
        print(f'Computing tst and {item}')
        print(f"FGD: {fgd}")

        # Calcular errores MSE y MAE
        mse, mae = calculate_errors(my_dataset1, my_dataset2)
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae}")
        print("")

        # Guardar resultados en archivo de texto
        with open(output_file, 'a') as f:  # Usar 'a' para añadir al archivo existente
            f.write(f"{item}\t{fgd}\t{mse}\t{mae}\n")  # Guardar directorio, FGD, MSE y MAE

    print("")


