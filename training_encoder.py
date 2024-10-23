import os
import numpy as np
import sys
import argparse
sys.path.append('.') 
from Computing_BVH_Loader import GeneaDatasetBVHLoader

from genea_numerical_evaluations.FGD.embedding_net import EmbeddingNet
from genea_numerical_evaluations.FGD.train_AE import AverageMeter, train_iter
import torch.nn.functional as F
from torch import optim

from torch.utils.data import DataLoader
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genea Dataset BVH Loader")
    parser.add_argument('--path_bvh', type=str, default='./Dataset/Genea2023/trn/main-agent/bvh', help='Path to the dataset')
    parser.add_argument('--load', type=bool, default=False, help='Load preprocessed data if True, otherwise compute and save')

    
    args = parser.parse_args()
    bvh_path=args.path_bvh
    load_data = args.load
    
    print(f'Size dataset={len(os.listdir(args.path_bvh))}')

    if load_data:
        # If load is True, load the preprocessed data
        my_dataset = GeneaDatasetBVHLoader(
            name='trn',
            path=bvh_path,
            load=True,  # Load preprocessed data
            pos_mean='./Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_mean.npy', 
            pos_std='./Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_positions_std.npy',
            rot3d_mean='./Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_mean.npy',
            rot3d_std='./Dataset/Genea2023/trn/main-agent/bvh/report/trn_bvh_3drotations_std.npy',
            step=10,
            window=90
        )
    else:
        # If load is False, recalculate the data and save it
        my_dataset = GeneaDatasetBVHLoader(
            name='trn',
            path=bvh_path,
            load=False,  #change to False to compute a save processed data
            pos_mean=None,  # Leave it "None" to force computing mean and std
            step=10,
            window=90
        )
    print(f'Size of the dataset position and rotation= {len(my_dataset.files)}')

###### Convertir a dataloader
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
        save_name = f'./evaluation_metric/output/model_checkpoint_epoch_{epoch}_{trn_loader.dataset.window}_{pose_dim}.bin'
        # Asegúrate de que el directorio existe
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        torch.save({'pose_dim': pose_dim, 'n_frames': trn_loader.dataset.window, 'gen_dict': gen_state_dict}, save_name)
        print(f'Saving model checkpoint to: {save_name}')


trainFGD(my_dataloader)
