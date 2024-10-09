# Benchmarking-SDGG-Models

## Step 1: Cloning repositories
1. Create a directory for your project
```angular2html
mkdir <name_of_your_project>
```

2. Inside of your project directory CLONE the DiffuseStyleGesture repository.
```angular2html
git clone https://github.com/YoungSeng/DiffuseStyleGesture.git
```

3. Inside of your project directory CLONE the Benchmarking-SDGG-Models repository.
```angular2html
git clone https://github.com/AI-Unicamp/Benchmarking-SDGG-Models.git
```

4. Enter your Benchmarking-SDGG-Models directory and CLONE the genea_numerical_evaluations repository.
```angular2html
cd Benchmarking-SDGG-Models
```
```angular2html
git clone https://github.com/genea-workshop/genea_numerical_evaluations.git
```

Sample here:

![Structure_of_directories](https://github.com/AI-Unicamp/Benchmarking-SDGG-Models/blob/main/Images-to-Readme/Structure_of_directories.png)

## Step 2: Downloading Genea 2023 Datasets 

2.1 Download the Genea 2023 Train Dataset. To obtain it, you can preferably use [our link of Google Drive](https://drive.google.com/drive/folders/1GvP67y8Ffi-3Y-pzGoZxMtyGKG0ZHT_4?usp=sharing), or as a last resort, you could use [the official web site of Genea 2023 in Zenodo](https://zenodo.org/records/8199133).  

Put the downloaded directory called `trn` in `./Benchmarking-SDGG-Models/Dataset/Genea2023/`

2.2 Download the Genea 2023 Test Dataset. To obtain it, you can preferably use [our link of Google Drive](https://drive.google.com/drive/folders/15IcRXcu6PI2DryfYLzMwSis4zEcMTFIK?usp=sharing), or as a last resort, you could use [the official web site of Genea 2023 in Zenodo](https://zenodo.org/records/8199133).

Put the downloaded directory called `tst` in `./Benchmarking-SDGG-Models/Dataset/Genea2023/`

2.3 Download the Genea 2023 Validation Dataset. To obtain it, you can preferably use [our link of Google Drive](https://drive.google.com/drive/folders/1qDKTZuwYnm-UstNtefLG4C8pPaXWg-Rh?usp=sharing), or as a last resort, you could use [the official web site of Genea 2023 in Zenodo](https://zenodo.org/records/8199133).

Put the downloaded directory called `val` in `./Benchmarking-SDGG-Models/Dataset/Genea2023/`

2.4 Download the audios WAV of Genea 2023 Test Dataset with only Speaker 1. To get it you can use [our link of Goolgle Drive](https://drive.google.com/drive/folders/1R-nvdXInAsqvJUuT8EY6fQ0TnbD7jlni?usp=sharing).
Copy the downloaded dataset in the next directory path.

Put the downloaded directory called `wav_spk_1` in `./Benchmarking-SDGG-Models/Dataset/Genea2023/`

Sample here:

![structure_dataset](https://github.com/AI-Unicamp/Benchmarking-SDGG-Models/blob/main/Images-to-Readme/structure_dataset.png)

## Step 3: Generating Unseen Voices 
3.1 If you dont want generate all voices, then you can download the datasets with [our link of Google Drive](https://drive.google.com/drive/folders/1MkpCmmM0C9dyS5w7wQXKg71UTUPhqbvO?usp=sharing).

Put the downloaded directory called `Unseen-Voices-with-VC` in `./Benchmarking-SDGG-Models/Dataset/`

3.2 To generate all voices with voice conversion launch the next command:
```angular2html
LEO
```

## Step 4: Generating Voices in Noisy Environment (TWH Party Dataset) 
4.1 If you dont want generate all voices, then you can download the datasets with [our link of Google Drive](https://drive.google.com/drive/folders/1IgvbrCVKkgDzZXfMyFUCZlEDsI6GU41j?usp=sharing).

Put the downloaded directory called `Voices-in-Noisy-Environment` in `./Benchmarking-SDGG-Models/Dataset/` 

4.2 To generate all voices in noisy environment launch the next command:
```angular2html
LEO
```

## Step 5: Processing 
### 5.1. Running Docker
1. Create docker image using the next command in your terminal:
```angular2html
docker build -t benchmarking_sdgg_models_image .
```

2. Run container using the next command in your terminal, but note that you must change the directory path of your local machine, for example my directory path was "/work/kevin.colque/DiffuseStyleGesture", but in your case must be another path according to your directory:
```angular2html
docker run --rm -it --gpus all --userns=host --shm-size 64G -v <path_of_your_project>:/workspace/benchmarking_sdgg/ -p ‘9669:9669’ --name BECHMARKING_SDGG_MODELS_CONTAINER benchmarking_sdgg_models_image:latest /bin/bash
```

3. Launch the virtual environment with the next command (Note that contain the activation of CUDA):
```angular2html
source activate sdgg
```

4. Go to our Workspace (Note that you can visualize it when launch us the container)
```angular2html
cd /workspace/benchmarking_sdgg/
```

Sample here:

![Structure of Directories](https://github.com/AI-Unicamp/Benchmarking-SDGG-Models/blob/main/Images-to-Readme/Sample_after_launch_container.png)

### 5.2. Download the models pre-trained needed to the Gestures Generation

1. Download files of DiffuseStyleGesture's pre-trained models from [google cloud](https://drive.google.com/drive/folders/1V83X4ZNYQZ_u5A1hKW8Tr9_4cui22TNw?usp=sharing). Put this two files inside of "DiffuseStyleGesture/BEAT-TWH-main/mydiffusion_beat_twh/TWH_mymodel4_512_v0/"
- Nota: If you want retrain and get your own checkpoints, you can go to the [DiffuseStyleGesture+](https://github.com/YoungSeng/DiffuseStyleGesture/tree/master/BEAT-TWH-main) repository and run the steps.

2. Download the "WavLM-Large.pt" from [google cloud](https://drive.google.com/drive/folders/14L5hR4q310KMt1SAt-1FNo4PfhT7Se3V?usp=sharing). Put this file inside of "DiffuseStyleGesture/BEAT-TWH-main/process/WavLM/"

3. Download the "crawl-300d-2M.vec" from [google cloud](https://drive.google.com/drive/folders/1wTB_dpLCVcvcmjwnjHb9esnNZL2cb1Rk?usp=sharing). Put this file inside of "DiffuseStyleGesture/BEAT-TWH-main/process/"

### 5.3. Generating Gestures to our own two Datasets (VC and TWH-Party).

1. Download the "generate.py" and "val_2023_v0_014_main-agent.npy" file from [google cloud](https://drive.google.com/drive/folders/1Pu9ob2YUm2rq4msSxeBrbsGsUeGjDnpz?usp=sharing). Put this file inside of "DiffuseStyleGesture/BEAT-TWH-main/mydiffusion_beat_twh/". (This file "generate.py" is similar to the given by DiffuseStyleGesture+, with respectively changes to our work)

2. Generate gestures from WAV audio files of **"Speaker 1 Test Dataset"**. To do this you can localize in "DiffuseStyleGesture/BEAT-TWH-main/mydiffusion_beat_twh/" and to run the next command in your terminal you need know which is the path of the WAV audios files of the Speaker 1 and which is the path of the tsv files of the "tst" dataset:
```angular2html
cd DiffuseStyleGesture/BEAT-TWH-main/mydiffusion_beat_twh/
```
```angular2html
python generate.py --wav_path ./../../../Benchmarking-SDGG-Models/Dataset/Genea2023/wav_spk_1/ --txt_path ./../../../Benchmarking-SDGG-Models/Dataset/Genea2023/tst/main-agent/tsv/
```
The output of BHVs is located in Benchmarking-SGDD-Models/BVH_generated/sample_model001200000/bvh_spk_1

It worked!! Right? 
Do you want to obtain the BVH files for the rest of the directories?
So, run the command with the same structure, but don't forget to change the <dataset_X_wav_path> part according to the BVH files you want to generate.
```angular2html
python generate.py --wav_path <dataset_X_wav_path> --txt_path ./../../../Benchmarking-SDGG-Models/Dataset/Genea2023/tst/main-agent/tsv/
```
3. To generate gestures from **Test Dataset with High, Mid and Low Noisy Environment** (TWH-Party) respectively
   - replace by: ./../../../Benchmarking-SDGG-Models/Dataset/Voices-in-Noisy-Environment/high/
   - replace by: ./../../../Benchmarking-SDGG-Models/Dataset/Voices-in-Noisy-Environment/mid/
   - replace by: ./../../../Benchmarking-SDGG-Models/Dataset/Voices-in-Noisy-Environment/low/
4. To generate gestures from ***Speaker 1 Test Dataset with Voice Conversion to Highest Pitch Man***
   - replace by: ./../../../Benchmarking-SDGG-Models/Dataset/Unseen-Voices-with-VC/wav_spk1w_ps-4_spk12m_high/
5. Generate gestures from ***Speaker 1 Test Dataset with Voice Conversion to Lowest Pitch Man***.
   - replace by: ./../../../Benchmarking-SDGG-Models/Dataset/Unseen-Voices-with-VC/wav_spk1w_ps-10_spk20m_low/
6. Generate gestures from ***Speaker 1 Test Dataset with Voice Conversion to Highest Pitch Woman***.
   - replace by: ./../../../Benchmarking-SDGG-Models/Dataset/Unseen-Voices-with-VC/wav_spk1w_ps3_spk18w_high/
7. Generate gestures from ***Speaker 1 Test Dataset with Voice Conversion to Lowest Pitch Woman***.
   - replace by: ./../../../Benchmarking-SDGG-Models/Dataset/Unseen-Voices-with-VC/wav_spk1w_ps-5_spk19w_low/

## Step 6: Evaluating metrics FGD and MSE 

<!-- ### 6.1 Calculate the Positions and 3D Rotations

First, it is necessary to calculate the 3D positions and rotations of the motion data in .bvh format of the dataset **trn**.

```angular2html
cd Benchmarking-SDGG-Models
```
```angular2html
python computing_positions_rotations_3D_dataloader.py --path './Dataset/Genea2023/trn/bvh' --load False
```
The `--load` argument: If set to `True`, the positions and rotations will be loaded if they have already been calculated; otherwise, they will be calculated from scratch.-->

### 6.1 Training autoencoder FGD

Calculate the 3D positions of the motion data in .bvh format from **trn** dataset that will be used to train the FGD autoencoder. We provide the pretrained autoencoder `model_checkpoint_epoch_49_90_246.bin` located inside the `'./Benchmarking-SDGG-Models/evaluation_metric/output'`.
```angular2html
cd ../../../Benchmarking-SDGG-Models/
```
```angular2html
python training_encoder.py
```
The checkpoints `model_checkpoint_epoch_xx_90_246.bin` generated from the training will be saved in `./Benchmarking-SDGG-Models/evaluation_metric/output`.

### 6.2 Calculate FGD and MSE

Calculate FGD and MSE metrics. 

```angular2html
python Computing_FGD.py --model_path 'model_checkpoint_epoch_49_90_246.bin'
```
The metric results will be saved in `Metrics-results-Noisy-Environment.txt` and `Metrics-results-Unseen-Voices-VC.txt` files, in `./Benchmarking-SDGG-Models/`.

## Citation

If our work is useful for you, please cite as:

```
@inproceedings{sanchez2024benchmarking,
  title={Benchmarking Speech-Driven Gesture Generation Models for Generalization to Unseen Voices and Noisy Environments},
  author={SANCHEZ, JOHSAC ISBAC GOMEZ and Inofuente-Colque, Kevin and Marques, Leonardo Boulitreau de Menezes Martins and Costa, Paula Dornhofer Paro and Tonoli, Rodolfo Luis},
  booktitle={GENEA: Generation and Evaluation of Non-verbal Behaviour for Embodied Agents Workshop 2024},
  year={2024}
}
```
