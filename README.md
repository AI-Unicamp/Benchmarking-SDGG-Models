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

4. Inside of your project directory CLONE the Benchmarking-SDGG-Models repository.
```angular2html
git clone https://github.com/AI-Unicamp/Benchmarking-SDGG-Models.git
```

3. Enter your Benchmarking-SDGG-Models directory and CLONE the genea_numerical_evaluations repository.
```angular2html
cd Benchmarking-SDGG-Models
```
```angular2html
git clone https://github.com/genea-workshop/genea_numerical_evaluations.git
```

Sample here:

![Structure of Directories](https://github.com/AI-Unicamp/Benchmarking-SDGG-Models/blob/main/Images-to-Readme/Structure-of-directories.png)

## Step 2: Downloading Genea 2023 Datasets ------------------------------------------------
Download the Genea 2023 Train Dataset. To obtain it, you can preferably use [our link of Google Drive](https://drive.google.com/drive/folders/1GvP67y8Ffi-3Y-pzGoZxMtyGKG0ZHT_4?usp=sharing), or as a last resort, you could use [the official web site of Genea 2023 in Zenodo](https://zenodo.org/records/8199133).  
Put the downloaded directory called "trn" in the next directory path.
```angular2html
Benchmarking-SDGG-Models/Dataset/Genea2023/
```

Download the Genea 2023 Test Dataset. To obtain it, you can preferably use [our link of Google Drive](https://drive.google.com/drive/folders/15IcRXcu6PI2DryfYLzMwSis4zEcMTFIK?usp=sharing), or as a last resort, you could use [the official web site of Genea 2023 in Zenodo](https://zenodo.org/records/8199133).

Put the downloaded directory called "tst" in the next directory path.
```angular2html
Benchmarking-SDGG-Models/Dataset/Genea2023/
```

Download the audios WAV of Genea 2023 Test Dataset with only Speaker 1. To get it you can use [our link of Goolgle Drive](https://drive.google.com/drive/folders/1R-nvdXInAsqvJUuT8EY6fQ0TnbD7jlni?usp=sharing).
Copy the downloaded dataset in the next directory path.

Put the downloaded directory called "wav_spk_1" in the next directory path.
```angular2html
Benchmarking-SDGG-Models/Dataset/Genea2023/
```

## Step 3: Generating Unseen Voices -------------------------------------------------------
If you dont want generate all voices, then you can download the datasets with [our link of Google Drive](https://drive.google.com/drive/folders/1MkpCmmM0C9dyS5w7wQXKg71UTUPhqbvO?usp=sharing).
After download you have to put the directories in:
```angular2html
Benchmarking-SDGG-Models/Dataset/Unseen-Voices-with-VC
```

To generate all voices with voice conversion launch the next command:
```angular2html
LEO
```

## Step 4: Generating Voices in Noisy Environment (TWH Party Dataset) ---------------------
If you dont want generate all voices, then you can download the datasets with [our link of Google Drive](https://drive.google.com/drive/folders/1IgvbrCVKkgDzZXfMyFUCZlEDsI6GU41j?usp=sharing).
After download you have to put the directories in:
```angular2html
Benchmarking-SDGG-Models/Dataset/Voices-in-Noisy-Environment
```

To generate all voices in noisy environment launch the next command:
```angular2html
LEO
```

## Step 5: Processing ----------------------------------------------------------------------
### Running Docker
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

### Download the models pre-trained needed to the Gestures Generation

1. Download files of DiffuseStyleGesture's pre-trained models from [google cloud](https://drive.google.com/drive/folders/1V83X4ZNYQZ_u5A1hKW8Tr9_4cui22TNw?usp=sharing). Put this two files inside of "DiffuseStyleGesture/BEAT-TWH-main/mydiffusion_beat_twh/TWH_mymodel4_512_v0/"
- Nota: If you want retrain and get your own checkpoints, you can go to the [DiffuseStyleGesture+](https://github.com/YoungSeng/DiffuseStyleGesture/tree/master/BEAT-TWH-main) repository and run the steps.

2. Download the "WavLM-Large.pt" from [google cloud](https://drive.google.com/drive/folders/14L5hR4q310KMt1SAt-1FNo4PfhT7Se3V?usp=sharing). Put this file inside of "DiffuseStyleGesture/BEAT-TWH-main/process/WavLM/"

3. Download the "crawl-300d-2M.vec" from [google cloud](https://drive.google.com/drive/folders/1wTB_dpLCVcvcmjwnjHb9esnNZL2cb1Rk?usp=sharing). Put this file inside of "DiffuseStyleGesture/BEAT-TWH-main/process/"

### Generating Gestures to our own two Datasets.

1. Download the "generate.py" file from [google cloud](https://drive.google.com/drive/folders/1Pu9ob2YUm2rq4msSxeBrbsGsUeGjDnpz?usp=sharing). Put this file inside of "DiffuseStyleGesture/BEAT-TWH-main/mydiffusion_beat_twh/". (This file "generate.py" is similar to the given by DiffuseStyleGesture+, with respectively changes to our work)

2. Generate gestures from WAV audio files of **"Speaker 1 Test Dataset"**. To do this you can localize in "DiffuseStyleGesture/BEAT-TWH-main/mydiffusion_beat_twh/" and to run the next command in your terminal you need know which is the path of the WAV audios files of the Speaker 1 and which is the path of the tsv files of the "tst" dataset:
```angular2html
python generate.py --wav_path ./../../../Benchmarking-SDGG-Models/Dataset/Genea2023/wav_spk_1/ --txt_path ./../../../Benchmarking-SDGG-Models/Dataset/Genea2023/tst/main-agent/tsv/
```

```angular2html
python generate.py --wav_path <dataset_X_wav_path> --txt_path ./../../../Benchmarking-SDGG-Models/Dataset/Genea2023/tst/main-agent/tsv/
```
3. Generate gestures from **Test Dataset with High Noisy Environment** (TWH-Party).
   - To do this replace <dataset_X_wav_path>
   - by: ./../../../Benchmarking-SDGG-Models/Dataset/Voices-in-Noisy-Environment/high/
5. Generate gestures from **Test Dataset with Mid Noisy Environment** (TWH-Party).
   - To do this replace <dataset_X_wav_path>
   - by: ./../../../Benchmarking-SDGG-Models/Dataset/Voices-in-Noisy-Environment/mid/
7. Generate gestures from **Test Dataset with Low Noisy Environment** (TWH-Party).
   - To do this replace <dataset_X_wav_path>
   - by: ./../../../Benchmarking-SDGG-Models/Dataset/Voices-in-Noisy-Environment/low/
9. Generate gestures from ***Speaker 1 Test Dataset with Voice Conversion to Highest Pitch Man***.
   - To do this replace <dataset_X_wav_path>
   - by: ./../../../Benchmarking-SDGG-Models/Dataset/Unseen-Voices-with-VC/kkkkkkkkk/
11. Generate gestures from ***Speaker 1 Test Dataset with Voice Conversion to Lowest Pitch Man***.
   - To do this replace <dataset_X_wav_path>
   - by: ./../../../Benchmarking-SDGG-Models/Dataset/Unseen-Voices-with-VC/kkkkkkkkk/
13. Generate gestures from ***Speaker 1 Test Dataset with Voice Conversion to Highest Pitch Woman***.
   - To do this replace <dataset_X_wav_path>
   - by: ./../../../Benchmarking-SDGG-Models/Dataset/Unseen-Voices-with-VC/kkkkkkkkk/
15. Generate gestures from ***Speaker 1 Test Dataset with Voice Conversion to Lowest Pitch Woman***.
   - To do this replace <dataset_X_wav_path>
   - by: ./../../../Benchmarking-SDGG-Models/Dataset/Unseen-Voices-with-VC/kkkkkkkkk/

### Calculate the Positions and 3D Rotations

```angular2html
cd Benchmarking-SDGG-Models
```
```angular2html
python computing_positions_rotations_3D_dataloader.py --path './Dataset/Genea2023/trn/bvh' --load False
```

### Training FGD
```angular2html
python training_FGD.py
```

### Calculate FGD and MSE
```angular2html
python calculate_FGD_MSE.py
```
