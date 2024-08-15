# Benchmarking-SDGGModels-2-UV---NE

## Step 1: Cloning repositories
1. Clone the DiffuseStyleGesture+ repository.
```angular2html
git clone https://github.com/YoungSeng/DiffuseStyleGesture.git
```

2. Inside the DiffuseStyleGesture+ directory that you cloned, you have to Clone the genea_numerical_evaluations repository.
```angular2html
git clone https://github.com/genea-workshop/genea_numerical_evaluations.git
```

3. Inside the DiffuseStyleGesture+ directory that you cloned, you have to Clone the genea_numerical_evaluations repository.
```angular2html
git clone https://github.com/AI-Unicamp/Benchmarking-SDGGModels-2-UV---NE.git
```

## Step 2: Downloading Genea 2023 Datasets
Download the Genea 2023 Train Dataset. To get it you can use [our link of Goolgle Drive](https://drive.google.com/drive/folders/1GvP67y8Ffi-3Y-pzGoZxMtyGKG0ZHT_4?usp=sharing) or you can use [the official web site of Genea 2023 in Zenodo](https://zenodo.org/records/8199133).  
Copy the downloaded dataset in the next directory path.
```angular2html
cd DiffuseStyleGesture+/Benchmarking-SDGGModels-2-UV---NE/Dataset/train_dataset
```

Download the Genea 2023 Test Dataset with only Speaker 1. To get it you can use [our link of Goolgle Drive](https://drive.google.com/drive/folders/1R-nvdXInAsqvJUuT8EY6fQ0TnbD7jlni?usp=sharing).
Copy the downloaded dataset in the next directory path.
```angular2html
cd DiffuseStyleGesture+/Benchmarking-SDGGModels-2-UV---NE/Dataset/tst_spk_1
```

## Step 3: Generating Unseen Voices
If you dont want generate all voices, then you can download the datasets with [our link of Google Drive](https://drive.google.com/drive/folders/1MkpCmmM0C9dyS5w7wQXKg71UTUPhqbvO?usp=sharing).
After download you have to put the directories in:
```angular2html
cd DiffuseStyleGesture+/Benchmarking-SDGGModels-2-UV---NE/Dataset/Unseen-Voices-with-Pitch-Changed
```

To generate all voices with voice conversion launch the next command:
```angular2html
LEO
```

## Step 4: Generating Voices in Noisy Environment
If you dont want generate all voices, then you can download the datasets with [our link of Google Drive]().
After download you have to put the directories in:
```angular2html
cd DiffuseStyleGesture+/Benchmarking-SDGGModels-2-UV---NE/Dataset/Voices-in-Noisy-Environment
```

To generate all voices in noisy environment launch the next command:
```angular2html
LEO
```




## Step 5: Processing
### Running Docker
1. Create docker image using the next command in your terminal:
```angular2html
docker build -t diffuse_style_gesture_plus .
```

2. Run container using the next command in your terminal, but note that you must change the directory path of your local machine, for example my directory path was "/work/kevin.colque/DiffuseStyleGesture", but in your case must be another path according to your directory:
```angular2html
docker run --rm -it --gpus all --userns=host --shm-size 64G -v /work/kevin.colque/DiffuseStyleGesture:/workspace/diffusestylegesture/ -p ‘9669:9669’ --name DIFFUSE_STYLE_GESTURE_PLUS_CONTAINER diffuse_style_gesture_plus_image:latest /bin/bash
```

3. Launch the virtual environment with the next command (Note that contain the activation of CUDA):
```angular2html
source activate diffuse
```

4. Go to our Workspace (Note that you can visualize it when launch us the container)
```angular2html
cd /workspace/diffusestylegesture/
```

### Gestures Generation
Tst Speaker 1
Tst VC 1
Tst VC 2
Tst VC 3
Tst VC 4
Tst Noise Low
Tst Noise Mid
Tst Noise High

### Calculate the Positions and 3D Rotations



### Training FGD


### Quick Start downloading all audios Datasets

