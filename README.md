# Benchmarking-SDGGModels-2-UV---NE

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

### Datasets

Copy the Genea Train directory in Benchmarking-SDGGModels-2-UV---NE Dataset directory. To Get the Genea Train directory you can do it by [our link of Goolgle Drive](https://drive.google.com/drive/folders/1V83X4ZNYQZ_u5A1hKW8Tr9_4cui22TNw?usp=sharing).

### Gestures Generation
Tst Speaker 1
Tst VC 1
Tst VC 2
Tst VC 3
Tst VC 4
Tst Noise Low
Tst Noise Mid
Tst Noise High

### Training FGD


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

4. This is ......
```angular2html
cd /workspace/diffusestylegesture/
```
