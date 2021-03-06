# Nuclei-Segmentation 

* The architecture was inspired by  [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

---

## Overview


### Data 
TNBC Data https://zenodo.org/record/1175282#.X29qm2gzZEY is used


* Dataset : Image 50개 / Mask 50개 

* Image Shape : 512x512 


Image            |  Mask
:-------------------------:|:-------------------------:
<img width='256' src='https://user-images.githubusercontent.com/52492949/96067497-d4254500-0ed4-11eb-8a41-9cd7717efb13.png'> | <img width='256' src='https://user-images.githubusercontent.com/52492949/96067543-eef7b980-0ed4-11eb-8cda-e46c83e94371.png'>



### Image Patch 

* 50개의 data를 patch size에  crop & 빈배경제거 ( 50개 -> 12250개 -> 7497개 )

* Patch size : 64x64 , Overlap = 32


Image            |  Mask
:-------------------------:|:-------------------------:
<img width='256' src='https://user-images.githubusercontent.com/52492949/96068639-76dec300-0ed7-11eb-9acf-a874dac0be29.png'> | <img width='256' src='https://user-images.githubusercontent.com/52492949/96068668-85c57580-0ed7-11eb-99ad-21cde23fb17a.png'>
<img width='256' src='https://user-images.githubusercontent.com/52492949/96068674-89f19300-0ed7-11eb-8631-eacb26765cf8.png'> | <img width='256' src='https://user-images.githubusercontent.com/52492949/96068653-7d6d3a80-0ed7-11eb-9930-e9bf2e42ef7c.png'>



### Data Split 

* train , val , test = 60% , 20% , 20%

* train , val , test = 4492 , 1497 , 1498


### Model [Unet]


<img width='512' src='https://user-images.githubusercontent.com/52492949/96069444-34b68100-0ed9-11eb-98da-7ab557b9ab1e.png'>


 
 
 
### Metric

* Iou 

---

## How to use 

### Languages 

<p align="left">
  <a href="#">
    <img src="https://github.com/MikeCodesDotNET/ColoredBadges/blob/master/svg/dev/languages/python.svg" alt="python" style="vertical-align:top; margin:6px 4px">
  </a> 

</p>

### Tools

<p align="left">
  <a href="#">
    <img src="https://github.com/MikeCodesDotNET/ColoredBadges/blob/master/svg/dev/tools/docker.svg" alt="docker" style="vertical-align:top; margin:6px 4px">
  </a> 

  <a href="#">
    <img src="https://github.com/MikeCodesDotNET/ColoredBadges/blob/master/svg/dev/tools/bash.svg" alt="bash" style="vertical-align:top; margin:6px 4px">
  </a> 

  <a href="#">
    <img src="https://github.com/MikeCodesDotNET/ColoredBadges/blob/master/svg/dev/tools/visualstudio_code.svg" alt="visualstudio_code" style="vertical-align:top; margin:6px 4px">
  </a> 

</p>

---

### Installing 

```sh 
pip install opencv-python-headless
```


### Run Example 
```sh
python3 main.py --lr=1e-3 --epochs=100 --train_batch=64 --val_batch=16 --test_batch=16 --weight_decay=0.0 --gpu=2,3
``` 

## Experiment :rocket:
Image            |  Output Segmentation Image  | Ground Truth
:-------------------------:|:-------------------------:|:-------------------------:
<img width='256' src='https://user-images.githubusercontent.com/52492949/97475604-ce2d6a80-1990-11eb-8d26-f009e1783fc0.png'>|<img width='256' src='https://user-images.githubusercontent.com/52492949/97475619-d1c0f180-1990-11eb-9ac9-a0199f6bfd23.png'>|<img width='256' src='https://user-images.githubusercontent.com/52492949/97475611-cff72e00-1990-11eb-9c5e-417b2ad9f18e.png'>

