# Nuclei-Segmentation using Pytorch

## Overview

### Data 
TNBC Data https://zenodo.org/record/1175282#.X29qm2gzZEY 


Dataset : Image 50개 / Mask 50개 

Image Shape : 512x512 

<div>
<img width='128' src='https://user-images.githubusercontent.com/52492949/96067497-d4254500-0ed4-11eb-8a41-9cd7717efb13.png'>
<img width='128' src='https://user-images.githubusercontent.com/52492949/96067543-eef7b980-0ed4-11eb-8cda-e46c83e94371.png'>
</div>


### Image Patch 
Data가 50개라 patch size로 잘라서 데이터 수를 증가시키고, 빈배경 제거 ( 50개 -> 12250개 -> 7497개 )

Patch size : 64x64 , Overlap = 32

<div>
<img width='128' src='https://user-images.githubusercontent.com/52492949/96068639-76dec300-0ed7-11eb-9acf-a874dac0be29.png'>
<img width='128' src='https://user-images.githubusercontent.com/52492949/96068668-85c57580-0ed7-11eb-99ad-21cde23fb17a.png'>
</div>

<div>
<img width='128' src='https://user-images.githubusercontent.com/52492949/96068674-89f19300-0ed7-11eb-8631-eacb26765cf8.png'>
<img width='128' src='https://user-images.githubusercontent.com/52492949/96068653-7d6d3a80-0ed7-11eb-9930-e9bf2e42ef7c.png'>
</div>


### Data Split 
train , val , test = 60% , 20% , 20%

train , val , test = 4492 , 1497 , 1498


### Model 
Unet

<img width='512' src='https://user-images.githubusercontent.com/52492949/96069444-34b68100-0ed9-11eb-98da-7ab557b9ab1e.png'>
