## Nuclei-Segmentation using Pytorch

#Overview

#Data 
TNBC Data https://zenodo.org/record/1175282#.X29qm2gzZEY 
Dataset : Image 50개 / Mask 50개 
Image Shape : 512x512 

<div>
<img width='128' src='https://user-images.githubusercontent.com/52492949/96067497-d4254500-0ed4-11eb-8a41-9cd7717efb13.png'>
<img width='128' src='https://user-images.githubusercontent.com/52492949/96067543-eef7b980-0ed4-11eb-8cda-e46c83e94371.png'>
</div>

#Image Patch 
Data가 50개라 patch size로 잘라서 데이터 수를 증가시켰다. ( 50개 -> 12250개 )
Patch size : 64x64
Overlap = 32






#Model 

U-net 
