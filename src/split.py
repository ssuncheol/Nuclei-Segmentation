import os 
import numpy as numpy
import shutil
import random
import numpy as np

root_dir = '/daintlab/data/TNBC/'


#Makedir Image & mask folders

classes_dir = ['image','mask']

for i in classes_dir:
  os.makedirs(root_dir+'train/' +i)
  os.makedirs(root_dir+'val/' + i)
  os.makedirs(root_dir+'test/'+i)

#Split Image

src = root_dir + 'patch_Image'
allFileNames = os.listdir(src)
allFileNames.sort()

#train,val,test = 0.6 : 0.2 : 0.2

val_ratio = 0.2 
test_ratio = 0.2 

np.random.seed(42)
np.random.shuffle(allFileNames)

train_filenames,val_filenames,test_filenames = np.split(np.array(allFileNames),
[int(len(allFileNames)*(1-val_ratio-test_ratio)),int(len(allFileNames)*(1-test_ratio))])

train_FileNames = [src+'/' + name for name in train_filenames.tolist()]
val_FileNames = [src+'/' + name for name in val_filenames.tolist()]
test_FileNames = [src+'/' + name for name in test_filenames.tolist()]

print('Total images:',len(allFileNames))
print('Train',len(train_FileNames))
print('Validation',len(val_FileNames))
print('Test',len(test_FileNames))

#Shutil Image

for name in train_FileNames:
    #print(name)
    shutil.copy(name,root_dir+'train/image/')

for name in val_FileNames:
    #print(name)
    shutil.copy(name,root_dir+'val/image/')    

for name in test_FileNames:
    #print(name)
    shutil.copy(name,root_dir+'test/image/')    


#Split mask 

src = root_dir + 'patch_mask'

allFileNames = os.listdir(src)
allFileNames.sort()

np.random.seed(42)
np.random.shuffle(allFileNames)

#train,val,test = 0.6 : 0.2 : 0.2

train_filenames,val_filenames,test_filenames = np.split(np.array(allFileNames),
[int(len(allFileNames)*(1-val_ratio-test_ratio)),int(len(allFileNames)*(1-test_ratio))])


train_FileNames = [src+'/' + name for name in train_filenames.tolist()]
val_FileNames = [src+'/' + name for name in val_filenames.tolist()]
test_FileNames = [src+'/' + name for name in test_filenames.tolist()]


print('Total images:',len(allFileNames))
print('Train',len(train_FileNames))
print('Validation',len(val_FileNames))
print('Test',len(test_FileNames))

#Shutil mask

for name in train_FileNames:
    #print(name)
    shutil.copy(name,root_dir+'train/mask/')

for name in val_FileNames:
    #print(name)
    shutil.copy(name,root_dir+'val/mask/')    

for name in test_FileNames:
    #print(name)
    shutil.copy(name,root_dir+'test/mask/')   