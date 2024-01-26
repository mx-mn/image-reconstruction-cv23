## A9 Group Repository

### naming convention for inputs
a **sample** must be provided in a directory, exactly like this. 
if there are duplicate files, behaviour is undefined. files with other names are ignored.
All images must have the same dimensions. only .png is supported. 
```
'000.png'  
'040.png'
'080.png'
'120.png'
'160.png'
'200.png'
'gt.png'
```

# Set up 
`conda env create -f environment.yml`

# test.py
use the trained model to create the reconstructed image from the focal stack.
provide a directory, that contains

# evaluate.py
evaluate the models mean squared error.
provide a directory that either:
- is a sample directory, containing files according to the **naming convention** defined above
- contains 1-n sample-directories.
a sample directory is only interpreted as such, if the name contains the word 'sample'.
The metrics are evaluated as average over the samples.

evaluation metrics of the test set:
> \> python submission/A9/code/evaluate.py DIRECTORY


# Create Predictions
```powershell
#invoke the test script to get predictions
python test.py dir/with/focal/stack 

# optionally specify output directory, otherwise its the same as input directory
python test.py dir/with/focal/stack --output_dir dir/to/save/result
```

## Overview
`this is intended to be a complete description of what we did.`

We received data containing many thousands of samples. Each sample consists of 11(?) images, a ground truth image and a parameters file, containing metadata.
First, we sorted out all the samples where not all files are present. This resulted in removing around 2.8% of all samples.

Then we create the integral image with the provided tools for each sample, for a focal stack `[0, -0.4, -0.8, -1.2, -1.6, -2.0]`.

All the samples are randomly split into training set and validation and test set.

The Training set contains 27281 samples. From these 27281, we remove the samples where there is no person for training. This reduces the training set size to 24632. Distribution:
```
by pose
laying          8213           
idle            8182           
sitting         8237   

by number of  trees per ha
0               2552  
100             14709          
200             7371           
```

The validation and test data together count `4815` samples.
A random validation set of `128` images was selected to use during the training process.
This validation set also contains samples with no person, to accurately represent the data.
```
by pose
no_person       12             
laying          39             
idle            31             
sitting         46             

by number of  trees per ha
0               12             
100             81             
200             35             
```

The remaining `4687` images are not used for training and can be used to assess final model quality.
We create a collection of images based on the scenario they depict. for each scenario, 45 images are selected at random. 
One such scenario is for example 'no_person, 200 trees per ha'.
All possible combinations of pose and number of trees, of which there are 12, are used to create the test set.
In total, there are 12*45=540 samples.

the evaluation script is run on each of the sets. The results are collected here and also in a table

![test loss by pose](results/test-loss-by-trees-and-pose.png)
test loss by pose and number of trees

```python
#DIRECTORY: data/test/cases/images/0_trees_idle
MSE  : 0.00004
PSNR : 44.31280 dB

#DIRECTORY: data/test/cases/images/0_trees_laying
MSE  : 0.00004
PSNR : 43.90106 dB

#DIRECTORY: data/test/cases/images/0_trees_sitting
MSE  : 0.00004
PSNR : 44.52174 dB

#DIRECTORY: data/test/cases/images/0_trees_no_person
MSE  : 0.00002
PSNR : 48.16854 dB

#DIRECTORY: data/test/cases/images/100_trees_idle
MSE  : 0.00328
PSNR : 24.83675 dB

#DIRECTORY: data/test/cases/images/100_trees_laying
MSE  : 0.00364
PSNR : 24.39364 dB

#DIRECTORY: data/test/cases/images/100_trees_sitting
MSE  : 0.00345
PSNR : 24.62316 dB

#DIRECTORY: data/test/cases/images/100_trees_no_person
MSE  : 0.00314
PSNR : 25.03689 dB

#DIRECTORY: data/test/cases/images/200_trees_idle
MSE  : 0.00782
PSNR : 21.07045 dB

#DIRECTORY: data/test/cases/images/200_trees_laying
MSE  : 0.01083
PSNR : 19.65303 dB

#DIRECTORY: data/test/cases/images/200_trees_sitting
MSE  : 0.00906
PSNR : 20.42896 dB

#DIRECTORY: data/test/cases/images/200_trees_no_person
MSE  : 0.00913
PSNR : 20.39679 dB
```

The dataset is still too large to load into memory at once, so in order to train on the whole dataset
