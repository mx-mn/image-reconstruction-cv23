## A9 Group Repository

Download the test datasets:
https://drive.google.com/file/d/10GYP_sZqlu62puwqV8AzgGA_1EoXE0kj/view?usp=drive_link

Download all the predictions on the test dataset:
https://drive.google.com/file/d/1LtFovxUHovmr2h-lXI97Firbe-ilrifL/view?usp=drive_link



# environment
cd into the code directory
  ```zsh
  % conda env create --prefix venv -f environment.yml 
  % conda activate ./venv 
  ```

# make predictions `test.py`
use the trained model to create the reconstructed image(s) from given sample(s).
a demo directory is included. Complete dataset must be downloaded extra.
create predictions with executing:
```zsh
python test.py test_data/images/demo -o predictions/demo --combine
python test.py test_data/images/sample_real -o predictions/ --combine
``` 
The `--combine` flag indicates if a combination image should be created. It includes the ground truth, 3 focal planes and the prediction in one image, for easy comparisson.

# calculate metrics `evaluate.py`
calculate mean squared error and PSNR of given sample(s)

```zsh
python evaluate.py test_data/images/demo
``` 
evaluation metrics of the test sets can be found below:

## directory format
For both, `test.py` and `evaluate.py`, the input must have the following characteristics.
The directory is a sample directory.
In this case, its name must contain the word 'sample' and it must contain the following files, with the exact names:
```
'000.png','040.png','080.png','120.png','160.png','200.png', 'gt.png'
```
if there are duplicate files, behaviour is undefined.

Or to predict multiple samples at once, the provided directory contains several sample directories. Again, the sample directories are only considered as such, when their name contains the word 'sample'. 
See the demo input for reference.

# Metrics on the Test sets
evaluation script is run on each of the sets. 
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

# Other 
The folder `other` contains the code used to prepare the dataset and train the model. 
The model is defined in `train.py`.
For training we transformed the input images into numpy arrays and stored them as .npz files, which are great for fast loading.
In order to train the model on the whole dataset, we needed to load the data lazily. Thats why we used a format that does not need any processing during the training phase. 
A demo dataset with just 2 samples was added. you can just run
```
python other/training/train.py
```
and it will do the training we finally did.
Code for creating the integral images and cleaning the dataset is also in here.