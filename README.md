# image-reconstruction-cv23
Timetable of delivery tasks is:
- [x] 20.11.2023, 16:00 Slides literature review 
- [x] 11.12.2023, 16:00 Slides idea 
- [ ] 08.01.2024, 16:00 Slides intermediate results  
- [ ] 22.01.2024, 16:00 Final results (slides, code) 

[project introduction slides](https://moodle.jku.at/jku/pluginfile.php/9527377/mod_resource/content/7/Project_Introduction_%28Abbass%29.pdf)  
[CVUE official google drive](https://drive.google.com/drive/folders/1UC6sGGWkRpJjqyYOnqByaa_mxeucFmqJ)

## TODOs:
- [ ] Waad - Dataloading -> check how it works with Keras https://keras.io/api/data_loading/image/ LazyLoading
- [ ] David - Training Loop with Keras!
  - Callback for creating predictions every ~15 epochs
  - Callback for checkpointing
- [ ] Max - Models in models.py

stage 2  
- [ ] ? - After Testrun in locally
- [ ] ? - produce several predictions for presentation


archive  
- [x] Raphi - Make AOS integrator work. Produce an Integral image. Ideally our complete focal stack.
- [x] Max - Make Architecture Work.  -> see code/main.ipynb Notes are in there
- [x] Moritz - Update the Slides, when Max and Raphi provide material.
- [x] Max - Check for changing shape of Residual connections. Is there a paper of some Blog about it ? (Our Encoder Convnet has 7 channels, our focal stack. And our decoder only 1, because the output is a grayscal image. Then we cannot simply reuse the classic residual connection, because it assumes same shape. I guess, we are not the first with this issue, there could be something out there)
- [x] Volunteer - Checkout neural network training API ? Maybe we can use keras, or some other Trainer (FastAI, MosaicML Composer). Would be nice maybe because then we probably save time writing the training loop etc. Not so important though, we can just implement classic pytorch Trainign loop.

### Open Questions: 
- is the AOS integrator complete? It looks like we need to adapt it
- What is the AOS integrator actually doing? for what do we need the parameters (sitting, standing etc.)

- How does changing shape affect the usefulness of Residual connections ? We have different shape in Encoder than in decoder. Collect ideas and read.
    -  Answer: One can use an extra conv block to change the number of channels.
     https://d2l.ai/chapter_convolutional-modern/resnet.html


# Selected Paper:
https://arxiv.org/pdf/1606.08921v3.pdf

Implementation on with Pytorch:  
https://github.com/yjn870/REDNet-pytorch/tree/master

# Notes on first steps:

### Rough overview of what we have to do.
*There are many ways to do this stuff, this is just a suggestion.*

1. Loop over ZIP files without uncompressing them. 
delete all that are incomplete (must be 13 files with prefix '<BATCH>_<ID>')

2. Check size if filtering of dataset was correct

3. Loop over ZIP again, take one run at a time and unzip, use AOS integrator, create integral images with focal stack [-0.2, -0.6, -1.0, -1.4, -1.8] (was updated), save them in a seperate directory.

4. When All integrals are there and look good, we dont need the ZIP files anymore (can delete to save disk space)

5. Create a Torch Dataset that loads one run into one Tensor.
    - Image: 512x512, 5 images per run
    - single Tensor : (5,512,512)            pytorch channels for CONVnets[B, C, H, W]

6. Split into train, test and validation set
    https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split

7. Create Dataloader

8. Use standard Hyperparameters and train on the train set and validate on the validation set.

9. Optionally adjust Hyperparameters or Preprocessing, repeat 8

10. Test on the test set.


---
---
---
---
---
---
---
---
---
---
---


# Archive:

## General Stuff:
List of different methods for machine learning denoising with code (convolutional AE, Multi-level wavelet CNN): 
https://towardsai.net/p/deep-learning/image-de-noising-using-deep-learning


## Denoising Autoencoder
take 'noisy'/ occluded image with trees, use AE or VAE to denoise the image https://towardsdatascience.com/denoising-autoencoders-explained-dbb82467fc2
So we only use a single integral image, with a focal length that is good. Or we take a focal stack, and use the max pixel of each image. The result should be okayish.

(+) cheap and fast to train. Good literature  
(-) coarse, loss of information throug using only one focal length, or through averaging/argmaxing  

Further References: 

Code Example:

https://github.com/chintan1995/Image-Denoising-using-Deep-Learning/blob/main/Models/(Baseline)_REDNet_256x256.ipynb
![image](https://github.com/mx-mn/image-reconstruction-cv23/assets/95431396/7af6449b-e540-496a-929e-71c8a442149b)

Denoising AEs:

https://omdena.com/blog/denoising-autoencoders/

Convolutional autoencoders for image noise reduction:

https://towardsdatascience.com/convolutional-autoencoders-for-image-noise-reduction-32fce9fc1763
![image](https://github.com/mx-mn/image-reconstruction-cv23/assets/95431396/10885e18-36d9-41b0-a1fc-e8faf87aa109)

## DEEEEEEP Learning
Take a stack of integral images with different focal stack. Use CNN or VIT. The different focal lengths are just like different color channels. 

(+) more information to use for learning  
(-) bigger network, more training, more compute, more time  
![image](https://github.com/mx-mn/image-reconstruction-cv23/assets/68200625/840d89f7-0bd2-4ee7-8e5f-6a3ded49e39b)

Further References:

Code Example:

https://github.com/chintan1995/Image-Denoising-using-Deep-Learning/blob/main/Models/MWCNN_256x256.ipynb
![image](https://github.com/mx-mn/image-reconstruction-cv23/assets/95431396/22939360-4a5c-4cc5-87c7-79018aeccfb2)

# Focal Stack
Just some notes.
The focal point can be arbitrarily selected ? 
Where could a person be?
It could be form 0-300cm 
Most likely in a continous block (if it is not a ghost) e.g.: from 0-170cm or from 0-40cm if the person is laying down or 0-110cm if the person is sitting

How can we match the focal length to these values? 
