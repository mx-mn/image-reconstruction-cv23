# image-reconstruction-cv23
This is the repository for our group 


# TODOs
-[ ] brainstorm for pipelines
-[ ] find good focal length.

# Pipeline Ideas:
## Denoising Autoencoder
take 'noisy'/ occluded image with trees, use AE or VAE to denoise the image https://towardsdatascience.com/denoising-autoencoders-explained-dbb82467fc2
So we only use a single integral image, with a focal length that is good. Or we take a focal stack, and use the max pixel of each image. The result should be okayish.

(+) cheap and fast to train. Good literature
(-) coarse, loss of information throug using only one focal length, or through averaging/argmaxing 

## DEEEEEEP Learning
Take a stack of integral images with different focal stack. Use CNN or VIT. The different focal lengths are just like different color channels. 
(+) more information to use for learning
(-) bigger network, more training, more compute, more time
![image](https://github.com/mx-mn/image-reconstruction-cv23/assets/68200625/840d89f7-0bd2-4ee7-8e5f-6a3ded49e39b)


# Focal Stack
Just some notes.
The focal point can be arbitrarily selected ? 
Where could a person be?
It could be form 0-200cm 
Most likely in a continous block (if it is not a ghost) e.g.: from 0-170cm or from 0-40cm if the person is laying down or 0-110cm if the person is sitting

How can we match the focal length to these values? 
