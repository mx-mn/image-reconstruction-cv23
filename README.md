# image-reconstruction-cv23
Timetable of delivery tasks is:
- [ ] 20.11.2023, 16:00 Slides literature review 
- [ ] 11.12.2023, 16:00 Slides idea 
- [ ] 08.01.2024, 16:00 Slides intermediate results  
- [ ] 22.01.2024, 16:00 Final results (slides, code) 
	 

[project introduction slides](https://moodle.jku.at/jku/pluginfile.php/9527377/mod_resource/content/7/Project_Introduction_%28Abbass%29.pdf)

[drive](https://drive.google.com/drive/folders/1UC6sGGWkRpJjqyYOnqByaa_mxeucFmqJ)


# TODOs
- [ ] brainstorm for pipelines
- [ ] find good focal length.



# Pipeline Ideas:

## General Stuff:
List of different methods for machine learning denoising: https://towardsai.net/p/deep-learning/image-de-noising-using-deep-learning

## Denoising Autoencoder
take 'noisy'/ occluded image with trees, use AE or VAE to denoise the image https://towardsdatascience.com/denoising-autoencoders-explained-dbb82467fc2
So we only use a single integral image, with a focal length that is good. Or we take a focal stack, and use the max pixel of each image. The result should be okayish.

(+) cheap and fast to train. Good literature  
(-) coarse, loss of information throug using only one focal length, or through averaging/argmaxing  

Further References: 
https://omdena.com/blog/denoising-autoencoders/
Convolutional autoencoders for image noise reduction ![image](https://github.com/mx-mn/image-reconstruction-cv23/assets/95431396/10885e18-36d9-41b0-a1fc-e8faf87aa109)
https://towardsdatascience.com/convolutional-autoencoders-for-image-noise-reduction-32fce9fc1763


## DEEEEEEP Learning
Take a stack of integral images with different focal stack. Use CNN or VIT. The different focal lengths are just like different color channels. 

(+) more information to use for learning  
(-) bigger network, more training, more compute, more time  
![image](https://github.com/mx-mn/image-reconstruction-cv23/assets/68200625/840d89f7-0bd2-4ee7-8e5f-6a3ded49e39b)


# Focal Stack
Just some notes.
The focal point can be arbitrarily selected ? 
Where could a person be?
It could be form 0-300cm 
Most likely in a continous block (if it is not a ghost) e.g.: from 0-170cm or from 0-40cm if the person is laying down or 0-110cm if the person is sitting

How can we match the focal length to these values? 
