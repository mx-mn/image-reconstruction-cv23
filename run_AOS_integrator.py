import os
import AOS_integrator_fct as AOS
import numpy as np
import pyaos
from LFR_utils import read_poses_and_images,pose_to_virtualcamera, init_aos, init_window

##########################################################

# Define the path to the dataset folder
dataset_path = r"D:\CV_Project\dataset"
output_path = r"D:\CV_Project\dataset_integrated"


#############################Start the AOS Renderer###############################################################
w,h,fovDegrees = 512, 512, 50 # # resolution and field of view. This should not be changed.

if 'window' not in locals() or window == None:
                                    
    window = pyaos.PyGlfwWindow( w, h, 'AOS' )  
    
aos = pyaos.PyAOS(w,h,fovDegrees) 


set_folder = r'C:\Users\Raphael\Desktop\PrivatRaphael\AI\ComputerVision\AOS_integrator\LFR\python'          # Enter path to your LFR/python directory
aos.loadDEM( os.path.join(set_folder,'zero_plane.obj'))
    

# Loop through each folder in the dataset folder
for root, dirs, files in os.walk(dataset_path):
    # Loop through each directory in the current folder

    for dir_name in dirs:
        # Check if the directory name is "thermal_imgs"
        if dir_name == "thermal_imgs":
            # Define the path to the current directory
            dir_path = os.path.join(root, dir_name)
            dir_output = os.path.join(output_path, os.path.basename(root))
            for focal_plane in np.arange(0, 3.2, 0.2):
                AOS.AOS_integrator(set_folder, aos, output_dir = dir_output, thermal_imgs_dir = dir_path, focal_plane = focal_plane)