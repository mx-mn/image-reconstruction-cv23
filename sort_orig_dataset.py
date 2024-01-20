import os
import shutil
#This code puts the thermal images of the dataset in respective, seperate folders

# Define the path to the dataset folder
dataset_path = r"C:\Users\Raphael\Desktop\CV_Project\dataset"

# Loop through each folder in the dataset folder
for foldername in os.listdir(dataset_path):
    # Define the path to the current folder
    folder_path = os.path.join(dataset_path, foldername)
    
    # Check if the current path is a directory
    if os.path.isdir(folder_path):
        # Define the path to the thermal_imgs folder
        thermal_imgs_path = os.path.join(folder_path, "thermal_imgs")

        # Create the thermal_imgs folder if it doesn't exist
        if not os.path.exists(thermal_imgs_path):
            os.makedirs(thermal_imgs_path)
        # Loop through each file in the current folder
        for filename in os.listdir(folder_path):
            # Check if the current file is not the ground truth img
            if not "GT" in filename and not "thermal_imgs" in filename and not "Parameters" in filename:
                # Define the source path of the current file
                source_path = os.path.join(folder_path, filename)
                
                # Define the destination path of the current file
                destination_path = os.path.join(thermal_imgs_path, filename)
                #print(filename)
                # Copy the current file to the thermal_imgs folder
                shutil.move(source_path, destination_path)
