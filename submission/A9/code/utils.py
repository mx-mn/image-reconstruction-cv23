
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

__FOCAL_LENGTHS =  ['000', '040', '080', '120', '160', '200']

def fing_ground_truth(directory: Path):
    gt_path = [file for file in directory.iterdir() if 'gt' in file.name][0]
    image = cv2.imread(gt_path.as_posix(), cv2.IMREAD_GRAYSCALE)
    return np.array(image, dtype=np.uint8)

def stack_integrals(integrals):
    images = [cv2.imread(f.as_posix(), cv2.IMREAD_GRAYSCALE) for f in integrals]
    return np.stack(images, axis=-1) # -> X,Y,C

def find_integrals(directory: Path):
    # find all png images in the folder
    files = [f for f in directory.iterdir() if (f.is_file() and f.name.endswith('.png'))]

    # match the given names
    planes = []
    for file in files:
        if file.stem in __FOCAL_LENGTHS:
            planes.append(file)

    # sort them ascending
    integrals = list(sorted(planes, key=lambda x: int(str(x.stem))))

    return integrals

def to_greyscale(img):
    return ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')

def to_pil(img):
    grayscale_img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
    image = Image.fromarray(grayscale_img.squeeze())
    return image
