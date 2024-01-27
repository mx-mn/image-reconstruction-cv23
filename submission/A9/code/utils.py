
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

__FOCAL_LENGTHS =  ['000', '040', '080', '120', '160', '200']

def prepare_samples(in_dir):
    if 'sample' in in_dir.name:
        target, focal_stack = prepare_sample(in_dir)
        samples, targets, paths = [focal_stack], [target], [in_dir]
    else:
        samples, targets, paths = [], [], []
        for sample_dir in in_dir.iterdir():
            if 'sample' not in sample_dir.name: continue
            target, focal_stack = prepare_sample(sample_dir)
            targets.append(target)
            samples.append(focal_stack)
            paths.append(sample_dir)

    if any(i is None for i in targets):
        y = None
    else:
        y = np.stack(targets)   
        y = y.astype(np.float32) / 255

    x = np.stack(samples)
    x = x.astype(np.float32) / 255
    return x,y, paths

def prepare_sample(sample_dir):
    target = fing_ground_truth(sample_dir)
    integrals = find_integrals(sample_dir)
    focal_stack = stack_integrals(integrals)
    return target, focal_stack

def fing_ground_truth(directory: Path):
    gt_path = [file for file in directory.iterdir() if 'gt' in file.name]
    if len(gt_path) == 0: return None

    image = cv2.imread(gt_path[0].as_posix(), cv2.IMREAD_GRAYSCALE)
    return np.array(image, dtype=np.uint8)

def stack_integrals(integrals):
    images = [cv2.imread(f.as_posix(), cv2.IMREAD_GRAYSCALE) for f in integrals]
    if images[0].shape != (512,512):
        images = [np.array(Image.fromarray(img).resize((512,512))) for img in images]
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
