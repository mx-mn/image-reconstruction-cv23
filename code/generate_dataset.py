from pathlib import Path
import cv2
import numpy as np

map_name_to_label = {'idle': 1, None: 0, 'sitting': 2, 'laying': 3}
map_label_to_name = [None, 'idle','sitting', 'laying']

def sort_by_number(path):
        return int(str(path.stem).split('_')[-1])

def extract_data_from_folders(directory, focal_stack_subset, crop):
    pose, trees = get_params(directory)
    no_person = (pose is None)

    image = open_gt_image(directory)

    integral_paths = find_integrals(directory, focal_stack_subset)
    integral_stack = stack_integrals(integral_paths)

    if not crop:
        x = integral_stack 
        y = np.array(image, dtype=np.uint8)

    else:
        x1, x2, y1, y2, cropped_gt_image = crop_image(image, no_person)
        x = integral_stack[x1:x2,y1:y2]
        y = cropped_gt_image
    
    pose_label = map_name_to_label[pose]
    return np.array(x), np.array(y), np.array(pose_label), np.array(trees)

def store_as_npz(target_path, x, y, pose, trees):
    np.savez(target_path, x=x, y=y, pose=pose, trees=trees)

def get_labels(dir):
    shapes = []
    for i, sample in enumerate(dir.iterdir()):
        if not sample.is_dir(): break
        shape, _ = get_params(sample)
        shapes += [shape]
        
    {name:i for i, name in enumerate(set(shapes))}

def find_integrals(dir: Path, subset=None):
    subdir = dir / 'integrals'
    if not subdir.exists(): return []

    if subset is None:
        integrals = list(subdir.iterdir())
    else:
        integrals = [f for f in subdir.iterdir() if f.name.split('_')[1].split('.')[0] in subset]
    
    integrals = sorted(integrals, key=lambda x: int(str(x.stem).split('_')[-1]))

    return integrals

def find_coordinates(grayscale_image):

    # Apply thresholding to highlight bright areas
    _, thresh = cv2.threshold(grayscale_image, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour which will be the person
    person_contour = max(contours, key=cv2.contourArea)

    # Calculate the bounding rectangle
    x, y, w, h = cv2.boundingRect(person_contour)

    # center of the rectangle
    x += w // 2
    y += h // 2

    return x, y 

def find_box(x, padding_top, padding_bottom, min_x, max_x):

    top = x + padding_top
    bottom = x - padding_bottom

    pixels_to_top = max_x - x
    pixels_to_bottom = x - min_x

    if pixels_to_top < padding_top:
        shift = (padding_top - pixels_to_top)
        top -= shift
        bottom -= shift

    elif pixels_to_bottom < padding_bottom:
        shift = (padding_bottom - pixels_to_bottom)
        top += shift
        bottom += shift

    return bottom, top

def crop_image(image, no_person, padding_top=64, padding_bottom=64, min_x=0, max_x=512):

    # random crop
    if no_person:
        x1 = np.random.randint(min_x, max_x - padding_bottom - padding_top)
        y1 = np.random.randint(min_x, max_x - padding_bottom - padding_top)
        x2 = x1 + padding_top+padding_bottom
        y2 = y1 + padding_top+padding_bottom

    # crop a rectangle around the person
    else:
        x,y = find_coordinates(image)
        x1, x2 = find_box(y, padding_top, padding_bottom, min_x, max_x)
        y1, y2 = find_box(x, padding_top, padding_bottom, min_x, max_x)

    crop = np.array(image[x1:x2, y1:y2])

    return x1, x2, y1, y2, crop

def stack_integrals(integrals):
    images = [cv2.imread(f.as_posix(), cv2.IMREAD_GRAYSCALE) for f in integrals]
    return np.stack(images, axis=-1) # -> X,Y,C
    return np.stack(images, axis=0)  # -> C,X,Y 

def get_params(dir: Path):
    params_path = [file for file in dir.iterdir() if 'Parameters' in file.name ][0]
    person_shape = None
    trees = None
    
    with open(params_path, 'r') as f:
        for line in  f.readlines():
            if 'person shape' in line:
                person_shape = line.split('=')[1].strip()
            
            if 'numbers of tree per ha' in line:
                n_trees = line.split('=')[1].strip()
                if not n_trees.isdigit():
                    print(line)
                else:
                    trees = int(n_trees)

            if trees is not None and person_shape is not None: break

    return person_shape, trees

def open_gt_image(dir: Path):
    gt_path = [file for file in dir.iterdir() if 'GT' in file.name ][0]
    image = cv2.imread(gt_path.as_posix(), cv2.IMREAD_GRAYSCALE)
    return image

def print_labels(labels):
    for v, c in zip(*np.unique(labels, return_counts=True)):
        print(f'{map_label_to_name[v]}({v}) : {c} times')

# Here you can put tests for the functions. 
# This will only be executed when you run this script directly
if __name__ == '__main__':



    raise ValueError(' update the calling of the conversion, like it is in the notebook.')
    import argparse
    
    parser = argparse.ArgumentParser(description='Process some parameters.')

    parser.add_argument('--batchsize', type=int, required=True,
                        help='The batch size for processing')
    parser.add_argument('--output', type=str, required=True,
                        help='Output filename')
    parser.add_argument('--basepath', type=str, required=True,
                        help='Base path for the data')
    parser.add_argument('--subset', nargs='+', required=True,
                        help='List of subset strings')

    # Parse the arguments
    args = parser.parse_args()
    print(f"Batch size: {args.batchsize}")
    print(f"Output folder: {args.output}")
    print(f"Base path: {args.basepath}")
    print(f"Subset: {args.subset}")
     
    path = Path(args.basepath) / args.output
    path.mkdir(parents=True, exist_ok=True)

    # TODO:  change this if needed

    for i, (x, y, pose, trees) in enumerate(folder_to_dataset(args.basepath, args.subset, crop=True, shuffle=True, batch_size=args.batchsize)):
        np.savez(path / f'sample_{i}', x=x, y=y)