'''Test file
Prepares focal stack from images
Loads model
predicts and saves as png
'''

import argparse
from pathlib import Path
import numpy as np
import keras
import math
from keras.models import Model
from utils import find_integrals, stack_integrals, fing_ground_truth

def prepare_sample(sample_dir):
    target = fing_ground_truth(sample_dir)
    integrals = find_integrals(sample_dir)
    focal_stack = stack_integrals(integrals)
    return target, focal_stack

def evaluate(in_dir: Path):
    if 'sample' in in_dir.name:
        target, focal_stack = prepare_sample(in_dir)
        samples, targets = [focal_stack], [target]
    else:
        samples, targets = [], []
        for sample_dir in in_dir.iterdir():
            if 'sample' not in sample_dir.name: continue
            target, focal_stack = prepare_sample(sample_dir)
            targets.append(target)
            samples.append(focal_stack)
        
    x = np.stack(samples)
    y = np.stack(targets)
    x = x.astype(np.float32) / 255
    y = y.astype(np.float32) / 255
    
    model: Model = keras.saving.load_model(Path(__file__).parent.resolve() / 'weights' / 'model.keras')
    model.compile(loss='mean_squared_error')

    mse = model.evaluate(x,y)
    manual_psnr = -10*math.log10(mse)

    print(f'MSE  : {mse:.5f}')
    print(f'PSNR : {manual_psnr:.5f} dB')

def main():
    
    parser = argparse.ArgumentParser(
        prog='Evaluator',
        description=(
            'Evaluate all samples in the directory. '
            'The directory must contain a full focal stack and a ground truth image.'
            
            ),
    )

    parser.add_argument('directory')
    args = parser.parse_args()
    in_dir = Path(args.directory)
    if not in_dir.exists():
        raise ValueError('Directory doesnt exist')
    
    #in_dir = Path.cwd()/'data'/'test'/'cases'/'images'/'100_trees_idle'
    #in_dir = Path.cwd()/'data'/'validation'/'crop'/'images'
    
    evaluate(in_dir)

if __name__ == '__main__':
    main()