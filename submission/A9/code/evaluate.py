'''Test file
Prepares focal stack from images
Loads model
predicts and saves as png
'''

import argparse
from pathlib import Path
import keras
import math
from keras.models import Model
from utils import prepare_samples

def evaluate(in_dir: Path):
    x, y, paths = prepare_samples(in_dir)

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
        )
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