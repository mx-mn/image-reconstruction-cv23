'''Test file
Prepares focal stack from images
Loads model
predicts and saves as png
'''

import argparse
from pathlib import Path
import numpy as np
from utils import find_integrals, stack_integrals, to_pil
import keras
from keras.models import Model

def predict(in_dir: Path, out_dir: Path):

    # find the path to the integrals
    integrals = find_integrals(in_dir)

    # stack the integrals in a numpy array
    focal_stack: np.ndarray = stack_integrals(integrals) / 255

    if focal_stack.shape[0:2] != (512,512):
        focal_stack = np.resize(focal_stack, (512,512,6))

    focal_stack = np.expand_dims(focal_stack, axis=0)

    # build the model
    model: Model = keras.saving.load_model(Path(__file__).parent.resolve() / 'weights' / 'model.keras')

    # run predictions
    pred = model.predict_on_batch(focal_stack)

    # create png image and save to outdir
    to_pil(pred.squeeze()).save(out_dir / f'prediction.png')

def main():
    
    parser = argparse.ArgumentParser(
        prog='Predictor',
        description='Predict an all in focus image given focal stack as directory.',
        epilog="Focal Stack must contain ['00', '40', '80', '120', '160', '200']"
    )

    parser.add_argument('focal_stack_directory')
    parser.add_argument('--output_dir', required=False)
    args = parser.parse_args()

    if args.focal_stack_directory is None:
        raise ValueError('Must Provide focal stack')

    in_dir = Path(args.focal_stack_directory)

    if args.output_dir is None:
        out_dir = in_dir
    else:
        out_dir = Path(args.output_dir) 

    if not in_dir.exists():
        raise ValueError('Directory doesnt exist')
    
    out_dir.mkdir(exist_ok=True, parents=True)
    
    #in_dir = Path.cwd()/'submission'/'A9'/'code'/'data'/'real_focal_stack'
    #out_dir = in_dir
    
    predict(in_dir, out_dir)

if __name__ == '__main__':
    main()