'''Test file
Prepares focal stack from images
Loads model
predicts and saves as png
'''

import argparse
from pathlib import Path
from utils import prepare_samples, to_pil
import keras
from keras.models import Model
from PIL import Image

def combo_1024_512(pred, gt, integrals):
    blank_image = Image.new("L", (512*2, 512))
    mini_gt = to_pil(gt).resize((256,256), Image.Resampling.NEAREST)
    mini_fp_000 = to_pil(integrals[:,:,0]).resize((256,256), Image.Resampling.NEAREST)
    mini_fp_080 = to_pil(integrals[:,:,2]).resize((256,256), Image.Resampling.NEAREST)
    mini_fp_160 = to_pil(integrals[:,:,4]).resize((256,256), Image.Resampling.NEAREST)
    blank_image.paste(mini_gt, (0,0))
    blank_image.paste(mini_fp_000, (256,0))
    blank_image.paste(mini_fp_080, (0,256))
    blank_image.paste(mini_fp_160, (256,256))
    blank_image.paste(to_pil(pred), (512,0))
    return blank_image

def predict(in_dir: Path, out_dir: Path, make_combo=False):

    x, y, paths = prepare_samples(in_dir)

    #if focal_stack.shape[0:2] != (512,512):
    #    focal_stack = np.resize(focal_stack, (512,512,6))

    # build the model
    model: Model = keras.saving.load_model(Path(__file__).parent.resolve() / 'weights' / 'model.keras')

    # run predictions
    pred = model.predict_on_batch(x)
    for i, path in enumerate(paths):
        directory = out_dir / path.name
        directory.mkdir(parents=True, exist_ok=True)
        to_pil(pred[i].squeeze()).save(directory / f'prediction.png')
        if make_combo:
            combo_img = combo_1024_512(pred[i], y[i], x[i])
            combo_img.save(directory / f'combo_gt_000_080_160_pred.png')

def main():
    
    parser = argparse.ArgumentParser(
        prog='Predictor',
        description='Predict an all in focus image given focal stack as directory.',
        epilog="Focal Stack must contain ['00', '40', '80', '120', '160', '200']"
    )

    parser.add_argument('focal_stack_directory')
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('--combine', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.focal_stack_directory is None:
        raise ValueError('Must Provide focal stack')

    in_dir = Path(args.focal_stack_directory)

    if args.output_dir is None:
        out_dir = in_dir.parent / 'predictions'
    else:
        out_dir = Path(args.output_dir) 

    if not in_dir.exists():
        raise ValueError('Directory doesnt exist')
    
    out_dir.mkdir(exist_ok=True, parents=True)
    
    #scenario = '200_trees_idle'
    #in_dir = Path.cwd()/'submission'/'A9'/'code'/'test_data'/'test'/'images'/scenario/ 'sample_76'
    #out_dir = in_dir.parent.parent.parent  / 'predictions' / scenario
    predict(in_dir, out_dir, args.combine if args.combine is not None else False)

if __name__ == '__main__':
    main()
