import pyprojroot
import sys
import os
root = pyprojroot.here()
sys.path.append(str(root))
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
from typing import List
from dataclasses import dataclass
from pathlib import Path

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary
)

from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.preprocessing.subtile_esd_hw02 import Subtile
from src.visualization.restitch_plot import (
    restitch_eval,
    restitch_and_plot
)
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import tifffile
@dataclass
class EvalConfig:
    processed_dir: str | os.PathLike = root / 'data/processed/4x4'
    raw_dir: str | os.PathLike = root / 'data/raw/Train'
    results_dir: str | os.PathLike = root / 'data/predictions' / "FCNResnetTransfer"
    selected_bands: None = None
    tile_size_gt: int = 4
    batch_size: int = 8
    seed: int = 12378921
    num_workers: int = 11
    model_path: str | os.PathLike = root / "models" / "FCNResnetTransfer" / "last.ckpt"



def main(options):
    """
    Prepares datamodule and loads model, then runs the evaluation loop

    Inputs:
        options: EvalConfig
            options for the experiment
    """
    # Complete this function using the code snippets below. Do not forget to remove this line.
    # Load datamodule
    datamodule = ESDDataModule(processed_dir = options.processed_dir, raw_dir = options.raw_dir, batch_size = options.batch_size, num_workers = options.num_workers)
    datamodule.prepare_data()
    datamodule.setup()
    # load model from checkpoint at options.model_path
    model = ESDSegmentation.load_from_checkpoint(checkpoint_path = options.model_path)
    # set the model to evaluation mode (model.eval())
    model.eval()
    # this is important because if you don't do this, some layers
    # will not evaluate properly

    # instantiate pytorch lightning trainer
    pytorchTrainer = pl.Trainer()
    # run the validation loop with trainer.validate
    ranloop = pytorchTrainer.validate(model, datamodule = datamodule)[0]
    # run restitch_and_plot
    restitch_and_plot(ranloop, save_dir = options.results_dir, selected_bands = options.selected_bands, tile_size_gt = options.tile_size_gt)

    # for every subtile in options.processed_dir/Val/subtiles
    # run restitch_eval on that tile followed by picking the best scoring class
    # save the file as a tiff using tifffile
    # save the file as a png using matplotlib

    ##REMEMBER TO CODE THE TILE. TILE IS LIST OF EVERY SUBITLE IN options.processed_dir/Val/subtiles
    tiles = os.listdir(options.processed_dir / "Val" / "subtiles")
    
    for parent_tile_id in tiles:
        subtile_path = options.processed_dir / "Val" / "subtiles" / f"{parent_tile_id}.npy"
        subtile = np.load(subtile_path, allow_pickle = True).item()
        y_pred, y_true = restitch_eval(model, subtile, datamodule)

        #save the file as a tiff using tifffile
        tifffile.imsave(options.results_dir / f"{parent_tile_id}.tiff", y_pred.astype(np.uint8))

        # freebie: plots the predicted image as a jpeg with the correct colors
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(y_pred, vmin=-0.5, vmax=3.5,cmap=cmap)
        plt.savefig(options.results_dir / f"{parent_tile_id}.png")
    

if __name__ == '__main__':
    config = EvalConfig()
    parser = ArgumentParser()

    parser.add_argument("--model_path", type=str, help="Model path.", default=config.model_path)
    parser.add_argument("--raw_dir", type=str, default=config.raw_dir, help='Path to raw directory')
    parser.add_argument("-p", "--processed_dir", type=str, default=config.processed_dir,
                        help=".")
    parser.add_argument("--results_dir", type=str, default=config.results_dir, help="Results dir")
    main(EvalConfig(**parser.parse_args().__dict__))