import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from src.preprocessing.subtile_esd_hw02 import TileMetadata, Subtile
import torch
import pytorch_lightning as pl

def restitch_and_plot(options, datamodule, model, parent_tile_id, satellite_type="sentinel2", rgb_bands=[3,2,1], image_dir: None | str | os.PathLike = None):
    """
    Plots the 1) rgb satellite image 2) ground truth 3) model prediction in one row.

    Args:
        options: EvalConfig
        datamodule: ESDDataModule
        model: ESDSegmentation
        parent_tile_id: str
        satellite_type: str
        rgb_bands: List[int]
    """
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)

    

    fig, axs = plt.subplots(nrows=1, ncols=3)

    datamodule.setup('eval')
    temp = datamodule.val_dataloader()
    itered = next(iter(temp))
    sat_img, mask, metadata = itered

    # Run model prediction
    model.eval()
    with torch.no_grad():
        pred = model.forward(sat_img)

    # Plot the images
    im_axis0 = axs[0].imshow(sat_img.squeeze().permute(1, 2, 0))
    # make sure to use cmap=cmap, vmin=-0.5 and vmax=3.5 when running
    im_axis1= axs[1].imshow(mask.squeeze(), cmap = cmap, vmin = -0.5, vmax = 3.5)
    # axs[i].imshow on the 1d images in order to have the correct 
    # colormap for the images.
    # On one of the 1d images' axs[i].imshow, make sure to save its output as 
    # `im`, i.e, im = axs[i].imshow
    im = axs[2].imshow(pred.squeeze().argmax(dim=0), cmap=cmap, vmin=-0.5, vmax=3.5)

    
    
    
    # The following lines sets up the colorbar to the right of the images    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0,1,2,3])
    cbar.set_ticklabels(['Sttlmnts Wo Elec', 'No Sttlmnts Wo Elec', 'Sttlmnts W Elec', 'No Sttlmnts W Elec'])
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "restitched_visible_gt_predction.png")
        plt.close()


def restitch_eval(dir: str | os.PathLike, satellite_type: str, tile_id: str, range_x: Tuple[int, int], range_y: Tuple[int, int], datamodule, model) -> np.ndarray:
    """
    Given a directory of processed subtiles, a tile_id and a satellite_type, 
    this function will retrieve the tiles between (range_x[0],range_y[0])
    and (range_x[1],range_y[1]) in order to stitch them together to their
    original image. It will also get the tiles from the datamodule, evaluate
    it with model, and stitch the ground truth and predictions together.

    Input:
        dir: str | os.PathLike
            Directory where the subtiles are saved
        satellite_type: str
            Satellite type that will be stitched
        tile_id: str
            Tile id that will be stitched
        range_x: Tuple[int, int]
            Range of tiles that will be stitched on width dimension [0,5)]
        range_y: Tuple[int, int]
            Range of tiles that will be stitched on height dimension
        datamodule: pytorch_lightning.LightningDataModule
            Datamodule with the dataset
        model: pytorch_lightning.LightningModule
            LightningModule that will be evaluated
    
    Output:
        stitched_image: np.ndarray
            Stitched image, of shape (time, bands, width, height)
        stitched_ground_truth: np.ndarray
            Stitched ground truth of shape (width, height)
        stitched_prediction_subtile: np.ndarray
            Stitched predictions of shape (width, height)
    """
    
    dir = Path(dir)
    satellite_subtile = []
    ground_truth_subtile = []
    predictions_subtile = []
    satellite_metadata_from_subtile = []
    for i in range(*range_x):
        satellite_subtile_row = []
        ground_truth_subtile_row = []
        predictions_subtile_row = []
        satellite_metadata_from_subtile_row = []
        for j in range(*range_y):
            subtile = Subtile().load(dir / 'subtiles' / f"{tile_id}_{i}_{j}.npz")
            # raise NotImplementedError # Complete the function in the lines below using the codes
            # snippets provided. Do NOT forget to remove the raise NotImplementedError line.
            # find the tile in the datamodule

            # evaluate the tile with the model
            
            # You need to add a dimension of size 1 at dim 0 so that
            # some CNN layers work
            # i.e., (batch_size, channels, width, height) with batch_size = 1
            # make sure that the tile is in GPU memory, i.e., X = X.cuda()

            # convert y to numpy array
            # detach predictions from the gradient, move to cpu and convert to numpy

            ground_truth_subtile_row.append(y)
            predictions_subtile_row.append(predictions)
            satellite_subtile_row.append(subtile.satellite_stack[satellite_type])
            satellite_metadata_from_subtile_row.append(subtile.tile_metadata)
        ground_truth_subtile.append(np.concatenate(ground_truth_subtile_row, axis=-1))
        predictions_subtile.append(np.concatenate(predictions_subtile_row, axis=-1))
        satellite_subtile.append(np.concatenate(satellite_subtile_row, axis=-1))
        satellite_metadata_from_subtile.append(satellite_metadata_from_subtile_row)
    return np.concatenate(satellite_subtile, axis=-2), np.concatenate(ground_truth_subtile, axis=-2), np.concatenate(predictions_subtile, axis=-2)
