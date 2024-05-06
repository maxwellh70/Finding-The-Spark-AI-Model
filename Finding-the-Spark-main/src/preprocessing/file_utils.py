"""
This module contains functions for loading satellite data from a directory of
tiles.
"""
from pathlib import Path
from typing import Tuple, List, Set
import os
from itertools import groupby
import re
from dataclasses import dataclass
import tifffile
import numpy as np


@dataclass
class Metadata:
    """
    A class to store metadata about a stack of satellite files from the same date.
    The attributes are the following:

    satellite_type: one of "viirs", "sentinel1", "sentinel2", "landsat", or "gt"
    file_name: a list of the original filenames of the satellite's bands
    tile_id: name of the tile directory, i.e., "Tile1", "Tile2", etc
    bands: a list of the names of the bands with correspondence to the
    indexes of the stack object, i.e. ["VH", "VV"] for sentinel-1
    time: time of the observations 遅くなってごめん
    """
    satellite_type: str
    file_name: List[str]
    tile_id: str
    bands: List[str]
    time: str

def process_viirs_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a VIIRS file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: DNB_VNP46A1_A2020221.tif
    Example output: ("2020221", "0")

    Parameters
    ----------
    filename : str
        The filename of the VIIRS file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    filename = Path(filename).stem
    pattern = "\d+$"

    found = re.search(pattern, filename)

    if found:
        return (found.group(0), "0")
    else:
        return ("0", "0")



def process_s1_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Sentinel-1 file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: S1A_IW_GRDH_20200804_VV.tif
    Example output: ("20200804", "VV")

    Parameters
    ----------
    filename : str
        The filename of the Sentinel-1 file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    filename = Path(filename).stem
    pattern = "(\d+)_([A-Z]{2})$"

    found = re.search(pattern, filename)

    if found:
        return (found.group(1), found.group(2))
    else:
        return ("0", "0")


def process_s2_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Sentinel-2 file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: L2A_20200816_B01.tif
    Example output: ("20200804", "01")

    Parameters
    ----------
    filename : str
        The filename of the Sentinel-2 file.

    Returns
    -------
    Tuple[str, str]
    """
    filename = Path(filename).stem
    pattern = "(\d+)_B([A-Z0-9]{2})$"

    found = re.search(pattern, filename)

    if found:
        return (found.group(1), found.group(2))
    else:
        return ("0", "0")


def process_landsat_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Landsat file and outputs
    a tuple containing two strings, in the format (date, band)

    Example input: LC08_L1TP_2020-08-30_B9.tif
    Example output: ("2020-08-30", "9")

    Parameters
    ----------
    filename : str
        The filename of the Landsat file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    filename = Path(filename).stem
    pattern = "(\d{4}-\d{2}-\d{2})_B(\d{1,2})$"

    found = re.search(pattern, filename)

    if found:
        return (found.group(1), found.group(2))
    else:
        return ("0", "0")

def process_ground_truth_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of the ground truth file and returns
    ("0", "0"), as there is only one ground truth file.

    Example input: groundTruth.tif
    Example output: ("0", "0")

    Parameters
    ----------
    filename: str
        The filename of the ground truth file though we will ignore it.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    return ("0", "0")


def get_satellite_files(tile_dir: Path, satellite_type: str) -> List[Path]:
    """
    Retrieve all satellite files matching the satellite type pattern.

    Parameters
    ----------
    tile_dir : Path
        The directory containing the satellite tiles.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    List[Path]
        A list of Path objects for each satellite file.
    """
    satTypeDict = {"viirs" : "DNB*", "sentinel1" : "S1A*", "sentinel2" : "L2A*",
        "landsat" : "LC08*", "gt" : "groundTruth.tif"}

    tile_dir = Path(tile_dir)
    files = list(tile_dir.glob(satTypeDict[satellite_type]))
    return files

def load_satellite(
        tile_dir: str | os.PathLike,
        satellite_type: str
        ) -> Tuple[np.ndarray, List[Metadata]]:
    """
    Load all bands for a given satellite type from a directory of tile files.

    Parameters
    ----------
    tile_dir : str or os.PathLike
        The Tile directory containing the satellite tiff files.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    Tuple[np.ndarray, List[Metadata]]
        A tuple containing the satellite data as a volume with
        dimensions (time, band, height, width) and a list of the filenames.
    """
    tile_dir = Path(tile_dir)
    sat_files = get_satellite_files(tile_dir, satellite_type)
    sat_data = read_satellite_files(sat_files)
    sat_data_stack, sat_filenames = stack_satellite_data(
        sat_data,
        sat_files,
        satellite_type
        )
    return sat_data_stack, sat_filenames

def get_satellite_files(tile_dir: Path, satellite_type: str) -> List[Path]:
    """
    Retrieve all satellite files matching the satellite type pattern.

    Parameters
    ----------
    tile_dir : Path
        The directory containing the satellite tiles.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    List[Path]
        A list of Path objects for each satellite file.
    """
    pattern = get_filename_pattern(satellite_type)
    return list(tile_dir.glob(pattern))


def get_filename_pattern(satellite_type: str) -> str:
    """
    Return the filename pattern for the given satellite type.

    Parameters
    ----------
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    str
        The filename pattern for the given satellite type.
    """
    patterns = {
        "viirs": 'DNB_VNP46A1_*',
        "sentinel1": 'S1A_IW_GRDH_*',
        "sentinel2": 'L2A_*',
        "landsat": 'LC08_L1TP_*',
        "gt": "groundTruth.tif"
    }
    return patterns[satellite_type]

def get_grouping_function(satellite_type: str):
    """
    Return the function to group satellite files by date and band.

    Parameters
    ----------
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    function
        The function to group satellite files by date and band.
    """
    grouping_functions = {
        "viirs": process_viirs_filename,
        "sentinel1": process_s1_filename,
        "sentinel2": process_s2_filename,
        "landsat": process_landsat_filename,
        "gt": process_ground_truth_filename
    }
    return grouping_functions[satellite_type]

def read_satellite_files(sat_files: List[Path]) -> List[np.ndarray]:
    """
    Read satellite files into a list of numpy arrays.

    Parameters
    ----------
    sat_files : List[Path]
        A list of Path objects for each satellite file.

    Returns
    -------
    List[np.ndarray]

    """
    sat_data_list = []
    for sat_file in sat_files:
        with tifffile.TiffFile(sat_file) as src:
            sat_data = src.asarray().astype(np.float32)
            sat_data_list.append(sat_data)
    return sat_data_list

def stack_satellite_data(
        sat_data: List[np.ndarray],
        file_names: List[str],
        satellite_type: str
        ) -> Tuple[np.ndarray, List[Metadata]]:
    """
    Stack satellite data into a single array and collect filenames.

    Parameters
    ----------
    sat_data : List[np.ndarray]
        A list of numpy arrays containing the satellite data.
    file_names : List[str]
        A dictionary containing multiple satellite filenames associated with
        the key being the date and band.

    Returns
    -------
    Tuple[np.ndarray, List[Metadata]
        A tuple containing the satellite data as a volume with dimensions
        (time, band, height, width) and a list of metadata since there are
        multiple timestamps per satellite.
    """
    # Get the function to group the data based on the satellite type
    grouping_fn = get_grouping_function(satellite_type)

    # Apply the grouping function to each file name to get the date and band
    date_bands = [grouping_fn(file_name.name) for file_name in file_names]

    # Sort the satellite data and file names based on the date and band
    sat_data = [x for _, x in sorted(zip(date_bands, sat_data))]
    file_names = [x for _, x in sorted(zip(date_bands, file_names))]
    date_bands = sorted(date_bands)

    # Initialize lists to store the stacked data and metadata
    date_band_stack = []
    sat_data_stack = []
    metadata_stack = []

    # Group the data by date
    for key, group in groupby(
        zip(date_bands, sat_data, file_names),
        key=lambda x: x[0][0]
    ):
        # Sort the group by band
        sorted_group = sorted(group)

        # Extract the date and band, satellite data, and file names from the
        # sorted group
        sorted_date_bands = [g[0] for g in sorted_group]
        grouped_sat_data = [g[1] for g in sorted_group]

        # Stack the satellite data along a new axis and append it to the list
        date_band_stack.append(sorted_date_bands)
        sat_data_stack.append(np.stack(grouped_sat_data, axis=0))

        # Create a Metadata object and append it to the list
        metadata_stack.append(Metadata(
            satellite_type=satellite_type,
            file_name=[g[2].name for g in sorted_group],
            tile_id=sorted_group[0][2].parent.name,
            bands=[g[0][1] for g in sorted_group],
            time=key
        ))

    # Stack the list of satellite data arrays along a new axis to create a
    # 4D array with dimensions (time, band, height, width)
    sat_data_stack = np.stack(sat_data_stack, axis=0)

    # Return the stacked satelliet data and the list Metadata objects.
    return sat_data_stack, metadata_stack