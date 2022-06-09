import os
import tarfile
import tempfile
from datetime import datetime
import numpy as np
import wradlib as wrl


def read_input_file_wradlib(filepath):
    # fx_filename = wrl.util.get_wradlib_data_file(filepath)
    data, meta = wrl.io.read_radolan_composite(filepath)

    # remove invalid data (yes, they use magic numbers...) and scale
    data_array = np.asarray(data)
    data_array[data_array == -9999] = 0. # np.nan
    # data = np.ma.masked_equal(data, -9999) / 2 - 32.5

    return data_array, meta

def read_files_dir(path_dir, folder_extracted, fn_average_rasters, n_files=None, start=None):
    files = sorted(os.listdir(path_dir))
    if start is not None:
        files = files[start:]
    if n_files is not None:
        files = files[:n_files]
    paths = [os.path.join(folder_extracted, filename) for filename in files]


    rasters = []
    metadata = []
    for filepath in paths:
        data, meta = read_input_file_wradlib(filepath)
        rasters.append(data)
        metadata.append(meta)

    # files are 5 min. avg 15 min
    rasters_15 = []
    metadata_15 = []
    for idx_file in range(2, len(rasters), 3):
        # rasters_stacked = np.stack(rasters[idx_file-2:idx_file+1], axis=0)
        raster_15 = fn_average_rasters(rasters[idx_file-2:idx_file+1])
        # raster_15 = np.mean(rasters_stacked, axis=0)
        rasters_15.append(raster_15)
        metadata_15.append(metadata[idx_file])

    return rasters_15, metadata_15

def get_roi(raster, lats, lngs, grid_coordinates):

    # lats = [6.888428, 7.022495]
    # lngs = [50.960049, 50.916779]

    # indices_radius = np.argwhere()

    mask_roi = np.logical_and(
                np.logical_and(
                    np.greater(grid_coordinates[:, :, 0], lats[0]),
                    np.less(grid_coordinates[:, :, 0], lats[1])
                ),
                np.logical_and(
                    np.greater(grid_coordinates[:, :, 1], lngs[0]),
                    np.less(grid_coordinates[:, :, 1], lngs[1])
                ),
            )

    return np.sum(
        raster[
            mask_roi
        ]
    )

def get_n_steps_past_present(n_steps_interval):
    return 2*n_steps_interval - 1

def get_idx_file_start_aligned(filepaths, n_steps_history, alignment=(0, 15, 30, 45)):
    # ensure alignment to 15 min:
    #   date of a file is the end of (5 min) observation interval.
    #   find first start index so that the last file of each interval of 3 files to average is aligned to
    #   15 min interval.
    idx_present_first = None
    for idx_present, filepath_present in enumerate(filepaths[n_steps_history-1:]):
        meta_present = read_input_file_wradlib(filepath_present)[1]
        date_: datetime = meta_present['datetime']
        if date_.minute in alignment:
            idx_present_first = idx_present + n_steps_history - 1
            break

    if idx_present_first is None:
        raise ValueError(f'No start index found')

    return idx_present_first

def make_predictions_rain_rasters_dir(path_dir, n_steps_predict, filepath_output, fn_make_predictions_rain_rasters, n_files=None, start=None):
    files = sorted(os.listdir(path_dir))
    if start is not None:
        files = files[start:]
    if n_files is not None:
        files = files[:n_files]
    paths = [os.path.join(path_dir, filename) for filename in files]

    files_past, files_present, df_result = fn_make_predictions_rain_rasters(paths, n_steps_predict)
    df_result.to_csv(filepath_output)


def extract_dwd_tar(filepath, dir_output):
    print('Extracting')
    with tarfile.open(filepath, 'r:gz') as zip_:
        zip_.extractall(dir_output)
    print('extracted files')

    filenames = [f for f in sorted(os.listdir(dir_output))]
    filepaths = [os.path.join(dir_output, f) for f in filenames]

    return filepaths

def make_predictions_rain_rasters_dir_compressed(path_dir, n_steps_predict, fn_make_predictions_rain_rasters, n_files=None, filename_start=None, n_processes=1, n_files_history=4):
    files_past_last_iter = []

    # always decompress the current and the next file and then run prediction making for all paths
    # must compress current and next in order to ensure that
    paths_archives = sorted([
        os.path.join(path_dir, name)
        for name in os.listdir(path_dir)
        if name.endswith('.tar.gz')
    ])

    if filename_start is not None:
        paths_archives = paths_archives[paths_archives.index(os.path.join(path_dir, filename_start)):]
    if n_files is not None:
        paths_archives = paths_archives[:n_files]
    # dfs_results = []
    for idx in range(len(paths_archives)-1):
        print(f'predicting archive {paths_archives[idx]}')

        with tempfile.TemporaryDirectory() as tmpdirname:
            print('created temporary directory', tmpdirname)
            print('Extracting')
            with tarfile.open(paths_archives[idx], 'r:gz') as zip_:
                zip_.extractall(tmpdirname)
            with tarfile.open(paths_archives[idx+1], 'r:gz') as zip_:
                zip_.extractall(tmpdirname)
            print('extracted files')

            filenames = [f for f in sorted(os.listdir(tmpdirname)) if f not in files_past_last_iter]
            filepaths = [os.path.join(tmpdirname, f) for f in filenames]

            # all files except the ones needed as history for the next archive are not used for next archive.
            files_past_last_iter = filenames[:len(filenames)-n_files_history+1]

            result = fn_make_predictions_rain_rasters(filepaths, n_steps_predict, n_files_history, n_processes=n_processes)
            # dfs_results.append(df_result)

            # for filepath, (predictions_filepath, metas_predictions, meta_present) in predictions_filepaths.items():
            #     # save geotiff
            #     # This is the RADOLAN projection
            #     proj_osr = wrl.georef.create_osr("dwd-radolan")
            #
            #     # Get projected RADOLAN coordinates for corner definition
            #     xy_raw = wrl.georef.get_radolan_grid(900, 900)
            #
            #     for idx_step_forecast in range(predictions_filepath.shape[0]):
            #         data, xy = wrl.georef.set_raster_origin(predictions_filepath[idx_step_forecast], xy_raw, 'upper')
            #
            #         # create 3 bands
            #         data = np.stack((data, data + 100, data + 1000))
            #         ds = wrl.georef.create_raster_dataset(data, xy, projection=proj_osr)
            #
            #         filename_save = f"{meta_present['datetime'].isoformat()}_prediction_{metas_predictions[idx_step_forecast]['datetime'].isoformat()}.tif"
            #
            #         wrl.io.write_raster_dataset(os.path.join(dir_save_predictions, filename_save) + "geotiff.tif", ds, 'GTiff')

            return result


def save_geotiff(data, metadata, filepath, compress=True):
    # save geotiff
    # This is the RADOLAN projection
    proj_osr = wrl.georef.create_osr("dwd-radolan")
    # proj_osr = wrl.georef.create_osr("dwd-radolan")

    # Get projected RADOLAN coordinates for corner definition
    # xy_raw = wrl.georef.get_radolan_grid(data.shape[0], data.shape[1])
    xy_raw = wrl.georef.get_radolan_grid(data.shape[0], data.shape[1])
    # xy_raw = wrl.georef.get_radolan_grid(900, 900)

    data, xy = wrl.georef.set_raster_origin(data, xy_raw, 'upper')

    # create 3 bands
    # data = np.stack(
    #     (
    #         data,
    #         data + 100,
    #         data + 1000
    #      )
    # )
    ds = wrl.georef.create_raster_dataset(data, xy, projection=proj_osr)

    if compress:
        options = [
        'COMPRESS=DEFLATE',
        'PREDICTOR=2'
    ]
    else:
        options = []

    wrl.io.write_raster_dataset(filepath, ds, 'GTiff', options=options
                                )

