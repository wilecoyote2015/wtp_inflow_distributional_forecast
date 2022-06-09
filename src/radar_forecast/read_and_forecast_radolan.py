import logging
import multiprocessing
import os

import cv2
import numpy as np
import pandas as pd
import wradlib as wrl
from tqdm import tqdm
from src.radar_forecast.helper_functions import read_input_file_wradlib, read_files_dir, get_roi, \
    get_idx_file_start_aligned, get_n_steps_past_present, make_predictions_rain_rasters_dir_compressed


# FIXME/REMARK/ATTENTION: There was a gap AT FIRST DATAPOINTS OF 2018.
#   Check this in the resulting time series and fix it.

def calc_flow(img_1, img_2):
    parameters = cv2.optflow.RLOFOpticalFlowParameter_create()

    win_size = int(
        20
    )
    parameters.setLargeWinSize(win_size)
    gridstep = 10

    img_1_uint = np.minimum(255, np.sqrt(np.round(img_1 * 255))).astype(np.uint8)
    img_2_uint = np.minimum(255, np.sqrt(np.round(img_2 * 255))).astype(np.uint8)
    flow = cv2.optflow.calcOpticalFlowDenseRLOF(
        cv2.cvtColor(img_1_uint, cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(img_2_uint, cv2.COLOR_GRAY2RGB),
        None,
        rlofParam=parameters,
        gridStep=(gridstep, gridstep),
        interp_type=cv2.optflow.INTERP_RIC,
        use_variational_refinement=False,
        forwardBackwardThreshold=1,
        use_post_proc=True
    )

    result = cv2.blur(flow, (50, 50))
    return result


def calc_flows_rasters(rasters, metadata):
    result = []
    # for each raster, calc flow relative to previous image for extrapolation
    print('Calculating Flows')
    for idx in tqdm(range(1, len(rasters))):
        result.append(calc_flow(rasters[idx - 1], rasters[idx]))
    return result, rasters[1:], metadata[1:]

# TODO: vectorize properly.
def convert_opt_flow_to_distortion_map(optical_flow):
    # get x and y resolution of optical flow (and so also of image)
    shape_optical_flow = optical_flow.shape[:-1]

    # create empty distortion maps for x and y separately because
    # opencv remap needs this
    distortion_map_x = np.zeros(shape_optical_flow, np.float32)  # only x and y
    distortion_map_y = np.zeros(shape_optical_flow, np.float32)  # only x and y

    # fill the distortion maps
    for x in range(shape_optical_flow[1]):
        distortion_map_x[:, x] = optical_flow[:, x, 0] + x
    for y in range(shape_optical_flow[0]):
        distortion_map_y[y] = optical_flow[y, :, 1] + y

    distortion_map = np.rollaxis(np.asarray([distortion_map_x, distortion_map_y]), 0, 3)

    return distortion_map


def forecast_rain(raster, flow, n_steps, remap_flow_reference_present):
    # optical flow is between past and present. hence, the movement vectors start at past.
    #   shift them to the present.
    if remap_flow_reference_present:
        distortion_map_flow = convert_opt_flow_to_distortion_map(-flow)
        flow = cv2.remap(flow, distortion_map_flow, None, cv2.INTER_LINEAR)

    # linear extrapolation
    flow_step = flow * n_steps

    distortion_map = convert_opt_flow_to_distortion_map(-flow_step)
    result = cv2.remap(raster, distortion_map, None, cv2.INTER_LINEAR)

    return result


def make_predictions_rain_rasters(filepaths, n_steps_predict, n_processes=4, rois=None):
    if rois is None:
        lats = [6.77, 7.0]
        lngs = [50.88, 51]

        rois = {
            'weiden': (lats, lngs)
        }

    grid_coordinates = wrl.georef.get_radolan_grid(
        *(read_input_file_wradlib(filepaths[0])[0].shape),
        wgs84=True
    )

    n_steps_interval = 3
    #  3 present and 3 past files are summed
    # n steps from first past to last present
    n_steps_past_present = get_n_steps_past_present(n_steps_interval)
    idx_start = get_idx_file_start_aligned(filepaths, n_steps_past_present)

    metas_last_present = []
    filenames_past, filenames_present = [], []
    indices, avgs_past, avgs_present = [], [], []
    print('Making predictions for files')

    print('Loading and averaging files')
    # generate average over the 15 min intervals for past and present data
    # for all data pairs
    for idx_first_file_past in tqdm(range(idx_start, len(filepaths) - n_steps_past_present, n_steps_interval)):
        # meta of last present
        metas_last_present.append(read_input_file_wradlib(filepaths[idx_first_file_past + n_steps_past_present])[1])

        rasters_past = []
        rasters_present = []
        for idx_step_avg in range(n_steps_interval):
            path_past = filepaths[idx_first_file_past + idx_step_avg]
            path_present = filepaths[idx_first_file_past + idx_step_avg + n_steps_interval]
            data_past, meta_ = read_input_file_wradlib(path_past)
            data_present, meta_ = read_input_file_wradlib(path_present)
            filenames_past.append(os.path.basename(path_past))
            filenames_present.append(os.path.basename(path_present))

            rasters_past.append(data_past)
            rasters_present.append(data_present)

        past_avg = average_rasters(rasters_past)
        present_avg = average_rasters(rasters_present)

        avgs_present.append(present_avg)
        avgs_past.append(past_avg)

    data_list = [
        (avg_past, avg_present, rois, grid_coordinates, n_steps_predict)
        for avg_past, avg_present in zip(avgs_past, avgs_present)
    ]

    if n_processes > 1:
        with multiprocessing.Pool(processes=4) as pool:

            results_tuples = pool.starmap(
                calc_mp,
                data_list
            )

    else:
        results_tuples = [
            calc_mp(
                *args
            )
            for args in data_list
        ]

    results = {
        key_roi: {
            # starts at 0 which is last known measurement.
            step_predict: [] for step_predict in range(n_steps_predict + 1)
        }
        for key_roi in rois.keys()
    }

    for forecasts_rois_timesteps in results_tuples:
        for key_roi, forecasts_timesteps in forecasts_rois_timesteps.items():
            for idx in range(0, n_steps_predict + 1):
                results[key_roi][idx].append(forecasts_timesteps[idx])

    dates_end_observation_window = [meta['datetime'] for meta in metas_last_present]

    dfs_results_rois = {
        key_roi: pd.DataFrame(results[key_roi], index=dates_end_observation_window)
        for key_roi in results.keys()
    }

    return filenames_past, filenames_present, dfs_results_rois


def average_rasters(rasters, n_intermediate_steps=5):
    rasters_intermediate = []
    steps_intermediate = np.linspace(0, 1, n_intermediate_steps + 2)[1:-1]
    for idx in range(len(rasters) - 1):
        rasters_intermediate.append(rasters[idx])
        flow = calc_flow(rasters[idx], rasters[idx + 1])
        for idx_step_intermediate in range(steps_intermediate.shape[0]):
            interpolation = forecast_rain(rasters[idx], flow, steps_intermediate[idx_step_intermediate], False)
            rasters_intermediate.append(interpolation)
    rasters_intermediate.append(rasters[-1])

    return np.mean(np.stack(rasters_intermediate, axis=0), axis=0)


def calc_mp(past_avg, present_avg, rois, grid_coordinates, n_steps_predict):
    result = {
        roi: [] for roi in rois.keys()
    }

    try:
        flow = calc_flow(past_avg, present_avg)
    except Exception as e:
        logging.error(f'Calculating flow failed: {e}. Assuming 0 rain')
        return [0.] * (n_steps_predict + 1)

    for key_roi, (lats, lngs) in rois.items():
        result[key_roi].append(get_roi(present_avg, lats, lngs, grid_coordinates))

    for idx_step in range(n_steps_predict):
        raster_forecast = forecast_rain(present_avg, flow, idx_step + 1, True)

        for key_roi, (lats, lngs) in rois.items():
            result[key_roi].append(get_roi(raster_forecast, lats, lngs, grid_coordinates))

    return result


def make_predictions_new_files_dir(date_last_prediction, path_dir, n_steps_predict, n_steps_interval=3, n_processes=1):
    """Used for MoVis in order to generate realtime predictions"""
    filepaths = sorted(os.listdir(path_dir))
    n_steps_past_present = get_n_steps_past_present(n_steps_interval)

    # get index of file corresponding to last prediction date
    idx_file_last_prediction_date = None
    for idx, filepath in enumerate(filepaths):
        meta = read_input_file_wradlib(filepaths[idx + n_steps_past_present])[1]
        date = meta['datetime']
        if date == date_last_prediction:
            idx_file_last_prediction_date = idx
            break
    if idx_file_last_prediction_date is None:
        raise ValueError(f'No matching file for start date {date_last_prediction}')

    # the present observation for first prediction is 15 min after the
    #   last prediction date (and the 15 min interval up to last prediction date is used as past data for optical flow
    #   calculation)
    idx_file_start = idx_file_last_prediction_date - 2

    filepaths_use = filepaths[idx_file_start:]

    dfs_results_rois = make_predictions_rain_rasters(
        filepaths_use,
        n_steps_predict,
        n_processes=n_processes
    )

    return dfs_results_rois

