from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz

import logging


def fix_daylight_saving(data):
    result = data.copy()
    result.index = result.index.tz_localize('UTC')
    tz_local = pytz.timezone('Europe/Berlin')

    # TODO: detect from data via timezone conversion
    times_summer_to_winter = [
        datetime(2021, 10, 31, 3, 00, tzinfo=tz_local),
        datetime(2020, 10, 25, 3, 00, tzinfo=tz_local),
        datetime(2019, 10, 27, 3, 00, tzinfo=tz_local),
        datetime(2018, 10, 28, 3, 00, tzinfo=tz_local),
        datetime(2017, 10, 29, 3, 00, tzinfo=tz_local),
        datetime(2016, 10, 30, 3, 00, tzinfo=tz_local),
    ]

    times_winter_to_summer = [
        datetime(2021, 3, 28, 2, 00, tzinfo=tz_local),
        datetime(2020, 3, 29, 2, 00, tzinfo=tz_local),
        datetime(2019, 3, 31, 2, 00, tzinfo=tz_local),
        datetime(2018, 3, 25, 2, 00, tzinfo=tz_local),
        datetime(2017, 3, 26, 2, 00, tzinfo=tz_local),
        datetime(2016, 3, 27, 2, 00, tzinfo=tz_local),
    ]

    # remove 4 timesteps when from summer to winter
    for time_transition in times_summer_to_winter:

        # if time_transition in data_all.index:
        drop_start = time_transition
        drop_end = time_transition + timedelta(hours=1)
        result  = pd.concat(
            [
                result.loc[result.index < drop_start],
                result.loc[result.index >= drop_end]
            ]
        )

    # interpolate 4 timesteps from winter to summer
    for time_transition in times_winter_to_summer:

        # if time_transition in data_all.index:
        df_before_transition = result.loc[result.index < time_transition]
        df_fill = pd.concat([
            df_before_transition.iloc[-1:],
            df_before_transition.iloc[-1:],
            df_before_transition.iloc[-1:],
            df_before_transition.iloc[-1:],
        ]
        )

        result = pd.concat(
            [
                df_before_transition,
                df_fill,
                result.loc[result.index >= time_transition]
            ]
        )

    stepsize = data.index[1] - data.index[0]
    index = [data.index[0] + stepsize * idx for idx in range(len(result))]

    # create new index with fake-utc-dates
    result.index = pd.DatetimeIndex(index, tz=pytz.utc)

    return result


def get_median_seasonal(data, name_column, period):
    num_seasons = int(np.floor(len(data) / period))

    data_sliced = data.iloc[:num_seasons * period]
    column_sliced = data_sliced[name_column].to_numpy()
    data_column_seasons = np.reshape(
        column_sliced,
        (num_seasons, period)
    ).transpose()

    medians = np.median(data_column_seasons, axis=1)

    return medians

def preprocess_data_realtime(
        data: pd.DataFrame,
        do_fix_daylight_saving=True,
        thresholds_columns_clip=None,
        thresholds_columns_ffill=None,
        thresholds_columns_fill_nan=None,
        # dict. values are tuples with (threshold of last value from which the detection works, min drop height to detect as drop)
        #   used to ffill level measurement drops in periods of high inflow.
        thresholds_max_diff_columns_ffill_drops=None,
        # dict. values are tuples with (
        #   threshold of last value from which the detection works,
        #   max difference to last datapoint so that it counts as drop
        #   )
        #   used to insert nas for level measurement drops in periods of high inflow.
        thresholds_max_diff_columns_ffill_na_drops=None,
        cols_fill_constant_nan=None,
        # keys are values to fill (e.g. means)
        cols_fill_constant_value=None,
        fill_nans=True,
        **kwargs
):

    # verify that timeseries has no missing datapoints
    diff_index = np.diff(data.index)
    min_diff = np.min(diff_index)
    max_diff = np.max(diff_index)

    if min_diff != max_diff:
        logging.error(f'inconsistent timesteps detected: {min_diff} != {max_diff}')

    if not fill_nans:
        raise NotImplementedError

    # if cols_fill_constant_value is not None:
    #     raise NotImplementedError

    if thresholds_max_diff_columns_ffill_na_drops is not None:
        raise NotImplementedError
    if thresholds_columns_fill_nan is not None:
        raise NotImplementedError

    if do_fix_daylight_saving:
        data_processed = fix_daylight_saving(data)
    else:
        data_processed = data.copy(True)

    # get booleans of constant phases to insert nans.
    #   must be done here before any other filling occurs, as cosntant values / ffill of other data
    #   may also result in constant regions (that shall stay), but would be detected and
    #   overwritten by nans here.
    bools_constant_for_nans = {}
    for name_column, window_size in cols_fill_constant_nan.items():
        if name_column not in data:
            continue

        data_column = data_processed[name_column]

        no_change = np.full_like(data_column, fill_value=True, dtype=np.bool)
        for shift in range(1, window_size + 1):
            diff = data_column.diff(periods=shift)
            no_change = np.logical_and(no_change, diff == 0.)

        bools_constant_for_nans[name_column] = no_change

        # data_processed[name_column][no_change] = np.nan

    # set constant periods value
    for name_column, (window_size, value) in cols_fill_constant_value.items():
        if name_column not in data:
            continue

        data_column = data_processed[name_column]

        no_change = np.full_like(data_column, fill_value=True, dtype=np.bool)
        for shift in range(1, window_size + 1):
            diff = data_column.diff(periods=shift)
            no_change = np.logical_and(no_change, diff == 0.)

        data_processed[name_column][no_change] = value

    # forward fill
    for name_column, (min_, max_) in thresholds_columns_ffill.items():
        if name_column not in data:
            continue
        bool_min = data_processed[name_column] < min_
        bool_max = data_processed[name_column] > max_

        data_processed[name_column][np.logical_or(bool_min, bool_max)] = np.nan
    # drops
    for name_column, (threshold, max_diff) in thresholds_max_diff_columns_ffill_drops:
        if name_column not in data:
            continue
        data_column = data_processed[name_column]
        bool_last_point_high = np.roll(data_column, 1) > threshold
        diff = data_column.diff()
        bool_large_drop = diff < max_diff
        data_processed[name_column][np.logical_and(bool_last_point_high, bool_large_drop)] = np.nan

    # fill the values set to nan (and nans in raw data)
    data_processed = data_processed.fillna(method='ffill')

    # insert nans for constant areas detected previously
    for name_column, bool_constant in bools_constant_for_nans.items():
        data_processed[name_column][bool_constant] = np.nan

    # clip columns
    for name_column, (min_, max_) in thresholds_columns_clip.items():
        if name_column not in data:
            continue
        data_processed[name_column] = data_processed[name_column].clip(min_, max_)

    return data_processed

