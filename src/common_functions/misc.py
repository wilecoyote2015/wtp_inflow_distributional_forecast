import json
import logging
from collections import OrderedDict
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.config_paths import *


KEY_TENSOR = 'TENSOR'
KEY_DATETIME = 'DATETIME_'
KEY_DATETIME64 = 'DATETIME64_'
KEY_TIMEDELTA64 = 'TIMEDELTA64_'
KEY_TENSOR_SCALAR = 'TENSOR_SCALAR'
KEY_TUPLE_KEY = 'TUPLE_'
KEY_DATETIME_KEY = 'DATETIME_'
from src.constants.misc  import *
import pandas as pd
from copy import deepcopy
import gzip
from bspline import Bspline


def dump_json_gzip(path_write_json, value):
    path_ = f'{path_write_json}.gz'
    # print(f'writing compressed json to {path_}')
    with gzip.open(path_, 'wt') as f:
        f.write(json.dumps(value, indent=4))


def load_json_gzip(path_file):
    # print(f'Extracting {path_file}')
    path_file_gz = f'{path_file}.gz' if not path_file.endswith('.gz') else path_file
    with gzip.open(path_file_gz, 'r') as f:
        result = json.load(f)
    # print(f'Finished extracting {path_file}')
    return result


def load_str_gzip(path_file):
    path_file_gz = f'{path_file}.gz' if not path_file.endswith('.gz') else path_file

    with gzip.open(path_file_gz, 'rt') as f:
        result = f.read()

    return result


def fn_cap_rain(data):
    # max_ = np.quantile(data[data > 0.], 0.98)
    # use fixed value for consistency throughtout datasets
    max_ = 0.085
    result = np.clip(data, None, max_)
    return result


def fn_cap_rain_forecast(data):
    # max_ = np.quantile(data[data > 0.], 0.98)
    # use fixed value for consistency throughtout datasets
    max_ = 100
    result = np.clip(data, None, max_)
    return result


def make_dir_if_missing(path_dir):
    logging.info(f'Making directory {path_dir}')
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)


def json_to_params(params, func_array):
    if isinstance(params, (list, tuple)):
        if len(params) > 0 and params[0] == KEY_TENSOR:
            result = func_array(
                params[1:]
            )
        elif len(params) > 0 and params[0] == KEY_TENSOR_SCALAR:
            result = func_array(
                params[1]
            )
        else:
            result = []
            for param in params:
                result.append(json_to_params(param, func_array))
    elif isinstance(params, (dict, OrderedDict)):
        result = OrderedDict()
        for key, param in params.items():
            if isinstance(key, str) and key.startswith(KEY_TUPLE_KEY):
                key = eval(key[len(KEY_TUPLE_KEY):])
            if isinstance(key, str) and key.startswith(KEY_DATETIME_KEY):
                key = datetime.fromisoformat(key[len(KEY_DATETIME_KEY):])
            result[key] = json_to_params(param, func_array)
    elif isinstance(params, (tf.Tensor, tf.Variable, np.ndarray)):
        result = func_array(params)

    elif isinstance(params, str):
        if params.startswith(KEY_DATETIME):
            return datetime.fromisoformat(params[len(KEY_DATETIME):])
        if params.startswith(KEY_DATETIME64):
            int_ = int(params[len(KEY_DATETIME64):])
            return np.datetime64(int_, 's')
        elif params.startswith(KEY_TIMEDELTA64):
            int_ = int(params[len(KEY_TIMEDELTA64):])
            return np.timedelta64(int_, 'ns')
        else:
            return params

    elif isinstance(params, (str, float, int, bool)):
        result = params
    elif params is None:
        result = params


    else:
        raise ValueError(f'Cannot convert params: {params}')

    return result


def params_to_json(params):
    if isinstance(params, (list, tuple)):
        result = []
        for param in params:
            result.append(params_to_json(param))
    elif isinstance(params, (dict, OrderedDict)):
        result = OrderedDict()
        for key, param in params.items():
            # tuples are no valid key
            if isinstance(key, tuple):
                key = f'{KEY_TUPLE_KEY}{str(key)}'
            elif isinstance(key, datetime):
                key = f'{KEY_DATETIME_KEY}{key.isoformat()}'
            result[key] = params_to_json(param)
    elif isinstance(params, (tf.Tensor, tf.Variable)):
        if len(params.shape) > 0:
            result = params_to_json(params.numpy())
        else:
            result = [KEY_TENSOR_SCALAR, params_to_json(params.numpy())]
    elif isinstance(params, np.ndarray):
        result = [KEY_TENSOR, *params.tolist()]
    elif isinstance(params, np.datetime64):
        result = f'{KEY_DATETIME64}{np.datetime64(params, "s").astype(int)}'
    elif isinstance(params, np.timedelta64):
        result = f'{KEY_TIMEDELTA64}{np.timedelta64(params, "ns").astype(int)}'
    elif isinstance(params, datetime):
        result = f'{KEY_DATETIME}{params.isoformat()}'
    elif isinstance(params, (np.float32, np.float64)):
        result = params.item()
    elif isinstance(params, (pd.Series, pd.DataFrame)):
        result = params.to_dict()
    elif isinstance(params, np.bool_):
        result = bool(params)
    else:
        result = params

    return result


def load_model_json_file(func_load_model_json_, data, path_file, datetime_first, timestep_data, dtype):
    with open(path_file) as f:
        params = json.load(f)

    # TODO: more robust way to get timedelta
    datetime_reference = pd.Timestamp(params[DATETIME_REFERENCE])

    result = func_load_model_json_(
        data,
        params[DATA_PASS_MODEL],
        datetime_reference,
        datetime_first,
        timestep_data,
        dtype
    )

    return result


def train_model_json_file(
        func_train_model_json,
        data: [tf.Tensor],
        path_model_start,
        datetime_first,
        timestep_data,
        dtype,
        n_iters,
        lambdas_lasso,
        optimizer, **kwargs_train
):
    # TODO: append generated trace to existing trace

    with open(path_model_start) as f:
        params = json.load(f)

    # TODO: more robust way to get timedelta
    datetime_reference = pd.Timestamp(params[DATETIME_REFERENCE])

    return func_train_model_json(
        data,
        params[DATA_PASS_MODEL],
        datetime_reference,
        datetime_first,
        timestep_data,
        dtype,
        n_iters,
        lambdas_lasso,
        optimizer,
        **kwargs_train
    )


def shrink_model(path_model, path_output, threshold_ar):
    with open(path_model) as f:
        params = json.load(f)

    params_ar_shrunken, lags_shrunken = shrink_params(
        params[DATA_PASS_MODEL][PARAMS][0][0],
        params[DATA_PASS_MODEL][PARAMS][0][1],
        params[DATA_PASS_MODEL][PARAMS][0][2],
        params[DATA_PASS_MODEL][LAGS_PREDICTOR],
        params[DATA_PASS_MODEL][LAGS_EXTERNAL],
        params[DATA_PASS_MODEL][LAGS_ORACLES],
        threshold_ar
    )

    result = deepcopy(params)

    result[DATA_PASS_MODEL][PARAMS][0] = [*params_ar_shrunken, *params[DATA_PASS_MODEL][PARAMS][0][3:]]
    result[DATA_PASS_MODEL][LAGS_PREDICTOR] = lags_shrunken[0]
    result[DATA_PASS_MODEL][LAGS_EXTERNAL] = lags_shrunken[1]
    result[DATA_PASS_MODEL][LAGS_ORACLES] = lags_shrunken[2]

    with open(path_output, 'w') as f_out:
        json.dump(result, f_out, indent=2)


def shrink_params(
        params_ar_predictor,
        params_ar_external,
        params_ar_oracles,
        lags_predictor,
        lags_external,
        lags_oracles,
        threshold_ar
):
    result_ar_predictor = []
    result_ar_external = []
    result_ar_oracles = []
    result_lags_predictor = []
    result_lags_external = []
    result_lags_oracles = []

    n_params_start = 0
    n_params_result = 0

    for idx_timestep in range(len(params_ar_predictor)):
        lags_predictor_timestep = []
        lags_external_timestep = []
        lags_oracles_timestep = []

        ar_predictor_timestep = []
        ar_external_timestep = []
        ar_oracles_timestep = []
        for idx_param_pdf in range(len(params_ar_predictor[idx_timestep])):
            lags_predictor_param = []
            lags_external_param = []
            lags_oracles_param = []

            ar_predictor_param = [KEY_TENSOR, ]
            ar_external_param = [KEY_TENSOR, ]
            ar_oracles_param = [KEY_TENSOR, ]

            idx_param = 1
            for idx_threshold in range(len(lags_predictor[idx_timestep][idx_param_pdf])):
                lags_predictor_threshold = [KEY_TENSOR, ]
                for lag_threshold in lags_predictor[idx_timestep][idx_param_pdf][idx_threshold][1:]:

                    param = params_ar_predictor[idx_timestep][idx_param_pdf][idx_param]
                    if abs(param) >= threshold_ar:
                        ar_predictor_param.append(param)
                        lags_predictor_threshold.append(lag_threshold)
                        n_params_result += 1
                    idx_param += 1
                    n_params_start += 1
                lags_predictor_param.append(lags_predictor_threshold)
            lags_predictor_timestep.append(lags_predictor_param)
            ar_predictor_timestep.append(ar_predictor_param)

            idx_param = 1
            for idx_column in range(len(lags_external[idx_timestep][idx_param_pdf])):
                lags_external_column = []
                for idx_threshold in range(len(lags_external[idx_timestep][idx_param_pdf][idx_column])):
                    lags_external_threshold = [KEY_TENSOR]
                    for lag_threshold in lags_external[idx_timestep][idx_param_pdf][idx_column][idx_threshold][1:]:

                        param = params_ar_external[idx_timestep][idx_param_pdf][idx_param]
                        if abs(param) >= threshold_ar:
                            ar_external_param.append(param)
                            lags_external_threshold.append(lag_threshold)
                            n_params_result += 1
                        idx_param += 1
                        n_params_start += 1
                    lags_external_column.append(lags_external_threshold)
                lags_external_param.append(lags_external_column)
            lags_external_timestep.append(lags_external_param)
            ar_external_timestep.append(ar_external_param)

            idx_param = 1
            for idx_column in range(len(lags_oracles[idx_timestep][idx_param_pdf])):
                lags_oracles_column = []
                for idx_threshold in range(len(lags_oracles[idx_timestep][idx_param_pdf][idx_column])):
                    lags_oracles_threshold = [KEY_TENSOR]
                    for lag_threshold in lags_oracles[idx_timestep][idx_param_pdf][idx_column][idx_threshold][1:]:

                        param = params_ar_oracles[idx_timestep][idx_param_pdf][idx_param]
                        if abs(param) >= threshold_ar:
                            ar_oracles_param.append(param)
                            lags_oracles_threshold.append(lag_threshold)
                            n_params_result += 1
                        idx_param += 1
                        n_params_start += 1
                    lags_oracles_column.append(lags_oracles_threshold)
                lags_oracles_param.append(lags_oracles_column)
            lags_oracles_timestep.append(lags_oracles_param)
            ar_oracles_timestep.append(ar_oracles_param)

        result_ar_predictor.append(ar_predictor_timestep)
        result_ar_external.append(ar_external_timestep)
        result_ar_oracles.append(ar_oracles_timestep)

        result_lags_predictor.append(lags_predictor_timestep)
        result_lags_external.append(lags_external_timestep)
        result_lags_oracles.append(lags_oracles_timestep)

        print(f'n params start: {n_params_start}')
        print(f'n params shrunken: {n_params_result}')

    return (
        (
            result_ar_predictor,
            result_ar_external,
            result_ar_oracles
        ),
        (
            result_lags_predictor,
            result_lags_external,
            result_lags_oracles
        )
    )


def count_params_model_file(filepath):
    with open(filepath) as f:
        params = json.load(f)

    params_json = json_to_params(params, np.asarray)

    return count_params(params_json[DATA_PASS_MODEL][PARAMS])


def count_params(params):
    # if params are ndarray, return size
    if isinstance(params, (np.ndarray, tf.Tensor)):
        return np.size(params)

    # if params are number, return 1
    if isinstance(params, (float, int)):
        return 1

    # if params are list or tuple, call function again for all elements
    if isinstance(params, (list, tuple)):
        result = 0
        for elemn in params:
            result += count_params(elemn)
        return result

    else:
        logging.warning(f'parameter {params} not recognized for counting')
        return 0


def calc_avg_likelihood_json_file(func_load_model_file, filepath, data_tf, datetime_first, timestep_data, dtype,
                                  model=None):
    with open(filepath) as f:
        params = json.load(f)

    f_constant_int = lambda x: tf.constant(tf.cast(x, tf.int32))

    if model is None:
        n_timesteps_predict = json_to_params(params[DATA_PASS_MODEL][N_TIMESTEPS_PREDICT], f_constant_int)
        model = func_load_model_file(
            [dataset[:-n_timesteps_predict] for dataset in data_tf],
            filepath,
            datetime_first,
            timestep_data,
            dtype
        )

    max_lag_wrt_first_step_prediction = get_max_lag_wrt_first_step_prediction(
        # No need to respect oracle lags as lags are always in the future
        json_to_params(params[DATA_PASS_MODEL][LAGS_PREDICTOR], f_constant_int),
        json_to_params(params[DATA_PASS_MODEL][LAGS_EXTERNAL], f_constant_int),
    )

    y_tuple = make_y_tuple(
        data_tf[0],
        max_lag_wrt_first_step_prediction,
        json_to_params(params[DATA_PASS_MODEL][N_TIMESTEPS_PREDICT], f_constant_int)
    )

    log_likelihood = tf.reduce_sum(model.log_prob(y_tuple))

    return log_likelihood / (y_tuple[0].shape[0] * len(y_tuple))


def calc_bic_json_file(func_load_model_file, filepath, data_tf, datetime_first, timestep_data, dtype, model=None):
    with open(filepath) as f:
        params = json.load(f)

    f_constant_int = lambda x: tf.constant(tf.cast(x, tf.int32))

    if model is None:
        n_timesteps_predict = json_to_params(params[DATA_PASS_MODEL][N_TIMESTEPS_PREDICT], f_constant_int)
        model = func_load_model_file(
            [dataset[:-n_timesteps_predict] for dataset in data_tf],
            filepath,
            datetime_first,
            timestep_data,
            dtype
        )

    max_lag_wrt_first_step_prediction = get_max_lag_wrt_first_step_prediction(
        # No need to respect oracle lags as lags are always in the future
        json_to_params(params[DATA_PASS_MODEL][LAGS_PREDICTOR], f_constant_int),
        json_to_params(params[DATA_PASS_MODEL][LAGS_EXTERNAL], f_constant_int),
    )

    y_tuple = make_y_tuple(
        data_tf[0],
        max_lag_wrt_first_step_prediction,
        json_to_params(params[DATA_PASS_MODEL][N_TIMESTEPS_PREDICT], f_constant_int)
    )

    log_likelihood = tf.reduce_sum(model.log_prob(y_tuple))

    n_parameters = count_params_model_file(filepath)
    n_observations = data_tf[0].shape[0]

    logging.info(f'bic: found {n_parameters} params')
    logging.info(f'bic: found {n_observations} observations')

    return n_parameters * np.log(n_observations) - 2 * log_likelihood


def get_max_lag_wrt_first_step_prediction(
        # No need to respect oracle lags as lags are always in the future
        lags_predictor,
        lags_external
):
    result = tf.constant(1)
    for idx_timestep, (lags_predictor_timestep, lags_external_timestep) in enumerate(
            zip(lags_predictor, lags_external)):

        for lags_predictor_param, lags_external_param in zip(lags_predictor_timestep, lags_external_timestep):
            # for predictor, lags are list of tensors for each threshold
            lags_predictor_concat = tf.concat(
                lags_predictor_param,
                axis=0
            )

            # for external variables, lags are two layers of lists. first is column, then threshold
            lags_external_all = []
            for lags_external_column in lags_external_param:
                lags_external_all.extend(lags_external_column)

            lags_concat = tf.concat(
                [lags_predictor_concat, *lags_external_all],
                axis=0
            )
            result = tf.cond(
                tf.greater(tf.size(lags_concat), 0),
                lambda: tf.maximum(result, tf.reduce_max(
                    lags_concat
                ) - idx_timestep),
                lambda: result
            )

    return result


def make_y_tuple(
        data_predictor,
        max_lag_wrt_first_step_prediction,
        n_timesteps_predict
):
    data_steps_future = []
    for idx_timestep_predict in tf.range(n_timesteps_predict):
        data_steps_future.append(
            tf.roll(
                data_predictor,
                -(idx_timestep_predict + 1),
                axis=0
            )
        )
    y_all_data = tf.stack(data_steps_future, axis=1)

    # first max_lag-1 datapoints are droppeed because there is no data further in the past for lagging
    #   so that predictions could be made.
    # last n_timesteps_predict datapoints are dropped because they contain rolled data
    #   from first timesteps where data from the future is missing
    y = y_all_data[(max_lag_wrt_first_step_prediction - 1):-n_timesteps_predict]

    # TODO: have to add newaxis to each datapoint?
    #   if yes, simply append new axis to y_all_data.
    # return [timestep_predict [datapoint]]
    return tf.unstack(y, axis=1)


@tf.function
def sample_model(model, num_samples):
    # Stack samples because they are tuple, where each element corresponds to timestep, having timestep dimension on axis 2.
    # return tf.stack(model.sample(num_samples), axis=2)

    n_samples_batch = 10
    samples_concat = []
    tf.print(f'Sampling {num_samples} samples')
    for num_step_sampling in range(int(np.ceil(num_samples / n_samples_batch))):
        samples_ = model.sample(n_samples_batch)

        samples_concat.append(tf.stack(samples_, axis=2))
        tf.print('sampled', (num_step_sampling + 1) * n_samples_batch, 'of', num_samples, 'samples.')

    tf.print('Finished sampling.')
    return tf.concat(samples_concat, axis=0)


def separate_data(
        data,
        date_start,
        date_end,
        column_predict,
        columns_external,
        columns_oracle,
        n_timesteps_predict,
        fn_transform_external='x',
        fn_transform_oracles='x',
):
    # slice time window by n_timesteps_predict steps further than date_end
    #   so that the last step in resulting data corresponds to date_end
    idx_end = len(data.index[data.index <= date_end]) - 1
    date_end_extended = data.index[idx_end + n_timesteps_predict]
    data_time_window = data[date_start:date_end_extended]

    if columns_oracle != [COLUMN_FORECAST_RADAR]:
        oracles = simulate_oracles(data_time_window[columns_oracle], n_timesteps_predict)
        df_all = data_time_window[
                     [column_predict]
                     + columns_external
                     + [col for col in columns_oracle if col not in columns_external]
                     ].iloc[:-n_timesteps_predict]
    else:
        # cut off last n_steps_predict datapoints because
        # they are cut off when simulating oracles.
        # Also if not using oracles, just to prevent bugs of handling this is forgotten somewhere else for now.
        # simulation data without present.
        # Make col names int timestep to ensure correspondence with simulate_oracles
        oracles = [
            (
                COLUMN_FORECAST_RADAR,
                data_time_window[
                    [str(idx_timestep) for idx_timestep in range(1, n_timesteps_predict + 1)]
                ].rename(
                    columns=lambda col: str(int(col) - 1)
                ).iloc[:-n_timesteps_predict]
            )
        ]

        df_all = data_time_window[
                     [column_predict]
                     + columns_external
                     + [str(idx_timestep) for idx_timestep in range(n_timesteps_predict + 1) if
                        str(idx_timestep) not in columns_external]
                     ].iloc[:-n_timesteps_predict]

    return (
               # cut off last n_steps_predict datapoints because
               # they are cut off when simulating oracles.
               data_time_window[column_predict].iloc[:-n_timesteps_predict],
               # TODO: make sqrt optional via argument that is also saved in model.
               eval(fn_transform_external, {
                   'np': np,
                   'tf': tf,
                   'tfp': tfp,
                   'x': data_time_window[columns_external].iloc[:-n_timesteps_predict]
               }),
               [
                   (
                       key,
                       eval(fn_transform_oracles, {
                           'np': np,
                           'tf': tf,
                           'tfp': tfp,
                           'x': values
                       })
                   )
                   for key, values in oracles
               ]
           ), df_all


def simulate_oracles(data: pd.DataFrame, n_timesteps_predict):
    result = []
    for name_column in data.columns:
        # make dataframe where columns are prediction timesteps
        data_result_column = []
        data_column = data[name_column].to_numpy()
        for idx_timestep in range(n_timesteps_predict):
            data_result_column.append(
                np.roll(data_column, -(idx_timestep + 1))
            )

        data_np = np.asarray(data_result_column).transpose()
        df_result_column = pd.DataFrame(
            data_np,
            columns=[str(idx_timestep_) for idx_timestep_ in range(n_timesteps_predict)],
            index=data.index
        ).iloc[:-n_timesteps_predict]  # remove last data without future information
        result.append((name_column, df_result_column))

    return result


def make_data_tf(
        data_predictor: pd.Series,
        data_external: pd.DataFrame,
        # each item is oracle for a column, where, in the df, columns are timesteps prediction
        #   and rows are datapoints
        data_oracles: [(str, pd.DataFrame)],
        dtype
) -> (tf.Tensor, tf.Tensor, tf.Tensor):
    predictor_tf = tf.constant(data_predictor.to_numpy(), dtype=dtype)
    external_tf = tf.constant(data_external.to_numpy(), dtype=dtype)

    oracles_columns = [
        oracle_column.to_numpy()
        for column, oracle_column in data_oracles
    ]

    if oracles_columns:
        oracles_tf = tf.cast(tf.stack(oracles_columns, axis=1), dtype)
    else:
        oracles_tf = tf.zeros((external_tf.shape[0], 0, 0), dtype=dtype)

    return predictor_tf, external_tf, oracles_tf


def normalize_data(
        data_predictor: pd.Series,
        data_external: pd.DataFrame,
        # each item is oracle for a column, where, in the df, columns are timesteps prediction
        #   and rows are datapoints
        data_oracles: [(str, pd.DataFrame)],
        params_normalization=None
):
    # calc normalization params if not provided
    if params_normalization is not None:
        (
            (mean_predictor, std_predictor),
            (means_external, stds_external),
            (means_oracles, stds_oracles),
        ) = params_normalization
    else:
        mean_predictor = data_predictor.mean()
        means_external = data_external.mean()
        means_oracles = {column: np.mean(oracle_column) for column, oracle_column in data_oracles}
        std_predictor = data_predictor.std()
        stds_external = data_external.std()
        stds_oracles = {column: np.std(oracle_column) for column, oracle_column in data_oracles}

    predictor_normalized = (data_predictor - mean_predictor) / std_predictor
    external_normalized = (data_external - means_external) / stds_external
    oracles_normalized = []
    for column, oracle_column in data_oracles:
        oracles_normalized.append(
            (column, (oracle_column - means_oracles[str(column)]) / stds_oracles[str(column)])
        )

    params_normalization_result = (
        (mean_predictor, std_predictor),
        (means_external, stds_external),
        (means_oracles, stds_oracles),
    )

    return (predictor_normalized, external_normalized, oracles_normalized), params_normalization_result


def unnormalize_prediction(prediction: tf.Tensor, params_normalization):
    mean, std = params_normalization

    return prediction * std + mean


def make_dataset(
        data,
        date_start,
        date_end,
        column_predict,
        columns_external,
        columns_oracle,
        n_timesteps_predict,
        dtype,
        params_normalization=None,
        fn_transform_external='x',
        fn_transform_oracles='x',
):
    data_separate, data_df = separate_data(
        data,
        date_start,
        date_end,
        column_predict,
        columns_external,
        columns_oracle,
        n_timesteps_predict,
        fn_transform_external,
        fn_transform_oracles
    )

    data_normalized, params_normalization_new = normalize_data(
        *data_separate,
        params_normalization=params_normalization
    )

    data_tf = make_data_tf(*data_normalized, dtype)

    return data_tf, data_df, params_normalization_new


def slice_data_model(data, max_lag_wrt_first_step_prediction, n_timesteps_predict, remove_samples_y=False):
    # slice away last steps that will be used for y-generation and first steps that are
    #   used as history,
    # so that datapoints in data correspond to last datapoints in history window
    #   for samples of model generated for data
    # if remove_samples_y, also the last n_timesteps_predict datapoints used for generation of
    #   y-data for training are removed.
    if remove_samples_y:
        return data[max_lag_wrt_first_step_prediction - 1:-n_timesteps_predict]
    else:
        return data[max_lag_wrt_first_step_prediction - 1:]


def softclip(x, low, high, hinge_softness=None):
    high = tf.cast(high, x.dtype)
    low = tf.cast(low, x.dtype)
    # implement forward of according tfp bijector because it cannot deal with tf.float16.
    return -tf.math.softplus(high - low - tf.math.softplus(x - low)) * \
           (high - low) / (tf.math.softplus(high - low)) + high


def get_pbas(Bindex, period=365.24, dK=365.24 / 6, order=4):
    """Estimates periodic B-splines to model the annual periodicity

    Parameters
    ----------
    Bindex : array_like of int
        The array of day numbers for which to estimate the B-splines.
    period : float
        The period of B-splines. By default set to 365.24.
    dK : float
        The equidistance distance used to calculate the knots.
    order : int
        The order of the B-splines. 3 indicates quadratic splines, 4 cubic etc.

    Returns
    -------
    ndarray
        an ndarray of estimated B-splines.
    """
    # ord=4 --> cubic splines
    # dK = equidistance distance
    # support will be 1:n
    n = len(Bindex)
    stp = dK
    x = np.arange(1, period)  # must be sorted!
    lb = x[0]
    ub = x[-1]
    knots = np.arange(lb, ub + stp, step=stp)
    degree = order - 1
    Aknots = np.concatenate(
        (knots[0] - knots[-1] + knots[-1 - degree:-1], knots,
         knots[-1] + knots[1:degree + 1] - knots[0]))

    bspl = Bspline(Aknots, degree)
    basisInterior = bspl.collmat(x)
    basisInteriorLeft = basisInterior[:, :degree]
    basisInteriorRight = basisInterior[:, -degree:]
    basis = np.column_stack(
        (basisInterior[:, degree:-degree],
         basisInteriorLeft + basisInteriorRight))
    ret = basis[np.array(Bindex % basis.shape[0], dtype="int"), :]
    return ret

def create_missing_dir(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

def get_path_make_dir(*args):
    path = os.path.join(*args)
    create_missing_dir(path)

    return path