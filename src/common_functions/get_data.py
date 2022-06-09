import pandas as pd
from src.common_functions.preprocessing import preprocess_data_realtime
import numpy as np
import logging
from src.constants.misc import *
from natsort import natsorted
from src.constants.params_preprocessing import PARAMS_PREPROCESSING_WEIDEN, \
    PARAMS_PREPROCESSING_WEIDEN_NANS, PARAMS_PREPROCESSING_WEIDEN_FILL_CONSTANTS

KEY_RAIN_CUMULATED = 'RAIN_CUMULATED'
COL_RAIN_MAX = 'RAIN_MAX'
COL_RAIN_MEDIAN = 'RAIN_MEDIAN'
COL_RAIN_RAW = 'RAIN_RAW'


def rolling_back_np(data, window_size, fn):
    """Cheap backwards rolling window with numpy, sinde pandas has severe numerical issues (sum of zeros yields > 0..."""
    data_rolled_list = []
    for shift in range(window_size):
        # positive shift rolls forward, meaning presenting past data, which is desired.
        data_rolled_shift = np.roll(data, shift)

        # fill invalid boundaries with nans
        data_rolled_shift[:shift] = np.nan
        data_rolled_list.append(data_rolled_shift)

    return fn(np.stack(data_rolled_list, axis=0))


def make_interactions_threshold_lagged(df_, pairs_interaction, lags, thresholds):
    df = df_.copy()
    logging.warning(f'interaction thresholded lagging df. first datapoints according to min lag are omitted!')
    columns_new = []
    min_lag = 0
    for (column_base, column_lag), lags_pair, threshold_pair in zip(pairs_interaction, lags, thresholds):
        for lag in lags_pair:
            min_lag = min(min_lag, lag)
            name_new_column = f'{INTERACTION}_{column_base}_{column_lag}_{LAG}_{lag}_{THRESHOLD}_{threshold_pair}'
            if name_new_column not in df.columns:
                column_base_below_threshold = df[column_base] <= threshold_pair
                column_lagged = np.roll(df[column_lag], -lag)
                df[name_new_column] = column_lagged * column_base_below_threshold.astype(float)
                columns_new.append(name_new_column)
    return df.iloc[-min_lag:], columns_new


def make_interactions_multiply_lagged(df_, pairs_interaction, lags, power_a=1, power_b=1):
    df = df_.copy()
    logging.warning(f'interaction lagging df. first datapoints according to min lag are omitted!')
    columns_new = []
    min_lag = 0
    for (column_base, column_lag), lags_pair in zip(pairs_interaction, lags):
        for lag in lags_pair:
            min_lag = min(min_lag, lag)
            name_new_column = f'{INTERACTION}_{column_base}_{column_lag}_lag_{lag}'
            if power_a != 1 or power_b != 1:
                name_new_column += f'_{POW}_{power_a}_{power_b}'
            if name_new_column not in df.columns:
                df[name_new_column] = np.power(df[column_base], power_a) * np.power(np.roll(df[column_lag], -lag),
                                                                                    power_b)
                columns_new.append(name_new_column)
    return df.iloc[-min_lag:], columns_new


def get_data(
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        use_rain_forecasts,
        n_steps_predict,
        variables_use: set,
        columns_measurement_rain,
        use_future_rain_variables,
        kwargs_preprocessing,
        path_csv_network_rain_measurements,
        path_csv_forecast_rain,
        steps_cumulate_rain=None,
        keep_raw_rain=False,
        df_forecasts_rain=None,
        df_data_network=None,
        make_temp_cumulations=True,
        **kwargs
):
    variables_use = set(variables_use)

    data_raw = pd.read_csv(
        path_csv_network_rain_measurements,
        delimiter=';',
        index_col='Timestamp',
        parse_dates=True
    ) if df_data_network is None else df_data_network

    columns_forecast = {}

    if use_future_rain_variables:
        if use_rain_forecasts:
            df_forecast_rain: pd.DataFrame = pd.read_csv(
                path_csv_forecast_rain,
                parse_dates=True,
                index_col=0
            ) if df_forecasts_rain is None else df_forecasts_rain

            # remark: timestep is 1-indiced! timestep 0 is present rain, not first future rain!
            #   hence, present is removed.
            for col_ in df_forecast_rain.columns:
                if col_.endswith('0'):
                    df_forecast_rain.drop(col_, inplace=True, axis=1)

            # name pattern convention in forecast data is key_shape_idx_timestep+1. 0 would hence correspond to present
            #   and 1 ist first prediction step.
            #   convert to zero-index w.r.t. first timestep, so that number corresponds to idx_timestep.
            df_forecast_rain.rename(columns=lambda col_: f'{FORECAST}_radar_{col_[:-1]}_{int(col_[-1]) - 1}',
                                    inplace=True)

            # Assume that forecasts of individual stations have same name and end with timestep.
            columns_forecasts_rain = natsorted(list(df_forecast_rain.columns))
            data_raw = data_raw.join(df_forecast_rain, how='left')

        else:
            columns_forecasts_rain = []
            for column_measurement_rain in columns_measurement_rain:
                for idx_step_predict in range(n_steps_predict):
                    name_column_forecast = f'{ORACLE}_{idx_step_predict}_{column_measurement_rain}'
                    data_raw[name_column_forecast] = np.roll(data_raw[column_measurement_rain], -(idx_step_predict + 1))
                    columns_forecasts_rain.append(name_column_forecast)

            # FIXME: in forecasts, Column 0 is the current rain and should be incorporated as further rain column.
        if steps_cumulate_rain is not None and make_temp_cumulations:
            logging.info(f'cumulating rain with {steps_cumulate_rain} steps.')
            # remark: don't use pd rolling due to numerical issues.
            for column_measurement_rain in columns_measurement_rain:
                name_column_cumulated = f'{CUMULATE}_{steps_cumulate_rain}_{column_measurement_rain}'

                # perform cumulation here for the later threshold calculation.
                #   after that, the created columns are removed,
                #   because cumulation will be performed in model.
                data_raw[name_column_cumulated] = rolling_back_np(
                    data_raw[column_measurement_rain],
                    steps_cumulate_rain,
                    lambda x: np.nansum(x, axis=0)
                )

                variables_use.add(name_column_cumulated)

            columns_forecast_cumulated = []
            for idx_forecast, column_forecast_rain in enumerate(columns_forecasts_rain):
                name_cumulated = f'{CUMULATE}_{steps_cumulate_rain}_{column_forecast_rain}'
                data_raw[name_cumulated] = rolling_back_np(
                    data_raw[column_forecast_rain],
                    steps_cumulate_rain,
                    lambda x: np.nansum(x, axis=0)
                )
                columns_forecast_cumulated.append(name_cumulated)

            columns_forecast[KEY_RAIN_CUMULATED] = columns_forecast_cumulated

        use_raw_rain = keep_raw_rain and steps_cumulate_rain is None
        if use_raw_rain:
            columns_forecast[COL_RAIN_RAW] = columns_forecasts_rain

        for columns_ in columns_forecast.values():
            variables_use.update(columns_)

        if not use_raw_rain:
            # Remark: cannot drop rain measurement because finding subsets for scoring needs the rain.
            # if column_measurement_rain in variables_use:
            #     variables_use.remove(column_measurement_rain)
            # data_preprocessed.drop(column_measurement_rain, inplace=True, axis=1)
            for idx_oracle, name_oracle in enumerate(columns_forecasts_rain):
                if name_oracle in variables_use:
                    variables_use.remove(name_oracle)

                data_raw.drop(name_oracle, inplace=True, axis=1)

    data_used_columns = data_raw[list(variables_use)]

    data_all_preprocessed = preprocess_data_realtime(data_used_columns,
                                                     **kwargs_preprocessing) if kwargs_preprocessing is not None else data_used_columns

    data_train = data_all_preprocessed[date_start_train:date_end_train]
    data_test = data_all_preprocessed[date_start_test:date_end_test]

    return data_train, data_test, columns_forecast


def get_data_lags_thresholds_w_interaction(
        cls_model,
        column_predict,
        variables_external,
        cols_nm,
        min_lags,
        n_thresholds_predict,
        n_thresholds_external,
        n_thresholds_rain,
        n_thresholds_interactions,
        fn_transform_thresholds,
        make_interaction_rain_multiply,
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        use_rain_forecasts,
        use_future_rain_variables,
        n_steps_predict,
        omit_invalid_datapoints,
        path_csv_network_rain_measurements,
        path_csv_forecast_rain,
        lags_target_additional=None,
        steps_cumulate_rain=None,
        keep_raw_rain=False,
        min_lags_interactions=None,
        interactions_only_known_data=False,
        params_preprocessing_fill_constants=PARAMS_PREPROCESSING_WEIDEN_FILL_CONSTANTS,
        params_preprocessing_nans=PARAMS_PREPROCESSING_WEIDEN_NANS,
        params_preprocessing_default=PARAMS_PREPROCESSING_WEIDEN,
        furain_all_lags=False,
        interactions_rain_w_rain=False,
        interact_also_raw_rain=True,
        interact_raw_with_cumulated=True,
        interact_all=False,
        **kwargs_get_data
):
    if min_lags_interactions is None:
        min_lags_interactions = min_lags

    variables_use = set([
        column_predict,
        *variables_external,
    ])

    data_train, data_test, columns_forecasts_rain = get_data(
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        use_rain_forecasts,
        n_steps_predict,
        variables_use,
        cols_nm,
        use_future_rain_variables,
        steps_cumulate_rain=steps_cumulate_rain,
        keep_raw_rain=True,
        kwargs_preprocessing=params_preprocessing_default if omit_invalid_datapoints else params_preprocessing_fill_constants,
        invalid_to_nan=False,
        path_csv_network_rain_measurements=path_csv_network_rain_measurements,
        path_csv_forecast_rain=path_csv_forecast_rain,
        **kwargs_get_data
    )

    if omit_invalid_datapoints:
        data_train_w_nans, data_test_w_nans, _ = get_data(
            date_start_train,
            date_end_train,
            date_start_test,
            date_end_test,
            use_rain_forecasts,
            n_steps_predict,
            variables_use,
            cols_nm,
            use_future_rain_variables,
            steps_cumulate_rain=steps_cumulate_rain,
            keep_raw_rain=True,
            kwargs_preprocessing=params_preprocessing_nans,
            invalid_to_nan=False,
            path_csv_network_rain_measurements=path_csv_network_rain_measurements,
            path_csv_forecast_rain=path_csv_forecast_rain,
            **kwargs_get_data
        )
    else:
        data_train_w_nans, data_test_w_nans = data_train, data_test

    columns_only_use_recent_lag = []
    if not furain_all_lags:
        for columns in columns_forecasts_rain.values():
            columns_only_use_recent_lag.extend(columns)

        data_train = data_train.fillna(method='pad')
        data_test = data_test.fillna(method='pad')

    pairs_interaction = []

    if interact_all:
        raise NotImplementedError

    # Interactions are made in model/
    #   ensure that needed columns are in data.
    if make_interaction_rain_multiply and not interact_all and use_future_rain_variables:
        interact_rain_future = True
        interact_rain_past = True
        interact_sum = True

        if keep_raw_rain and interact_also_raw_rain:
            if interact_rain_past:
                for col_nm in cols_nm:
                    pairs_interaction.append((column_predict, col_nm))
            if interact_rain_future:
                pairs_interaction.extend([(column_predict, col_) for col_ in columns_forecasts_rain[COL_RAIN_RAW]])

            if interactions_rain_w_rain and interact_rain_future:
                for idx_1 in range(len(columns_forecasts_rain[COL_RAIN_RAW])):

                    if steps_cumulate_rain is not None and interact_raw_with_cumulated:
                        for idx_cumulated in range(len(columns_forecasts_rain[KEY_RAIN_CUMULATED])):
                            pairs_interaction.append(
                                (
                                    columns_forecasts_rain[COL_RAIN_RAW][idx_1],
                                    columns_forecasts_rain[KEY_RAIN_CUMULATED][idx_cumulated],
                                )
                            )

                    for idx_2 in range(idx_1 + 1, len(columns_forecasts_rain[COL_RAIN_RAW])):
                        pairs_interaction.append(
                            (
                                columns_forecasts_rain[COL_RAIN_RAW][idx_1],
                                columns_forecasts_rain[COL_RAIN_RAW][idx_2],
                            )
                        )
                    for col_nm in cols_nm:
                        pairs_interaction.append((
                            columns_forecasts_rain[COL_RAIN_RAW][idx_1],
                            col_nm
                        ))

        if steps_cumulate_rain is not None and interact_sum:
            if interact_rain_past:
                for col_nm in cols_nm:
                    pairs_interaction.append((column_predict, f'{CUMULATE}_{steps_cumulate_rain}_{col_nm}'))
            if interact_rain_future and use_future_rain_variables:
                pairs_interaction.extend(
                    [(column_predict, col_) for col_ in columns_forecasts_rain[KEY_RAIN_CUMULATED]])
            if interactions_rain_w_rain and interact_rain_future:
                for idx_1 in range(len(columns_forecasts_rain[KEY_RAIN_CUMULATED])):
                    for idx_2 in range(idx_1 + 1, len(columns_forecasts_rain[KEY_RAIN_CUMULATED])):
                        pairs_interaction.append(
                            (
                                columns_forecasts_rain[KEY_RAIN_CUMULATED][idx_1],
                                columns_forecasts_rain[KEY_RAIN_CUMULATED][idx_2],
                            )
                        )
                    for col_nm in cols_nm:
                        pairs_interaction.append((
                            columns_forecasts_rain[KEY_RAIN_CUMULATED][idx_1],
                            f'{CUMULATE}_{steps_cumulate_rain}_{col_nm}'
                        ))

    n_thresholds_variables = {}
    for name_variable in data_train.columns:
        if name_variable == column_predict:
            n_thresholds_variables[name_variable] = n_thresholds_predict
        # TODO: generalize to omit hardcoded ORACLE and mm/min
        elif cls_model.check_variable_is_rain(name_variable):
            n_thresholds_variables[name_variable] = n_thresholds_rain
        elif name_variable in variables_external:
            n_thresholds_variables[name_variable] = n_thresholds_external
        else:
            raise ValueError(f'Cannot make threshold for variable {name_variable}')

    for pair in pairs_interaction:
        n_thresholds_variables[pair] = n_thresholds_interactions

    thresholds = cls_model.make_thresholds_variables(
        data_train_w_nans,
        n_thresholds_variables,
        fn_transform_thresholds,
        n_steps_predict,
    )

    # need names of columsn and pair for making min and max lags
    columns_for_lags = list(data_train.columns) + pairs_interaction

    min_lags_variables = {name_variable: min_lags_interactions if isinstance(name_variable, (tuple, list)) else min_lags
                          for name_variable in columns_for_lags}
    max_lags_variables = {name_variable: -1 for name_variable in columns_for_lags}
    lags = cls_model.make_dense_lags(
        min_lags_variables,
        max_lags_variables,
        n_thresholds_variables,
        n_steps_predict,
        column_predict,
        columns_only_use_recent_lag,
        include_predictions_target=True,  # Must be true because autoregressive is very important
        min_lags_relative_to_first_step_predict=True,  # True is good with -6 lags
        lags_target_additional=lags_target_additional,
        furain_all_lags=furain_all_lags
        # better false because no information for late timesteps in data
    )

    lags_interaction, thresholds_interaction = cls_model.make_lags_thresholds_interactions(
        data_train_w_nans,
        columns_only_use_recent_lag,
        column_predict,
        not interactions_only_known_data,
        min_lags_variables,
        max_lags_variables,
        n_thresholds_variables,
        fn_transform_thresholds,
        n_steps_predict,
        True,
    )

    lags.update(lags_interaction)
    thresholds.update(thresholds_interaction)

    bool_valid_train = pd.DataFrame(
        np.logical_not(np.isnan(data_train_w_nans)),
        index=data_train.index,
        columns=data_train.columns
    )

    bool_valid_test = pd.DataFrame(
        np.logical_not(np.isnan(data_test_w_nans)),
        index=data_test.index,
        columns=data_test.columns
    )

    # drop the auxiliary columns used for calculating thresholds (currently, this is only cumulated columns)
    columns_needed = natsorted([col_ for col_ in data_train.columns if not col_.startswith(CUMULATE)])

    return data_train[columns_needed], data_test[columns_needed], bool_valid_train[columns_needed], bool_valid_test[
        columns_needed], thresholds, lags, columns_only_use_recent_lag
