import json
from src.common_functions.get_data import get_data, get_data_lags_thresholds_w_interaction
from src.common_functions.misc import params_to_json, fn_cap_rain, make_dir_if_missing, dump_json_gzip

import numpy as np
import os
import logging
from src.constants.misc import *

# TODO / FIXME: 

def fit_score_model(
        dir_save_base,
        class_model,
        date_train_start,
        date_train_end,
        date_test_start,
        date_test_end,
        min_lag,
        min_lag_interaction,
        n_thresholds_predictor,
        n_thresholds_external,
        n_thresholds_rain,
        n_thresholds_interactions,
        n_seasons_annual,
        steps_cumulate_rain,
        interaction_only_known_data,
        lambda_lasso,
        use_future_rain_variables,
        use_rain_forecast,
        variables_external,
        class_optimizer,
        kwargs_optimizer,
        n_iters_fit,
        n_samples_predict,
        suffix_name_save=None,
        n_steps_predict=10,
        weight_mse=10,
        save_data=False,
        n_iters_per_param=3,
        use_fitted_model_if_exists=True
):

    model_description = dict(
        date_train_start=date_train_start.isoformat(),
        date_train_end=date_train_end.isoformat(),
        date_test_start=date_test_start.isoformat(),
        date_test_end=date_test_end.isoformat(),
        min_lag=min_lag,
        min_lag_interaction=min_lag_interaction,
        n_thresholds_predictor=n_thresholds_predictor,
        n_thresholds_external=n_thresholds_external,
        n_thresholds_rain=n_thresholds_rain,
        n_thresholds_interactions=n_thresholds_interactions,
        n_seasons_annual=n_seasons_annual,
        steps_cumulate_rain=steps_cumulate_rain,
        interaction_only_known_data=interaction_only_known_data,
        lambda_lasso=lambda_lasso,
        use_future_rain_variables=use_future_rain_variables,
        use_rain_forecast=use_rain_forecast,
        variables_external=variables_external,
        n_steps_predict=n_steps_predict,
        n_iters_fit=n_iters_fit,
        class_model=class_model.__name__,
        weight_mse=weight_mse
    )

    logging.info(f'Fitting Model: {model_description}')

    suffix_name_save = f'_{suffix_name_save}' if suffix_name_save is not None else ''
    filename_save = f'lambda_{lambda_lasso}{suffix_name_save}.json'

    dir_save_all = os.path.join(
        dir_save_base,
        f'min_lag__{min_lag}',
        f'min_lag_interaction__{min_lag_interaction}',
        f'weight_mse__{weight_mse}',
        f'steps_cum_rain__{steps_cumulate_rain}',
        f'n_seasons_annual__{n_seasons_annual}',
        f'use_future_rain__{use_future_rain_variables}',
        f'rain_forecast' if use_rain_forecast else 'rain_oracle',
        f'thresh_predictor__{n_thresholds_predictor}',
        f'thresh_external__{n_thresholds_external}',
        f'thresh_rain__{n_thresholds_rain}',
        f'thresh_interactions__{n_thresholds_interactions}',
    )

    dir_save_model = os.path.join(dir_save_all, 'models')
    dir_save_data = os.path.join(dir_save_all, 'data')
    dir_save_samples = os.path.join(dir_save_all, 'samples_truth')
    dir_save_scores = os.path.join(dir_save_all, 'scores')

    make_dir_if_missing(dir_save_model)
    make_dir_if_missing(dir_save_data)
    make_dir_if_missing(dir_save_scores)
    make_dir_if_missing(dir_save_samples)

    path_save_model = os.path.join(
        dir_save_model,
        filename_save
    )
    path_save_data_train = os.path.join(
        dir_save_data,
        f'train_{filename_save}'
    )
    path_save_data_train_nans = os.path.join(
        dir_save_data,
        f'train_nans_{filename_save}'
    )
    path_save_data_test = os.path.join(
        dir_save_data,
        f'test_{filename_save}'
    )
    path_save_data_test_nans = os.path.join(
        dir_save_data,
        f'test_nans_{filename_save}'
    )

    path_save_scores_train_scalar = os.path.join(
        dir_save_scores,
        f'scores_train_scalar_{filename_save}'
    )
    path_save_scores_test_scalar = os.path.join(
        dir_save_scores,
        f'scores_test_scalar_{filename_save}'
    )
    path_save_scores_train_intraday = os.path.join(
        dir_save_scores,
        f'scores_train_datapoints_{filename_save}'
    )
    path_save_scores_test_intraday = os.path.join(
        dir_save_scores,
        f'scores_test_datapoints_{filename_save}'
    )

    path_save_samples_1_train = os.path.join(
        dir_save_samples,
        f'samples_1_train_{filename_save}'
    )
    path_save_samples_2_train = os.path.join(
        dir_save_samples,
        f'samples_2_train_{filename_save}'
    )
    path_save_samples_1_test = os.path.join(
        dir_save_samples,
        f'samples_1_test_{filename_save}'
    )
    path_save_samples_2_test = os.path.join(
        dir_save_samples,
        f'samples_2_test_{filename_save}'
    )
    path_save_indices_valid_train = os.path.join(
        dir_save_samples,
        f'indices_valid_train_{filename_save}'
    )
    path_save_indices_valid_test = os.path.join(
        dir_save_samples,
        f'indices_valid_test_{filename_save}'
    )
    path_save_indices_subsets_train = os.path.join(
        dir_save_samples,
        f'indices_subsets_train_{filename_save}'
    )
    path_save_indices_subsets_test = os.path.join(
        dir_save_samples,
        f'indices_subsets_test_{filename_save}'
    )

    path_save_truth_train = os.path.join(
        dir_save_samples,
        f'truth_train_{filename_save}'
    )
    path_save_truth_test = os.path.join(
        dir_save_samples,
        f'truth_test_{filename_save}'
    )
    path_save_datetimes_train = os.path.join(
        dir_save_samples,
        f'datetimes_train_{filename_save}'
    )
    path_save_datetimes_test = os.path.join(
        dir_save_samples,
        f'datetimes_test_{filename_save}'
    )

    column_predict = 'Zulauf Klaeranlage (Venturi) [l/s]'
    COL_NM = 'NMS1302 Weiden [mm/min]'

    seasons_days_weeks = [[0, 1, 2, 3, 4], [5], [6]]

    fns_transform_variables_raw = {
        COL_NM: fn_cap_rain
    }

    variables_fix_constant = [
        'MWE1303 Goethestrasse [mNHN]',
        'SKU1399 Koelnerstrasse [mNHN]',
        'MWE1208 Beller Weg [mNHN]',
        'MWP1347 Am Randkanal Regenwettersumpf [mNHN]',  # usable for predicting decline?
        'Abfluss SKU [l/s]',
        'MWE1305 Bahnstrasse [mNHN]',
        'SKU1302 [mNHN]',
    ]

    (
        data_train,
        data_test,
        data_train_w_nans,
        data_test_w_nans,
        thresholds,
        lags,
        columns_only_use_recent_lag
    ) = get_data_lags_thresholds_w_interaction(
        get_data,
        class_model,
        column_predict,
        variables_external,
        COL_NM,
        min_lag,
        n_thresholds_predictor,
        n_thresholds_external,
        n_thresholds_rain,
        n_thresholds_interactions,
        'x',
        True,
        False,
        variables_fix_constant,
        variables_fix_constant,
        date_train_start,
        date_train_end,
        date_test_start,
        date_test_end,
        use_rain_forecast,
        use_future_rain_variables,
        n_steps_predict,
        True,
        steps_cumulate_rain=steps_cumulate_rain,
        steps_max_rain=None,  # None
        keep_raw_rain=True,  # True
        min_lags_interactions=min_lag_interaction,
        interactions_only_known_data=interaction_only_known_data,
        fns_transform_variables_raw=fns_transform_variables_raw
    )

    if not os.path.isfile(path_save_model) or not use_fitted_model_if_exists:
        logging.info('Fitting new model')
        model = class_model(
            # columns_w_rain_forecast,
            column_predict,
            n_steps_predict,
            -min_lag + 1,
            lags,
            thresholds,
            seasons_days_weeks,
            n_seasons_annual,
            lambda_lasso,
            kwargs_optimizer=kwargs_optimizer,
            make_design_matrix_sparse=False,
            start_tensorboard=False,
            debug=False,
            class_optimizer=class_optimizer,
            n_iters_per_param=n_iters_per_param,
            model_description=model_description
        )
        logging.info('fitting model')
        model.fit(
            data_train,
            data_invalid_nan=data_train_w_nans,
            path_save_model=path_save_model,
            overwrite_params_scaling=False,
            n_iters=n_iters_fit,
            params_pdf_fit=None,
            fit_threshold_seasonal=True,
            fit_threshold_rain=True,
            fit_threshold_loc=True,
            warm_start=False,
            weight_mse=weight_mse,
        )

        logging.info('Save model')
        model.save_model(path_save_model)

    else:
        logging.info('Using existing model')
        model = class_model.model_from_file(filepath=path_save_model)

    if save_data:
        logging.info('Save data')
        data_train.to_json(path_save_data_train)
        data_train_w_nans.to_json(path_save_data_train_nans)
        data_test.to_json(path_save_data_test)
        data_test_w_nans.to_json(path_save_data_test_nans)

    logging.info('evaluating model')
    (
        scores_train,
        scores_test,
        criteria,
        predictions_train_1,
        predictions_train_2,
        predictions_test_1,
        predictions_test_2,
        truth_train,
        truth_test,
        indices_valid_train,
        indices_valid_test,
            indices_subsets_train,
            indices_subsets_test,
        datetimes_x_train,
        datetimes_x_test
    ) = model.evaluate_model(
        data_train,
        data_test,
        COL_NM,
        data_nans_invalid_train=data_train_w_nans,
        data_nans_invalid_test=data_test_w_nans,
        include_intraday=True,
        n_samples_predict=n_samples_predict
    )
    
    scores_train_scalar = {
        name_subset: scores_subset[SCORES_SCALAR]
        for name_subset, scores_subset in scores_train.items()
    }
    scores_train_datapoints = {
        name_subset: scores_subset[SCORES_INTRADAY]
        for name_subset, scores_subset in scores_train.items()
    }
    
    scores_test_scalar = {
        name_subset: scores_subset[SCORES_SCALAR]
        for name_subset, scores_subset in scores_test.items()
    }
    scores_test_datapoints = {
        name_subset: scores_subset[SCORES_INTRADAY]
        for name_subset, scores_subset in scores_test.items()
    }

    with open(path_save_scores_train_scalar, 'w') as f:
        json.dump(params_to_json(scores_train_scalar), f, indent=4)
    dump_json_gzip(path_save_scores_train_intraday, params_to_json(scores_train_datapoints))
    with open(path_save_scores_test_scalar, 'w') as f:
        json.dump(params_to_json(scores_test_scalar), f, indent=4)
    dump_json_gzip(path_save_scores_test_intraday, params_to_json(scores_test_datapoints))

    dump_json_gzip(path_save_indices_subsets_train, params_to_json(indices_subsets_train))
    dump_json_gzip(path_save_indices_subsets_test, params_to_json(indices_subsets_test))

    dump_json_gzip(path_save_samples_1_train, np.asarray(predictions_train_1).tolist())
    dump_json_gzip(path_save_samples_2_train, np.asarray(predictions_train_2).tolist())
    dump_json_gzip(path_save_samples_1_test, np.asarray(predictions_test_1).tolist())
    dump_json_gzip(path_save_samples_2_test, np.asarray(predictions_test_2).tolist())
    dump_json_gzip(path_save_indices_valid_train, np.asarray(indices_valid_train).tolist())
    dump_json_gzip(path_save_indices_valid_test, np.asarray(indices_valid_test).tolist())
    dump_json_gzip(path_save_truth_train, np.asarray(truth_train).tolist())
    dump_json_gzip(path_save_truth_test, np.asarray(truth_test).tolist())


    # TODO: save datetimes

    return filename_save
