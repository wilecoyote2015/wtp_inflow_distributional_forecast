from src.common_functions.misc import get_path_make_dir, dump_json_gzip, params_to_json

import json
import numpy as np
import tensorflow as tf

from src.common_functions.get_data import get_data
from src.constants.params_preprocessing import *
from src.inflow_forecast.model.model_base import ModelBase

from src.config_paths import *
from src.config_variables import *

from src.constants.misc  import *
import pandas as pd



OMIT_INVALID = True
N_SAMPLES = 60

only_datetimes=False


def sample_score_model(
        path_model,
        path_save_scores_timesteps_train,
        path_save_scores_timesteps_test,
        path_save_scores_datapoints_train,
        path_save_scores_datapoints_test,
        path_save_scores_aggregated_train,
        path_save_scores_aggregated_test,
        path_save_samples_train,
        path_save_samples_test,
        path_save_indices_subsets_train,
        path_save_indices_subsets_test,
        path_save_indices_valid_train,
        path_save_indices_valid_test,
        path_save_criteria,
        path_save_truth_train,
        path_save_truth_test,
        path_save_datetimes_x_train,
        path_save_datetimes_x_test,
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        column_predict,
        variables_external,
        cols_nm,
        use_rain_forecasts,
        use_future_rain,
        only_datetimes,
        steps_predict=10,
        n_samples=80,
        omit_invalid=True
        ):

    # TODO: load from file with auto model detection
    model = ModelBase.model_from_file(
                    path_model
                )

    variables_use = {column_predict, *variables_external}

    data_train, data_test, columns_forecasts_rain = get_data(
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        use_rain_forecasts,
        steps_predict,
        variables_use,
        cols_nm,
        use_future_rain,
        PARAMS_PREPROCESSING_WEIDEN if omit_invalid else PARAMS_PREPROCESSING_WEIDEN_FILL_CONSTANTS,
        path_csv_data_wtp_network_rain,
        path_csv_forecast_rain_radar,
        steps_cumulate_rain=6,
        keep_raw_rain=True,
    )
    data_train_w_nans, data_test_w_nans, _ = get_data(
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        use_rain_forecasts,
        steps_predict,
        variables_use,
        cols_nm,
        use_future_rain,
        PARAMS_PREPROCESSING_WEIDEN_NANS,
        path_csv_data_wtp_network_rain,
        path_csv_forecast_rain_radar,
        # cols_fill_negative_nan=variables_use,
        # cumulation is done in Model now. but cumulated columns are
        # needed here for threshold generation.
        steps_cumulate_rain=6,
        keep_raw_rain=True,
    )


    if only_datetimes:
        _, _, datetimes_x_test = model.make_x_y(data_test, omit_unknown_future_truth=True)
        datetimes_x_test = model.slice_tensor_sample_predictions(datetimes_x_test)
        _, _, datetimes_x_train = model.make_x_y(data_train, omit_unknown_future_truth=True)
        datetimes_x_train = model.slice_tensor_sample_predictions(datetimes_x_train)

    else:
        bool_valid_train = pd.DataFrame(
            np.logical_not(np.isnan(data_train_w_nans)),
            index = data_train.index,
            columns=data_train.columns
        )

        bool_valid_test = pd.DataFrame(
            np.logical_not(np.isnan(data_test_w_nans)),
            index = data_test.index,
            columns=data_test.columns
        )


        # sampling and scoring
        (
            scores_train,
            scores_test,
            criteria,
            predictions_train,
            predictions_test,
            truth_train,
            truth_test,
            indices_valid_train,
            indices_valid_test,
            indices_subsets_train,
            indices_subsets_test,
            datetimes_x_train,
            datetimes_x_test,
            x_train,
            x_test
        ) = model.evaluate_model(
            data_train,
            data_test,
            cols_nm[0],
            bool_datapoints_valid_train=bool_valid_train,
            bool_datapoints_valid_test=bool_valid_test,
            include_intraday=True,
            include_timesteps=True,
            n_samples_predict=n_samples,
            include_criteria=True
        )

        scores_train_scalar = {
            name_subset: scores_subset[SCORES_SCALAR]
            for name_subset, scores_subset in scores_train.items()
        }
        scores_train_datapoints = {
            name_subset: scores_subset[SCORES_INTRADAY]
            for name_subset, scores_subset in scores_train.items()
        }
        scores_train_timesteps = {
            name_subset: scores_subset[SCORES_TIMESTEPS]
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
        scores_test_timesteps = {
            name_subset: scores_subset[SCORES_TIMESTEPS]
            for name_subset, scores_subset in scores_test.items()
        }

        with open(path_save_scores_aggregated_train, 'w') as f:
            json.dump(params_to_json(scores_train_scalar), f, indent=4)
        with open(path_save_scores_timesteps_train, 'w') as f:
            json.dump(params_to_json(scores_train_timesteps), f, indent=4)
        dump_json_gzip(path_save_scores_datapoints_train, params_to_json(scores_train_datapoints))

        with open(path_save_scores_aggregated_test, 'w') as f:
            json.dump(params_to_json(scores_test_scalar), f, indent=4)
        with open(path_save_scores_timesteps_test, 'w') as f:
            json.dump(params_to_json(scores_test_timesteps), f, indent=4)
        dump_json_gzip(path_save_scores_datapoints_test, params_to_json(scores_test_datapoints))

        with open(path_save_criteria, 'w') as f:
            json.dump(params_to_json(criteria), f, indent=4)

        # save indices
        dump_json_gzip(path_save_indices_subsets_train, params_to_json(indices_subsets_train))
        dump_json_gzip(path_save_indices_subsets_test, params_to_json(indices_subsets_test))

        dump_json_gzip(path_save_samples_train, np.asarray(predictions_train).tolist())
        dump_json_gzip(path_save_samples_test, np.asarray(predictions_test).tolist())
        dump_json_gzip(path_save_indices_valid_train, np.asarray(indices_valid_train).tolist())
        dump_json_gzip(path_save_indices_valid_test, np.asarray(indices_valid_test).tolist())
        dump_json_gzip(path_save_truth_train, np.asarray(truth_train).tolist())
        dump_json_gzip(path_save_truth_test, np.asarray(truth_test).tolist())

    # save datetimes
    with open(path_save_datetimes_x_train, 'w') as f:
        json.dump(params_to_json(datetimes_x_train), f, indent=4)
    with open(path_save_datetimes_x_test, 'w') as f:
        json.dump(params_to_json(datetimes_x_test), f, indent=4)

    del model
    tf.keras.backend.clear_session()


for name_class_model in os.listdir(dir_models):
    dir_models_ = os.path.join(dir_models, name_class_model)
    for filename_model in os.listdir(dir_models_):
        filename_train = filename_model[:-5] + f'{SUFFIX_TRAIN}.json'
        filename_test = filename_model[:-5] + f'{SUFFIX_TEST}.json'

        dir_samples_model = get_path_make_dir(dir_samples, name_class_model)
        dir_truth_model = get_path_make_dir(dir_truth, name_class_model)
        dir_indices_valid_model = get_path_make_dir(dir_indices_valid, name_class_model)
        dir_indices_subsets_model = get_path_make_dir(dir_indices_subsets, name_class_model)
        dir_scores_aggregated_model = get_path_make_dir(dir_scores, name_class_model, NAME_DIR_SCORES_AGGREGATED)
        dir_scores_timesteps_model = get_path_make_dir(dir_scores, name_class_model, NAME_DIR_SCORES_TIMESTEPS)
        dir_scores_datapoints_model = get_path_make_dir(dir_scores, name_class_model, NAME_DIR_SCORES_DATAPOINTS)
        dir_criteria_model = get_path_make_dir(dir_samples, name_class_model)
        dir_datetimes_x_model = get_path_make_dir(dir_samples, name_class_model)

        path_save_samples_train = os.path.join(dir_samples_model, filename_train)
        path_save_samples_test = os.path.join(dir_samples_model, filename_test)
        path_save_truth_train = os.path.join(dir_truth_model, filename_train)
        path_save_truth_test = os.path.join(dir_truth_model, filename_test)
        path_save_indices_valid_train = os.path.join(dir_indices_valid_model, filename_train)
        path_save_indices_valid_test = os.path.join(dir_indices_valid_model, filename_test)
        path_save_indices_subsets_train = os.path.join(dir_indices_subsets_model, filename_train)
        path_save_indices_subsets_test = os.path.join(dir_indices_subsets_model, filename_test)
        path_save_scores_aggregated_train = os.path.join(dir_scores_aggregated_model, filename_train)
        path_save_scores_aggregated_test = os.path.join(dir_scores_aggregated_model, filename_test)
        path_save_scores_timesteps_train = os.path.join(dir_scores_timesteps_model, filename_train)
        path_save_scores_timesteps_test = os.path.join(dir_scores_timesteps_model, filename_test)
        path_save_scores_datapoints_train = os.path.join(dir_scores_datapoints_model, filename_train)
        path_save_scores_datapoints_test = os.path.join(dir_scores_datapoints_model, filename_test)
        path_save_criteria = os.path.join(dir_criteria_model, filename_model)
        path_save_datetimes_x_train = os.path.join(dir_datetimes_x_model, filename_train)
        path_save_datetimes_x_test = os.path.join(dir_datetimes_x_model, filename_test)

        use_future_rain = 'furain_True' in filename_model
        use_rain_forecasts = 'rfore_True' in filename_model

        print(f'Sampling and scoring model {filename_model}')

        if os.path.isfile(path_save_datetimes_x_test ):
            print(f'skipping {filename_model} because it exists')
            continue

        sample_score_model(
            os.path.join(dir_models, filename_model),
            path_save_scores_timesteps_train,
            path_save_scores_timesteps_test,
            path_save_scores_datapoints_train,
            path_save_scores_datapoints_test,
            path_save_scores_aggregated_train,
            path_save_scores_aggregated_test,
            path_save_samples_train,
            path_save_samples_test,
            path_save_indices_subsets_train,
            path_save_indices_subsets_test,
            path_save_indices_valid_train,
            path_save_indices_valid_test,
            path_save_criteria,
            path_save_truth_train,
            path_save_truth_test,
            path_save_datetimes_x_train,
            path_save_datetimes_x_test,
            date_start_train,
            date_end_train,
            date_start_test,
            date_end_test,
            name_column_inflow,
            variables_external,
            [name_column_rain_history],
            use_rain_forecasts,
            use_future_rain,
            only_datetimes=only_datetimes,
            n_samples=N_SAMPLES
        )


