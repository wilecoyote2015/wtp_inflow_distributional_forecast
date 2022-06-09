import os
import tensorflow as tf
from src.inflow_forecast.model.gamlss.models_gamlss import ModelGamlssJsu
from src.inflow_forecast.model.point_and_sarx.model_point import ModelPointBase
from src.common_functions.get_data import get_data
from src.constants.params_preprocessing import PARAMS_PREPROCESSING_WEIDEN, \
    PARAMS_PREPROCESSING_WEIDEN_NANS, PARAMS_PREPROCESSING_WEIDEN_FILL_CONSTANTS
import numpy as np
import pandas as pd
from src.config_variables import *
from src.config_paths import *

from src.common_functions.misc import get_path_make_dir


OMIT_INVALID = True

def fit_model(
        cls_model,
        path_model_point,
        path_save_model,
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        column_predict,
        variables_external,
        cols_nm,
        kwargs_optimizer,
        class_optimizer,
        n_iters,
        n_iters_per_param,
        factor_mse,
        use_rain_forecasts,
        use_future_rain,
        warm_start,
        lambda_lasso
        ):
    model = cls_model.model_from_point_forecast_file(
                    path_model_point,
                    kwargs_optimizer=kwargs_optimizer,
                    class_optimizer=class_optimizer,
                    lambdas_lasso=lambda_lasso,
                    path_tensorboard=None,
                    n_iters_per_param=n_iters_per_param
                )

    variables_use = {column_predict, *variables_external}

    data_train, data_test, columns_forecasts_rain = get_data(
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        use_rain_forecasts,
        n_steps_predict,
        variables_use,
        cols_nm,
        use_future_rain,
        PARAMS_PREPROCESSING_WEIDEN if OMIT_INVALID else PARAMS_PREPROCESSING_WEIDEN_FILL_CONSTANTS,
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
        n_steps_predict,
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
        rain_from_historical_radar=False
    )

    bool_valid_train = pd.DataFrame(
        np.logical_not(np.isnan(data_train_w_nans)),
        index = data_train.index,
        columns=data_train.columns
    )

    model.fit(data_train, bool_datapoints_valid=bool_valid_train,
              # path_save_model=path_save_model,
              overwrite_params_scaling=False,
              n_iters=n_iters,
              params_pdf_fit=None,
              fit_threshold_seasonal=True,
              fit_threshold_rain=True,
              fit_threshold_loc=True,
              # TODO: NO!
              warm_start=warm_start,
              weight_penalty=factor_mse,
              )

    model.save_model(path_save_model)

    tf.keras.backend.clear_session()
    del model
    del data_test
    del data_train
    del data_test_w_nans
    del data_train_w_nans
    del _
    del columns_forecasts_rain

cls_model = ModelGamlssJsu

dir_models_point = os.path.join(dir_models, ModelPointBase.__name__)
dir_save_models = os.path.join(dir_models, cls_model.__name__)

if not os.path.exists(dir_save_models):
    os.makedirs(dir_save_models)

names_include_exclusive = []
models_exclude = []

names_models_point = [
    name_
    for name_ in os.listdir(dir_models_point) if 'shrunken' in name_ and name_ not in models_exclude
] if not names_include_exclusive else names_include_exclusive

n_iters = 20000
n_iters_per_param = 3
factor_mse = 10
warm_start = True
lambda_lasso = 0

class_optimizer = tf.optimizers.Adam

learning_rate_common = 0.0002
learning_rate_end_common = learning_rate_common / 2.
decay_steps = 2000 * n_iters_per_param
amsgrad = False
beta_1 = 0.9
beta_2 = 0.5

kwargs_optimizer = [
    # skew
    {
        'learning_rate': tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=learning_rate_common,
                                                                       end_learning_rate=learning_rate_end_common,
                                                                       decay_steps=decay_steps,
                                                                       power=0.5
                                                                       ),
        'amsgrad': amsgrad,
        'beta_1': beta_1,
        'beta_2': beta_2
    },
    # tail
    {
        'learning_rate': tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=learning_rate_common * 10,
                                                                       end_learning_rate=learning_rate_end_common * 10,
                                                                       decay_steps=decay_steps,
                                                                       power=0.5
                                                                       ),
        'amsgrad': amsgrad,
        'beta_1': beta_1,
        'beta_2': beta_2
    },
    # loc
    {
        'learning_rate': tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=learning_rate_common / 2.,  # learning rate reasonable?
            end_learning_rate=learning_rate_end_common / 2.,
            decay_steps=decay_steps,
            power=0.5
            ),
        'amsgrad': amsgrad,
        'beta_1': beta_1,
        'beta_2': beta_2
    },
    # std
    {
        'learning_rate': tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=learning_rate_common,
                                                                       end_learning_rate=learning_rate_end_common,
                                                                       decay_steps=decay_steps,
                                                                       power=0.5
                                                                       ),
        'amsgrad': amsgrad,
        'beta_1': beta_1,
        'beta_2': beta_2
    },
]

cols_nm = ['NMS1302 Weiden [mm/min]']


for name_model_point in names_models_point:
    name_model_save = f'{name_model_point[:-5]}_fmse={factor_mse}_niters={n_iters}_warm={warm_start}_lbda={lambda_lasso}.json'

    path_model_point = os.path.join(dir_models_point, name_model_point)
    path_save_model = get_path_make_dir(dir_save_models, name_model_save)

    use_future_rain = 'furain_True' in name_model_point
    use_rain_forecasts = 'rfore_True' in name_model_point
    interactions_only_known = 'intunknown=False' in name_model_point

    print(f'fitting {name_model_point}')

    fit_model(
        cls_model,
        path_model_point,
        path_save_model,
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        name_column_inflow,
        variables_external,
        cols_nm,
        kwargs_optimizer,
        class_optimizer,
        n_iters,
        n_iters_per_param,
        factor_mse,
        use_rain_forecasts,
        use_future_rain,
        warm_start,
        lambda_lasso
    )

