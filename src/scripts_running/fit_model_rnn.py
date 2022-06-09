from src.common_functions.get_data import get_data_lags_thresholds_w_interaction
from src.inflow_forecast.model.benchmark.model_rnn_separate_steps import ModelRnnSeparateTimestepsJsu
import logging

logging.basicConfig(level=logging.DEBUG)

from src.config_variables import *
from src.config_paths import *

use_rain_forecasts = True
use_future_rain = True
do_training = True
omit_invalid_datapoints = True
cls_model = ModelRnnSeparateTimestepsJsu


weight_mse = 10.
n_epochs = 100

n_units = 10  # best trial 35
dropout = 0.0001  # best trial 35

name_model_save = f'l_{min_lag}_furain_{use_future_rain}_rfore_{use_rain_forecasts}_mse_{weight_mse}_units_{n_units}_dopout_{dropout}.json'
dir_save_models = os.path.join(dir_models, cls_model.__name__)
if not os.path.exists(dir_save_models):
    os.makedirs(dir_save_models)
path_save_model = os.path.join(dir_save_models, name_model_save)


column_predict = name_column_inflow
COL_NM = name_column_rain_history

(
    data_train,
    data_test,
    bool_valid_train,
    bool_valid_test,
    thresholds,
    lags,
    columns_only_use_recent_lag
) = get_data_lags_thresholds_w_interaction(
    cls_model,
    column_predict,
    variables_external,
    [COL_NM],
    min_lag,
    1,
    1,
    1,
    1,
    'x',
    False,
    date_start_train,
    date_end_train,
    date_start_test,
    date_end_test,
    use_rain_forecasts,
    use_future_rain,
    n_steps_predict,
    True,
    path_csv_data_wtp_network_rain,
    path_csv_forecast_rain_radar,
)

if do_training:
    model = cls_model(
        n_units,
        column_predict,
        n_steps_predict,
        min_lag
    )

    model.fit(data_train, n_epochs, data_validate=None, bool_datapoints_valid=bool_valid_train, weight_mse=weight_mse)

    # Test saving and loading model
    # TODO
    model.save_model(path_save_model)
    # model = cls.model_from_file(path_save_model)


else:
    model = cls_model.model_from_file(path_save_model)

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
    COL_NM,
    bool_datapoints_valid_train=bool_valid_train,
    bool_datapoints_valid_test=bool_valid_test,
    include_intraday=False,
    n_samples_predict=60
)
print('scores train:')
print(scores_train)
print('scores test:')
print(scores_test)
print('Criteria:')
print(criteria)