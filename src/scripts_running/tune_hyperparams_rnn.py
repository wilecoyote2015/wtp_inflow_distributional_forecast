import optuna
from src.constants.misc import CRITERIA_BIC
import json

from src.common_functions.get_data import get_data_lags_thresholds_w_interaction
from src.inflow_forecast.model.benchmark.model_rnn_separate_steps import ModelRnnSeparateTimestepsJsu
import logging
from src.common_functions.misc import get_path_make_dir

logging.basicConfig(level=logging.DEBUG)

from src.config_variables import *
from src.config_paths import *


logging.basicConfig(level=logging.DEBUG)

use_future_rain = True
do_training = True
omit_invalid_datapoints = True

cls = ModelRnnSeparateTimestepsJsu
n_samples_predict = 60

n_epochs = 100

range_n_units = [0, 100]
range_dropout = [0., 0.9]

min_lags = -6  # -6 is good

if not os.path.exists(dir_hyperparams_benchmarks):
    os.makedirs(dir_hyperparams_benchmarks)

path_dir_out = get_path_make_dir(dir_hyperparams_benchmarks, cls.__name__)


for use_rain_forecasts in [True, False]:
    file_out = os.path.join(path_dir_out, f'furain_{use_future_rain}_rfore_{use_rain_forecasts}')
    (
        data_train,
        data_test,
        bool_valid_train,
        bool_valid_test,
        thresholds,
        lags,
        columns_only_use_recent_lag
    ) = get_data_lags_thresholds_w_interaction(
        cls,
        name_column_inflow,
        variables_external,
        [name_column_rain_history],
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

    def train(trial: optuna.Trial):
        n_units = trial.suggest_int('n_units', *range_n_units)
        dropout = trial.suggest_float('dropout', *range_dropout)

        model = cls(
            n_units,
            name_column_inflow,
            n_steps_predict,
            min_lags,
            dropout=dropout
        )

        model.fit(data_train, n_epochs, bool_datapoints_valid=bool_valid_train)

        bic = model.get_criteria(
            data_train,
            bool_valid_train
        )[CRITERIA_BIC]

        return bic

    study = optuna.create_study(direction='minimize')
    study.optimize(train, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open(file_out, 'w') as f:
        json.dump({
            'params_best': trial.params,
            'trials': [
                {
                    'bic': trial_.value,
                    'params': trial_.params
                }
                for trial_ in study.trials
            ]

        },
        f)