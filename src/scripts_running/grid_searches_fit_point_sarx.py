import json
from src.common_functions.run_lambda_grid_search import grid_search_lambda
from src.inflow_forecast.model.point_and_sarx.model_point import ModelPointBase
from src.common_functions.get_data import get_data_lags_thresholds_w_interaction
from src.common_functions.misc import params_to_json
from os.path import join
from src.config_variables import *
from src.config_paths import *

SEASONS_DAYS_WEEKS = [[0, 1, 2, 3, 4, 5, 6]]
N_SEASONS_ANNUAL = 1

def do_grid_seach_model(
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        lags_target_additional,
        column_predict,
        variables_external,
        cols_nm,
        dir_save_result,
        dir_save_best_model,
        lambda_start,
        class_model,
        min_lag,
        thresh_prediction,
        thresh_external,
        thresh_rain,
        thresh_interactions,
        use_future_rain,
        use_rain_forecasts,
        steps_cum_rain,
        interactions_only_known_data
):
    filename = f'sarx_l_{min_lag}_tpred_{thresh_prediction}_text_{thresh_external}_train_{thresh_rain}_tinter_{thresh_interactions}_furain_{use_future_rain}_rfore_{use_rain_forecasts}_cumr_{steps_cum_rain}_intunknown={not interactions_only_known_data}.json'

    print(f'grid search for {filename}')

    path_save_result = join(dir_save_result, filename)
    path_save_best_fit = join(dir_save_best_model, filename)
    path_save_best_fit_shrunken = join(dir_save_best_model, filename[:-5] + '_shrunken.json')

    (
        data_train,
        data_test,
        bool_valid_train,
        bool_valid_test,
        thresholds_,
        lags_,
        columns_only_use_recent_lag
    ) = get_data_lags_thresholds_w_interaction(
        ModelPointBase,
        column_predict,
        variables_external,
        cols_nm,
        min_lag,
        thresh_prediction,
        thresh_external,
        thresh_rain,
        thresh_interactions,
        'x',
        True,
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        use_rain_forecasts,
        use_future_rain,
        STEPS_PREDICT,
        True,
        path_csv_data_wtp_network_rain,
        path_csv_forecast_rain_radar,
        lags_target_additional=lags_target_additional,
        # 5 culmulate, 5 max: 30.39
        steps_cumulate_rain=steps_cum_rain,  # 6 is good with linear thresholds, 5 with quantiles.
        steps_median_rain=None,  # 6 is good
        # steps_median_rain=6,
        steps_max_rain=None,  # None
        keep_raw_rain=True,  # True
        min_lags_interactions=-1,
        interactions_only_known_data=interactions_only_known_data,
    )

    # for sarx, exogenous variables only have one lag.
    lags, thresholds = {}, {}

    for name_variable in lags_.keys():
        idx_pdf = 0

        if name_variable == column_predict:
            lags[name_variable] = lags_[name_variable]
            thresholds[name_variable] = thresholds_[name_variable]
        else:
            lags[name_variable] = []
            thresholds[name_variable] = []
            for idx_timestep, (lags_timestep, thresholds_timestep) in enumerate(zip(lags_[name_variable], thresholds_[name_variable])):
                lags_timestep_,  thresholds_timestep_ = [[]], [[]]
                lags_param, thresholds_param = lags_timestep[idx_pdf], thresholds_timestep[idx_pdf]

                for lags_threshold, threshold in zip(lags_param, thresholds_param):
                    for lag_threshold in lags_threshold:
                        if lag_threshold == -idx_timestep - 1:
                            lags_timestep_[0].append([lag_threshold])

                    thresholds_timestep_[0].append(threshold)

                lags[name_variable].append(lags_timestep_)
                thresholds[name_variable].append(thresholds_timestep_)


    args_model = dict(
            variable_target=column_predict,
            n_steps_forecast_horizon=n_steps_predict,
            lags_columns=lags,
            thresholds=thresholds,
            seasons_days_weeks=SEASONS_DAYS_WEEKS,
            n_seasons_annual=N_SEASONS_ANNUAL,
        )

    scores_lambdas, lambda_best, model_best = grid_search_lambda(
        class_model,
        df_in_sample=data_train,
        df_out_sample=data_test,
        col_nm=cols_nm[0],
        value_grid_start=lambda_start,
        step_exp=0.2,
        n_steps_grid=15,
        bool_valid_train=bool_valid_train,
        bool_valid_test=bool_valid_test,
        **args_model
    )

    model_best.save_model(path_save_best_fit)
    model_best.save_model_shrunken(path_save_best_fit_shrunken, 0.001)

    with open(path_save_result, 'w') as f:
        json.dump(
            params_to_json(scores_lambdas),
            f,
            indent=4
        )
    del model_best
    del scores_lambdas
    del data_train
    del data_test
    del bool_valid_train
    del bool_valid_test




# lbda_start, class model, min_lag, thresh_pred, thresh_ext, thresh_rain, thresh_interactions, use_future_rain, use_rain_forecasts, steps_cum_rain, interactions_only_known_data
cls = ModelPointBase
min_lag = -6
data_models = [
    (
        1e-4,
        cls,
        min_lag,
        1, 1, 1, 0,
        True,
        False,
        None,
        False
    ),
    (
        1e-4,
        cls,
        min_lag,
        1, 1, 1, 0,
        True,
        True,
        None,
        False
    ),
]


lags_target_additional = []

dir_save_models = os.path.join(dir_models, cls.__name__)
dir_save_results = os.path.join(dir_grid_search, cls.__name__)
for args in data_models:
    do_grid_seach_model(
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        lags_target_additional,
        name_column_inflow,
        variables_external,
        [name_column_rain_history],
        dir_save_results,
        dir_save_models,
        *args
    )