import json

from src.common_functions.run_lambda_grid_search import grid_search_lambda
from src.inflow_forecast.model.point_and_sarx.model_point import ModelPointBase
from src.common_functions.get_data import get_data_lags_thresholds_w_interaction
from src.common_functions.misc import params_to_json, get_path_make_dir
from os.path import join
from src.config_variables import *
from src.config_paths import *

SEASONS_DAYS_WEEKS = [[0, 1, 2, 3, 4], [5], [6]]
N_SEASONS_ANNUAL = 4

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
        interactions_only_known_data,
        furain_all_lags,
        interactions_rain_w_rain
):
    filename = f'l_{min_lag}_tpred_{thresh_prediction}_text_{thresh_external}_train_{thresh_rain}_tinter_{thresh_interactions}_furain_{use_future_rain}_rfore_{use_rain_forecasts}_cumr_{steps_cum_rain}_intunknown={not interactions_only_known_data}_interrainrain_{interactions_rain_w_rain}.json'

    print(f'grid search for {filename}')


    # interactions_rain_w_rain = True
    interact_also_raw_rain = True
    interact_raw_with_cumulated = False

    path_save_result = join(dir_save_result, filename)
    path_save_best_fit = join(dir_save_best_model, filename)
    path_save_best_fit_shrunken = join(dir_save_best_model, filename[:-5] + '_shrunken.json')

    (
        data_train,
        data_test,
        bool_valid_train,
        bool_valid_test,
        thresholds,
        lags,
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
        n_steps_predict,
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
        furain_all_lags=furain_all_lags,
        interactions_rain_w_rain=interactions_rain_w_rain,
        interact_also_raw_rain=interact_also_raw_rain,
        interact_raw_with_cumulated=interact_raw_with_cumulated
    )

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
    model_best.save_model_shrunken(path_save_best_fit_shrunken, 0.0001)

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




# lbda_start, class model, min_lag, thresh_pred, thresh_ext, thresh_rain, thresh_interactions, use_future_rain, use_rain_forecasts, steps_cum_rain, interactions_only_known_data, lags_all_future_rain, interaction rain with rain
cls = ModelPointBase
data_models = [
    ### Oracles
    ( 1e-5,     cls,    min_lag,    1,      0,      0,      0,      False,      False,      None,       False,      False,      False),
    ( 1e-5,     cls,    min_lag,    1,      1,      0,      0,      False,      False,      None,       False,      False,      False),
    ( 1e-5,     cls,    min_lag,    1,      1,      1,      0,      False,      False,      None,       False,      False,      False),
    ( 1e-5,     cls,    min_lag,    1,      1,      1,      0,      True,       False,      None,       False,      False,      False),
    ( 1e-5,     cls,    min_lag,    3,      3,      3,      0,      True,       False,      None,       False,      False,      False),
    ( 1e-5,     cls,    min_lag,    15,     15,     5,      0,      True,       False,      None,       False,      False,      False),
    ( 1e-5,     cls,    min_lag,    15,     15,     5,      0,      True,       False,      6,          False,      False,      False),
    ( 1e-5,     cls,    min_lag,    15,     15,     5,      3,      True,       False,      None,       False,      False,      False),
    ( 1e-5,     cls,    min_lag,    15,     15,     5,      3,      True,       False,      6,          False,      False,      False),
    ### Rain Forecast
    ( 1e-5,     cls,    min_lag,    15,     15,     5,      3,      True,       True,       6,          False,      True,       True),
]

lags_target_additional = []

cols_nm = [name_column_rain_history]

dir_save_models = get_path_make_dir(dir_models, cls.__name__)
dir_save_results = get_path_make_dir(dir_grid_search, cls.__name__)
for args in data_models:
    if len(args) == 11:
        # Add furain all lags and rain rain inter
        args = (*args, False, False)
    do_grid_seach_model(
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        lags_target_additional,
        name_column_inflow,
        variables_external,
        cols_nm,
        dir_save_results,
        dir_save_models,
        *args
    )