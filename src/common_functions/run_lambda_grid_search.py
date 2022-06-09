import numpy as np
from collections import defaultdict


def make_grid(value_center, stepsize, n_steps, dtype):
    exp_center = -np.log10(value_center)
    exp_start = exp_center - stepsize * n_steps / 2
    exp_last = exp_center + stepsize * n_steps / 2
    steps_grid = np.linspace(exp_start, exp_last, n_steps, endpoint=False, dtype=dtype)

    grid: np.ndarray = np.power(10., -steps_grid)

    return np.concatenate([grid, np.asarray([0.])])


def grid_search_lambda(class_model, value_grid_start, step_exp, n_steps_grid, df_in_sample, df_out_sample, col_nm,
                       *args_model,
                       bool_valid_test=None,
                       bool_valid_train=None,
                       **kwargs_model
                       ):
    grid: np.ndarray = make_grid(value_grid_start, step_exp, n_steps_grid, class_model.DTYPE)

    print(f'grid: {grid.tolist()}')

    scores_lambdas = {
        'scores': {},
        'best': None
    }

    def get_score_lambda(scores):
        return scores['criteria']['bic']

    scores_history = []
    model_best = None
    score_best = None

    for lambda_ in grid:
        model_ = class_model(
            *args_model,
            **kwargs_model,
            lambdas_lasso=lambda_
        )

        model_.fit(df_in_sample, bool_datapoints_valid=bool_valid_train)

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
        ) = model_.evaluate_model(
            df_in_sample,
            df_out_sample,
            col_nm,
            bool_datapoints_valid_train=bool_valid_train,
            bool_datapoints_valid_test=bool_valid_test,
            include_intraday=False,
            n_samples_predict=None,
            include_criteria=True
        )

        scores_lambdas['scores'][lambda_] = {
            'scores_train': scores_train,
            'scores_test': scores_test,
            'criteria': criteria
        }

        print(f'scores for lambda {lambda_}: {scores_lambdas["scores"][lambda_]}')

        score_ = get_score_lambda(scores_lambdas['scores'][lambda_])
        scores_history.append(score_)

        if score_best is None or score_ < score_best:
            score_best = score_
            model_best = model_

        if len(scores_history) > 2:
            if scores_history[-3] < scores_history[-2] and scores_history[-2] < scores_history[-1]:
                print('Aborting as scores do not improve anymore.')
                if len(scores_history) == 3:
                    print('WARNING: No local minimum found! Best lambda is higher than max lambda')
                break

        # for some reason, memory is leaked
        del scores_train,
        del scores_test,
        del criteria,
        del predictions_train,
        del predictions_test,
        del truth_train,
        del truth_test,
        del indices_valid_train,
        del indices_valid_test,
        del indices_subsets_train,
        del indices_subsets_test,
        del datetimes_x_train,
        del datetimes_x_test,
        del x_train,
        del x_test
        del model_

    lambdas_sorted_best = sorted(scores_lambdas['scores'].keys(),
                                 key=lambda key_: get_score_lambda(scores_lambdas['scores'][key_])
                                 )
    lambda_best = lambdas_sorted_best[0]
    if lambda_best == grid[-1]:
        print('WARNING: last lambda ist best. No local minimum found! real best lambda is smaller than smallest.')
    scores_lambdas['best'] = lambda_best
    print(f'best lambda: {lambda_best}')

    return scores_lambdas, lambda_best, model_best


def grid_search_lambda_consecutive_timesteps(class_model, values_grid_start_timesteps_params_pdf,
                                             steps_exp_timesteps_params_pdf, n_steps_grid, df_in_sample, df_out_sample,
                                             score_use, *args_model,
                                             data_nans_invalid_out_sample=None,
                                             data_nans_invalid_sample=None,
                                             path_save_steps=None,
                                             **kwargs_model):
    """
        Lambdas for individual timesteps. Start with first timestep and then make the following
    """

    lambdas = values_grid_start_timesteps_params_pdf
    result = defaultdict(dict)
    for idx_timestep, (values_grid_start_params_pdf, steps_exp_timestep_params_pdf) in enumerate(
            zip(values_grid_start_timesteps_params_pdf, steps_exp_timesteps_params_pdf)):
        for idx_param_pdf, (value_grid_start, step_size_exp) in enumerate(
                zip(values_grid_start_params_pdf, steps_exp_timestep_params_pdf)):
            grid: np.ndarray = make_grid(value_grid_start, step_size_exp, n_steps_grid, class_model.DTYPE)

            print(f'grid: {grid.tolist()}')

            scores_lambdas = {
                'scores': {},
                'best': None
            }

            kwargs_timestep = {
                **kwargs_model,
                'lambdas_lasso': lambdas[:idx_timestep + 1],
                'n_steps_forecast_horizon': idx_timestep + 1,
                'lags_columns': {col: lags_[:idx_timestep + 1] for col, lags_
                                 in kwargs_model['lags_columns'].items()},
                'thresholds': {col: thresh_[:idx_timestep + 1]
                               for col, thresh_ in kwargs_model[
                                   'thresholds'].items()}
            }

            for lambda_ in grid:
                lambdas[idx_timestep][idx_param_pdf] = lambda_
                model_ = class_model(
                    *args_model,
                    **kwargs_timestep,

                )
                (
                    scores_model,
                    predictions_in_sample,
                    predictions_out_sample,
                    x_in_sample,
                    x_out_sample,
                    truth_in_sample,
                    truth_out_sample
                ) = model_.fit_get_scores_model(df_in_sample, df_out_sample, scalar_only=True,
                                                data_nans_invalid_out_sample=data_nans_invalid_out_sample,
                                                data_nans_invalid_sample=data_nans_invalid_sample, )

                scores_lambdas['scores'][lambda_] = scores_model

                print(f'Results Timestep {idx_timestep}, param {idx_param_pdf}, lambda {lambda_}:')
                print(f'Out Sample {score_use}: {scores_model["out_sample"][score_use]}')
                print(f'In Sample {score_use}: {scores_model["in_sample"][score_use]}')

            lambdas_sorted_best = sorted(scores_lambdas['scores'].keys(),
                                         key=lambda key_: scores_lambdas['scores'][key_]['out_sample'][score_use])
            lambda_best = lambdas_sorted_best[0]
            print(f'best lambda for timestep {idx_timestep}, param {idx_param_pdf}: {lambda_best}')
            scores_lambdas['best'] = lambda_best
            lambdas[idx_timestep][idx_param_pdf] = lambda_best

            result[idx_timestep][idx_param_pdf] = scores_lambdas

    print(f'best lambdas: {lambdas}')

    return result, lambdas


def format_lambdas_vec(lambdas_vec, n_params_pdf, n_steps_forecast_horizon):
    lambdas_formatted = []
    idx_current = 0
    for idx_timestep in range(n_steps_forecast_horizon):
        lambdas_formatted_timestep = []
        for idx_param_pdf in range(n_params_pdf):
            lambdas_formatted_timestep.append(lambdas_vec[idx_current])
            idx_current += 1
        lambdas_formatted.append(lambdas_formatted_timestep)
    return lambdas_formatted
