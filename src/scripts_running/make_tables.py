import json
from src.config_paths import *

from src.common_functions.misc import load_json_gzip, json_to_params
import numpy as np

template_table = """    \\begin{{table}}
		\caption{{{caption}}}
		\centering
		\\begin{{tabular}}{{{cols} }}
			\hline
{header}
			\hline
{rows}
		\end{{tabular}}
		\label{{{label}}}
\end{{table}}
"""


def make_table_multi_datasets(
        fns_columns_model_params_from_filename: dict,
        fns_scores_from_json_subsets: {str: {str: object}},  # subset -> score
        fns_p_values_subsets: {str: {str: object}},  # subset -> score
        filepaths_scores_aggregated,
        caption,
        label,
        path_p_values,
        significance=0.05
):
    # TODO: Make compiled table with dataset column
    # raise NotImplementedError
    first_key = list(fns_scores_from_json_subsets.keys())[0]
    cols_head = ['data'] + list(fns_columns_model_params_from_filename.keys()) + list(
        fns_scores_from_json_subsets[first_key].keys())
    cols_head_names = [col_.split('[')[0] for col_ in cols_head]
    cols_head_units = ['[' + col_.split('[')[1] if len(col_.split('[')) == 2 else '' for col_ in cols_head]
    head_names = '\t\t\t' + ' & '.join(cols_head_names) + ' \\\\'
    head_units = '\t\t\t' + ' & '.join(cols_head_units) + ' \\\\'

    head = head_names + '\n' + head_units

    cols = ' '.join(['c' for col_ in cols_head])

    rows = []
    cols_rows = []

    for key_subset, fns_scores_from_json in fns_scores_from_json_subsets.items():
        fns_p_values = fns_p_values_subsets[key_subset]
        with open(path_p_values) as f:
            p_values = json.load(f)

        for filepath in filepaths_scores_aggregated:
            model_params = [
                fn_(
                    filepath
                    # os.path.basename(filepath)
                ) for fn_ in fns_columns_model_params_from_filename.values()
            ]
            # with open(filepath) as f:
            #     j = json.load(f)
            scores = [fn_(filepath) for fn_ in fns_scores_from_json.values()]

            # for each score, check if it is the best wrt dm tes
            is_best_overall = {score_: True for score_ in fns_scores_from_json.keys()}
            is_best_gamlss = {score_: True for score_ in fns_scores_from_json.keys()}
            is_gamlss = 'gamlss' in filepath
            for filepath_b in filepaths_scores_aggregated:
                if filepath_b == filepath:
                    continue

                for key_score, fn_p in fns_p_values.items():
                    p_value = fn_p(filepath, filepath_b, p_values)

                    if p_value >= significance:
                        is_best_overall[key_score] = False

                        if 'gamlss' in filepath_b:
                            is_best_gamlss[key_score] = False

            scores_formatted = []
            for score, name_score in zip(scores, fns_scores_from_json.keys()):
                score_formatted = score
                if is_best_overall[name_score]:
                    score_formatted = r'\textbf{' + str(score_formatted) + '}'
                if is_gamlss and is_best_gamlss[name_score]:
                    score_formatted = r'\textit{' + str(score_formatted) + '}'

                scores_formatted.append(score_formatted)

            cols_row = [key_subset] + model_params + scores_formatted
            cols_rows.append(cols_row)

            row = '\t\t\t' + ' & '.join(cols_row) + '\\\\'
            rows.append(row)
        rows.append('\t\t\t' + '\hline' + '\\\\')

    rows_str = '\n'.join(rows)

    return template_table.format(label=label, caption=caption, header=head, rows=rows_str, cols=cols)


def make_table(
        fns_columns_model_params_from_filename: dict,
        fns_scores_from_json,
        fns_p_values,
        filepaths_scores_aggregated,
        caption,
        label,
        path_p_values,
        significance=0.05
):
    cols_head = list(fns_columns_model_params_from_filename.keys()) + list(fns_scores_from_json.keys())
    cols_head_names = [col_.split('[')[0] for col_ in cols_head]
    cols_head_units = ['[' + col_.split('[')[1] if len(col_.split('[')) == 2 else '' for col_ in cols_head]
    head_names = '\t\t\t' + ' & '.join(cols_head_names) + ' \\\\'
    head_units = '\t\t\t' + ' & '.join(cols_head_units) + ' \\\\'

    head = head_names + '\n' + head_units

    cols = ' '.join(['c' for col_ in cols_head])

    rows = []
    cols_rows = []

    with open(path_p_values) as f:
        p_values = json.load(f)

    for filepath_scores_aggregated in filepaths_scores_aggregated:
        model_params = [
            fn_(
                filepath_scores_aggregated
                # os.path.basename(filepath)
            ) for fn_ in fns_columns_model_params_from_filename.values()
        ]

        scores = [fn_(filepath_scores_aggregated) for fn_ in fns_scores_from_json.values()]

        # for each score, check if it is the best wrt dm tes
        is_best_overall = {score_: True for score_ in fns_scores_from_json.keys()}
        is_best_gamlss = {score_: True for score_ in fns_scores_from_json.keys()}
        is_gamlss = 'gamlss' in filepath_scores_aggregated
        for filepath_b in filepaths_scores_aggregated:
            if filepath_b == filepath_scores_aggregated:
                continue

            for key_score, fn_p in fns_p_values.items():
                p_value = fn_p(filepath_scores_aggregated, filepath_b, p_values)

                if p_value >= significance:
                    is_best_overall[key_score] = False

                    if 'gamlss' in filepath_b:
                        is_best_gamlss[key_score] = False

        scores_formatted = []
        for score, name_score in zip(scores, fns_scores_from_json.keys()):
            score_formatted = score
            if is_best_overall[name_score]:
                score_formatted = r'\textbf{' + str(score_formatted) + '}'
            if is_gamlss and is_best_gamlss[name_score]:
                score_formatted = r'\textit{' + str(score_formatted) + '}'

            scores_formatted.append(score_formatted)

        cols_row = model_params + scores_formatted
        cols_rows.append(cols_row)

        row = '\t\t\t' + ' & '.join(cols_row) + '\\\\'
        rows.append(row)

    rows_str = '\n'.join(rows)

    return template_table.format(label=label, caption=caption, header=head, rows=rows_str, cols=cols)


def save_table(dir, name_table, str_table):
    path_save = os.path.join(dir, f'{name_table}.tex')

    with open(path_save, 'w') as f:
        f.write(str_table)


if __name__ == '__main__':

    # Filepaths to aggregated scores of fitted models using rain oracles
    filepaths_scores_oracle_test = []
    filepaths_scores_oracle_train = []

    # Filepaths to aggregated scores of fitted models using rain forecasts
    filepaths_scores_rfore_test = []
    filepaths_scores_rfore_train = []

    for name_model in os.listdir(dir_scores):
        dir_scores_aggregated = os.path.join(name_model, NAME_DIR_SCORES_AGGREGATED)
        for filename in os.listdir(dir_scores_aggregated):
            path_file = os.path.join(dir_scores_aggregated, filename)
            if 'rfore_True' in filename:
                if 'test' in filename:
                    filepaths_scores_rfore_test.append(path_file)
                else:
                    filepaths_scores_rfore_train.append(path_file)
            else:
                if 'test' in filename:
                    filepaths_scores_oracle_test.append(path_file)
                else:
                    filepaths_scores_oracle_train.append(path_file)

    # Filepaths to p values from DM tests
    path_p_values_oracle_train = os.path.join(dir_dm, 'oracle_train.json')
    path_p_values_oracle_test = os.path.join(dir_dm, 'oracle_test.json')
    path_p_values_rfore_test = os.path.join(dir_dm, 'rfore_test.json')
    path_p_values_rfore_train = os.path.join(dir_dm, 'rfore_train.json')

    key_rmse = r'RMSE [l/s] '
    key_mae = r'MAE [l/s] '
    key_energy = r'Energy [l/s] '


    def load_json(fp):
        with open(fp) as f:
            return json.load(f)


    def load_scores_datapoints(fp):
        fp_datapoints = fp.replace('aggregated', 'datapoints')
        return load_json_gzip(fp_datapoints)


    def get_score_rmse(fp, subset):
        # fix for old scores that do not have the rmse_mean_forecast_horizons score used in paper and
        #   used for dm-test
        scores = json_to_params(load_scores_datapoints(fp), np.asarray)
        return np.mean(scores[subset]['rmse_intraday'])


    fns_scores_all = {
        key_rmse: lambda x: str(round(get_score_rmse(x, 'all'), 2)),
        # key_rmse: lambda x: str(round(load_json(x)['all']['rmse_mean_forecast_horizons'], 2)),
        key_mae: lambda x: str(round(load_json(x)['all']['mae'], 2)),
        key_energy: lambda x: str(round(load_json(x)['all']['energy'], 2)),
    }
    fns_scores_dry = {
        key_rmse: lambda x: str(round(get_score_rmse(x, 'dry'), 2)),
        # key_rmse: lambda x: str(round(load_json(x)['dry']['rmse_mean_forecast_horizons'], 2)),
        key_mae: lambda x: str(round(load_json(x)['dry']['mae'], 2)),
        key_energy: lambda x: str(round(load_json(x)['dry']['energy'], 2)),
    }
    fns_scores_raise = {
        key_rmse: lambda x: str(round(get_score_rmse(x, 'raise'), 2)),
        # key_rmse: lambda x: str(round(load_json(x)['raise']['rmse_mean_forecast_horizons'], 2)),
        key_mae: lambda x: str(round(load_json(x)['raise']['mae'], 2)),
        key_energy: lambda x: str(round(load_json(x)['raise']['energy'], 2)),
    }
    fns_scores_constant = {
        key_rmse: lambda x: str(round(get_score_rmse(x, 'constant'), 2)),
        # key_rmse: lambda x: str(round(load_json(x)['constant']['rmse_mean_forecast_horizons'], 2)),
        key_mae: lambda x: str(round(load_json(x)['constant']['mae'], 2)),
        key_energy: lambda x: str(round(load_json(x)['constant']['energy'], 2)),
    }
    fns_scores_decline = {
        key_rmse: lambda x: str(round(get_score_rmse(x, 'decrease'), 2)),
        # key_rmse: lambda x: str(round(load_json(x)['decrease']['rmse_mean_forecast_horizons'], 2)),
        key_mae: lambda x: str(round(load_json(x)['decrease']['mae'], 2)),
        key_energy: lambda x: str(round(load_json(x)['decrease']['energy'], 2)),
    }

    fns_p_all = {
        key_rmse: lambda file_a, file_b, scores:
        scores[file_a.replace('aggregated', 'datapoints')][file_b.replace('aggregated', 'datapoints')]['all'][
            'rmse_intraday'],
        key_mae: lambda file_a, file_b, scores:
        scores[file_a.replace('aggregated', 'datapoints')][file_b.replace('aggregated', 'datapoints')]['all'][
            'mae_intraday'],
        key_energy: lambda file_a, file_b, scores:
        scores[file_a.replace('aggregated', 'datapoints')][file_b.replace('aggregated', 'datapoints')]['all'][
            'energy_intraday'],
    }
    fns_p_dry = {
        key_rmse: lambda file_a, file_b, scores:
        scores[file_a.replace('aggregated', 'datapoints')][file_b.replace('aggregated', 'datapoints')]['dry'][
            'rmse_intraday'],
        key_mae: lambda file_a, file_b, scores:
        scores[file_a.replace('aggregated', 'datapoints')][file_b.replace('aggregated', 'datapoints')]['dry'][
            'mae_intraday'],
        key_energy: lambda file_a, file_b, scores:
        scores[file_a.replace('aggregated', 'datapoints')][file_b.replace('aggregated', 'datapoints')]['dry'][
            'energy_intraday'],
    }
    fns_p_raise = {
        key_rmse: lambda file_a, file_b, scores:
        scores[file_a.replace('aggregated', 'datapoints')][file_b.replace('aggregated', 'datapoints')]['raise'][
            'rmse_intraday'],
        key_mae: lambda file_a, file_b, scores:
        scores[file_a.replace('aggregated', 'datapoints')][file_b.replace('aggregated', 'datapoints')]['raise'][
            'mae_intraday'],
        key_energy: lambda file_a, file_b, scores:
        scores[file_a.replace('aggregated', 'datapoints')][file_b.replace('aggregated', 'datapoints')]['raise'][
            'energy_intraday'],
    }
    fns_p_constant = {
        key_rmse: lambda file_a, file_b, scores:
        scores[file_a.replace('aggregated', 'datapoints')][file_b.replace('aggregated', 'datapoints')]['constant'][
            'rmse_intraday'],
        key_mae: lambda file_a, file_b, scores:
        scores[file_a.replace('aggregated', 'datapoints')][file_b.replace('aggregated', 'datapoints')]['constant'][
            'mae_intraday'],
        key_energy: lambda file_a, file_b, scores:
        scores[file_a.replace('aggregated', 'datapoints')][file_b.replace('aggregated', 'datapoints')]['constant'][
            'energy_intraday'],
    }
    fns_p_decline = {
        key_rmse: lambda file_a, file_b, scores:
        scores[file_a.replace('aggregated', 'datapoints')][file_b.replace('aggregated', 'datapoints')]['decrease'][
            'rmse_intraday'],
        key_mae: lambda file_a, file_b, scores:
        scores[file_a.replace('aggregated', 'datapoints')][file_b.replace('aggregated', 'datapoints')]['decrease'][
            'mae_intraday'],
        key_energy: lambda file_a, file_b, scores:
        scores[file_a.replace('aggregated', 'datapoints')][file_b.replace('aggregated', 'datapoints')]['decrease'][
            'energy_intraday'],
    }


    def get_model_type(path_model):
        if '/rnn/' in path_model:
            return 'LSTM'
        elif '/gamlss/' in path_model:
            return 'GAMLSS'
        elif 'sarx' in path_model:
            return 'SARX'
        else:
            raise ValueError


    def value_or_none(path_, key, fn_is_none, replacement, sep='_'):
        if key not in path_ or 'gamlss' not in path_:
            # return replacement
            return 'n.a.'
        val = path_.split(f'{key}{sep}')[1].split('_')[0]
        return replacement if fn_is_none(val) else val


    fns_params = {
        r'Model': get_model_type,
        r'$N_{\tau, \text{WTP}}$': lambda path_: value_or_none(path_, 'tpred', lambda x: x == '0', '-'),
        r'$N_{\tau, \text{net}}$': lambda path_: value_or_none(path_, 'text', lambda x: x == '0', '-'),
        r'$N_{\tau, \text{rain}}$': lambda path_: value_or_none(path_, 'train', lambda x: x == '0', '-'),
        r' $N_{\tau, \text{int}}$ ': lambda path_: value_or_none(path_, 'tinter', lambda x: x == '0', '-'),
        r'$R_{fut}$': lambda path_: 'yes' if path_.split('furain_')[1].split('_')[0] == 'True' else 'no',
        r'$R_{cum}$ ': lambda path_: 'yes' if 'cumr' in path_ and path_.split('cumr_')[1].split('_')[
            0] == '6' else 'no' if 'gamlss' in path_ else 'n.a.',
    }

    # shorter names for hbox....
    fns_params = {
        r'Model': get_model_type,
        r'$N_{\tau, \text{W}}$': lambda path_: value_or_none(path_, 'tpred', lambda x: x == '0', '-'),
        r'$N_{\tau, \text{n}}$': lambda path_: value_or_none(path_, 'text', lambda x: x == '0', '-'),
        r'$N_{\tau, \text{r}}$': lambda path_: value_or_none(path_, 'train', lambda x: x == '0', '-'),
        r' $N_{\tau, \text{i}}$ ': lambda path_: value_or_none(path_, 'tinter', lambda x: x == '0', '-'),
        r'$R_{f}$': lambda path_: 'yes' if path_.split('furain_')[1].split('_')[0] == 'True' else 'no',
        r'$R_{c}$ ': lambda path_: 'yes' if 'cumr' in path_ and path_.split('cumr_')[1].split('_')[
            0] == '6' else 'no' if 'gamlss' in path_ else 'n.a.',
    }

    # TODO: italic for best gamlss
    # suffix = r' Bold and italic: Best configurations overall and of proposed Model configurations according to Diebold-Mariano test with significance level $0.01$.'
    suffix = r' $N_{\tau, \text{W}}, N_{\tau, \text{n}}, N_{\tau, \text{r}}, N_{\tau, \text{i}}$: ' \
             r'Number of linear splines for inflow, network, rain and interactions. ' \
             r'$R_{f}, R_{c}$: Whether rain forecasts and rain cumulation are used. ' \
             r'Bold and italic: Statistically significant best scores of all models and proposed model configurations.'

    table = make_table(
        fns_params,
        fns_scores_constant,
        fns_p_constant,
        filepaths_scores_oracle_test,
        r'In-sample scores for different configurations of the proposed model for inflow near hydraulic capacity limit after rain events.' + suffix,
        'tbl_oracle_constant_in_sample',
        path_p_values_oracle_train
    )
    save_table(dir_tables, 'tbl_oracle_constant_in_sample', table)
    print(table)
    print('\n')
    #
    table = make_table(
        fns_params,
        fns_scores_constant,
        fns_p_constant,
        filepaths_scores_oracle_train,
        r'Out-of-sample scores for different configurations of the proposed model with rain oracles for inflow near hydraulic capacity limit after rain events.' + suffix,
        'tbl_oracle_constant_out_sample',
        path_p_values_oracle_test
    )
    save_table(dir_tables, 'tbl_oracle_constant_out_sample', table)
    print(table)
    print('\n')

    table = make_table(
        fns_params,
        fns_scores_constant,
        fns_p_constant,
        filepaths_scores_rfore_test,
        r'Out-of-sample scores for different configurations of the proposed model with rain forecasts for inflow near hydraulic capacity limit after rain events.' + suffix,
        'tbl_rfore_constant_out_sample',
        path_p_values_rfore_test
    )
    save_table(dir_tables, 'tbl_rfore_constant_out_sample', table)
    print(table)
    print('\n')

    fns_scores_subsets_main = {
        'all': fns_scores_all,
        'dry': fns_scores_dry,
        'rise': fns_scores_raise,
        'decline': fns_scores_decline,
    }
    fns_p_subsets_main = {
        'all': fns_p_all,
        'dry': fns_p_dry,
        'rise': fns_p_raise,
        'decline': fns_p_decline,
    }

    fns_scores_subsets_all = {
        'all': fns_scores_all,
        'dry': fns_scores_dry,
        'rise': fns_scores_raise,
        'constant': fns_scores_constant,
        'decline': fns_scores_decline,
    }
    fns_p_subsets_all = {
        'all': fns_p_all,
        'dry': fns_p_dry,
        'rise': fns_p_raise,
        'constant': fns_p_constant,
        'decline': fns_p_decline,
    }

    #### CUMULATED TABLES

    ### Paper content
    table = make_table_multi_datasets(
        fns_params,
        fns_scores_subsets_main,
        fns_p_subsets_main,
        filepaths_scores_rfore_test,
        r'Out-of-sample scores for different configurations of the proposed model with rain forecasts.' + suffix,
        'tbl_cumulated_all_dry_constant_decrease_rfore_out_sample',
        path_p_values_rfore_test
    )
    save_table(dir_tables, 'tbl_cumulated_all_dry_constant_decrease_rfore_out_sample', table)
    print(table)
    print('\n')

    table = make_table_multi_datasets(
        fns_params,
        fns_scores_subsets_main,
        fns_p_subsets_main,
        filepaths_scores_oracle_train,
        r'Out-of-sample scores for different configurations of the proposed model with rain oracles.' + suffix,
        'tbl_cumulated_all_dry_constant_decrease_roracle_out_sample',
        path_p_values_oracle_test
    )
    save_table(dir_tables, 'tbl_cumulated_all_dry_constant_decrease_roracle_out_sample', table)
    print(table)
    print('\n')

    ### Appendix

    table = make_table_multi_datasets(
        fns_params,
        fns_scores_subsets_all,
        fns_p_subsets_all,
        filepaths_scores_rfore_test,
        r'In-sample scores for different configurations of the proposed model with rain forecasts.' + suffix,
        'tbl_cumulated_all_phases_rfore_in_sample',
        path_p_values_rfore_test
    )
    save_table(dir_tables, 'tbl_cumulated_all_phases_rfore_in_sample', table)
    print(table)
    print('\n')

    table = make_table_multi_datasets(
        fns_params,
        fns_scores_subsets_main,
        fns_p_subsets_main,
        filepaths_scores_oracle_train,
        r'In-sample scores for different configurations of the proposed model with rain oracles.' + suffix,
        'tbl_cumulated_all_dry_constant_decrease_roracle_in_sample',
        path_p_values_oracle_test
    )
    save_table(dir_tables, 'tbl_cumulated_all_dry_constant_decrease_roracle_in_sample', table)
    print(table)
    print('\n')
