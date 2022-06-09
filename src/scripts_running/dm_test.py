import json
from collections import defaultdict

import pandas as pd

from src.common_functions.misc import load_json_gzip, json_to_params
from src.config_paths import *
import os

import numpy as np
from scipy.stats import t

from tqdm import tqdm

#### SET THE FOLOWING VARIABLES
filepaths_scores_datapoints_oracle_train = []
filepaths_scores_datapoints_oracle_test = []
filepaths_scores_datapoints_rfore_test = []
filepaths_scores_datapoints_rfore_train = []

for name_dir_scores_model in os.listdir(dir_scores):
    path_dir_scores_model = os.path.join(name_dir_scores_model, name_dir_scores_model)

    for name_scores in os.listdir(path_dir_scores_model):
        if 'rfore_True' in name_scores:
            if 'test' in name_scores:
                filepaths_scores_datapoints_rfore_test.append(
                    os.path.join(path_dir_scores_model, name_scores)
                )
            else:
                filepaths_scores_datapoints_rfore_train.append(
                    os.path.join(path_dir_scores_model, name_scores)
                )
        else:
            if 'test' in name_scores:
                filepaths_scores_datapoints_oracle_test.append(
                    os.path.join(path_dir_scores_model, name_scores)
                )
            else:
                filepaths_scores_datapoints_oracle_train.append(
                    os.path.join(path_dir_scores_model, name_scores)
                )

dir_out_p_values = dir_dm
if not os.path.exists(dir_out_p_values):
    os.makedirs(dir_out_p_values)



# for each model, load the scores for all datapoints and compare the means with all other models.
#   also give the best model for each score as well as the best own model.

def dm_test(loss_a, loss_b, hmax=1):
    """
        errors: [datapoint, timestep]
    """
    # as dm_test with alternative == "less"
    delta = loss_a - loss_b
    # estimation of the variance
    delta_var = np.var(delta) / delta.shape[0]
    statistic = delta.mean() / np.sqrt(delta_var)
    delta_length = delta.shape[0]
    k = ((delta_length + 1 - 2 * hmax + (hmax / delta_length)
          * (hmax - 1)) / delta_length) ** (1 / 2)
    statistic = statistic * k
    p_value = t.cdf(statistic, df=delta_length - 1)

    return statistic, p_value


def load_json(filename):
    with open(filename) as f:
        result = json.load(f)
    return result


def calc_dms(filepaths, niveau=0.01):
    # p_values are for h0 that a is better than b
    p_values = {}
    a_better_but_not_significant = defaultdict(lambda: defaultdict(list))
    for idx, filepath_a in tqdm(enumerate(filepaths)):
        print(f'Calc dms for filepath {filepath_a}')
        p_values_filepath_a = {}
        scores_a = json_to_params(load_json_gzip(filepath_a), np.asarray)
        scores_a_avg = load_json(filepath_a.replace('datapoints', 'aggregated'))

        path_datetimes_a = filepath_a.replace('scores', 'datetimes').replace('/datapoints', '')
        with open(path_datetimes_a) as f:
            datetimes_a = np.asarray(json.load(f)[1:])
        path_indices_subsets_a = filepath_a.replace('scores', 'indices_subsets').replace('/datapoints', '')
        indices_subsets_a = json_to_params(load_json_gzip(path_indices_subsets_a), np.asarray)

        path_indices_valid_a = filepath_a.replace('scores', 'indices_valid').replace('/datapoints', '')
        indices_valid_a = json_to_params(load_json_gzip(path_indices_valid_a), np.asarray)

        for filepath_b in filepaths:
            scores_b = json_to_params(load_json_gzip(filepath_b), np.asarray)
            scores_b_avg = load_json(filepath_b.replace('datapoints', 'aggregated'))
            path_datetimes_b = filepath_b.replace('scores', 'datetimes').replace('/datapoints', '')
            with open(path_datetimes_b) as f:
                datetimes_b = np.asarray(json.load(f)[1:])

            path_indices_subsets_b = filepath_b.replace('scores', 'indices_subsets').replace('/datapoints', '')
            indices_subsets_b = json_to_params(load_json_gzip(path_indices_subsets_b), np.asarray)

            path_indices_valid_b = filepath_b.replace('scores', 'indices_valid').replace('/datapoints', '')
            indices_valid_b = json_to_params(load_json_gzip(path_indices_valid_b), np.asarray)

            p_values_subsets = {}
            for key_subset, scores_subset_a in scores_a.items():
                p_values_subset = {}
                indices_subsets_b_subset = indices_subsets_b[key_subset]
                if not isinstance(indices_subsets_b_subset, np.ndarray):
                    indices_subsets_b_subset = np.asarray(list(indices_subsets_b_subset.values()))
                indices_subsets_a_subset = indices_subsets_a[key_subset]
                if not isinstance(indices_subsets_a_subset, np.ndarray):
                    indices_subsets_a_subset = np.asarray(list(indices_subsets_a_subset.values()))

                bool_use_a = np.zeros_like(indices_subsets_a_subset, dtype=bool)
                bool_use_b = np.zeros_like(indices_subsets_b_subset, dtype=bool)
                bool_use_a[indices_valid_a] = True
                bool_use_b[indices_valid_b] = True
                bool_use_a = np.logical_and(bool_use_a, indices_subsets_a_subset)
                bool_use_b = np.logical_and(bool_use_b, indices_subsets_b_subset)

                for key_score, scores_datapoints_a in scores_subset_a.items():
                    scores_datapoints_b = scores_b[key_subset][key_score]

                    datetimes_a_use = datetimes_a[bool_use_a]
                    datetimes_b_use = datetimes_b[bool_use_b]

                    scores_datapoints_a_df = pd.DataFrame(scores_datapoints_a, index=datetimes_a_use)
                    scores_datapoints_b_df = pd.DataFrame(scores_datapoints_b, index=datetimes_b_use)

                    scores_joined = scores_datapoints_a_df.join(
                        scores_datapoints_b_df,
                        lsuffix='_a',
                        rsuffix='_b',
                        how='inner'
                    )

                    scores_aligned_a = scores_joined['0_a']
                    scores_aligned_b = scores_joined['0_b']

                    stat, p_value = dm_test(scores_aligned_a, scores_aligned_b, )

                    key_score_avg = key_score.replace('_intraday', '')
                    score_a_avg = scores_a_avg[key_subset][key_score_avg]
                    score_b_avg = scores_b_avg[key_subset][key_score_avg]

                    if score_a_avg < score_b_avg and not np.isnan(p_value) and p_value > niveau:
                        print(
                            f'{key_score_avg} for a better than b, but insignificant, with p value {p_value}: {os.path.basename(filepath_a)}, {os.path.basename(filepath_b)}')
                        a_better_but_not_significant[key_subset][key_score_avg].append(
                            [filepath_a, filepath_b, score_a_avg, score_b_avg, p_value])

                    p_values_subset[key_score] = p_value
                p_values_subsets[key_subset] = p_values_subset

            p_values_filepath_a[filepath_b] = p_values_subsets
        p_values[filepath_a] = p_values_filepath_a

    return p_values


p_values_test = calc_dms(filepaths_scores_datapoints_oracle_test)
with open(os.path.join(dir_out_p_values, 'oracle_test.json'), 'w') as f:
    json.dump(p_values_test, f, indent=2)


p_values_train = calc_dms(filepaths_scores_datapoints_oracle_train)
with open(os.path.join(dir_out_p_values, 'oracle_train.json'), 'w') as f:
    json.dump(p_values_train, f, indent=2)


p_values_test = calc_dms(filepaths_scores_datapoints_rfore_test)
with open(os.path.join(dir_out_p_values, 'rfore_test.json'), 'w') as f:
    json.dump(p_values_test, f, indent=2)

p_values_train = calc_dms(filepaths_scores_datapoints_rfore_train)
with open(os.path.join(dir_out_p_values, 'rfore_train.json'), 'w') as f:
    json.dump(p_values_train, f, indent=2)
