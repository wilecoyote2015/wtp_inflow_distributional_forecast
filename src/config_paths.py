import os

NAME_DIR_SCORES = 'scores'
NAME_DIR_SCORES_AGGREGATED = 'aggregated'
NAME_DIR_SCORES_DATAPOINTS = 'datapoints'
NAME_DIR_SCORES_TIMESTEPS = 'timesteps'
NAME_DIR_MODELS = 'models'
NAME_DIR_DM = 'dm_test'
NAME_DIR_GRID_SEARCH = 'grid_search'
NAME_DIR_SAMPLES = 'samples'
NAME_DIR_TRUTH = 'truth'
NAME_DIR_INDICES_VALID = 'indices_valid'
NAME_DIR_INDICES_SUBSETS = 'indices_subsets'
NAME_DIR_DATETIMES_X = 'datetimes_x'
NAME_DIR_CRITERIA = 'criteria'
NAME_DIR_HYPERPARAMS = 'hyperparameters_benchmarks'
NAME_DIR_TABLES = 'tables'
NAME_DIR_PLOTS = 'plots'
SUFFIX_TEST = 'test'
SUFFIX_TRAIN = 'train'

# Directory to the results
dir_results = '/home/bjoern/PycharmProjects/wtp_gamlss_publication/test/results'

# extracted radolan radar images
dir_raster_radar_historical = ''

# TODO: use importlib for resources
path_csv_forecast_rain_radar = '/home/bjoern/PycharmProjects/wtp_gamlss_publication/data/radar_forecast.csv'
path_csv_data_wtp_network_rain = '/home/bjoern/PycharmProjects/wtp_gamlss_publication/data/wtp_network_rain.csv'

# Model names to use for plotting.
# This corresponds to the names of the files in the models result directory.
# TODO!!!! INTERRAINRAIN best fit!
# TODO: set to None
name_model_gamlss_best_fit_oracles = 'l_-6_tpred_15_text_15_train_5_tinter_3_furain_True_rfore_False_cumr_6_intunknown=True_shrunken_fmse=10_niters=20000_warm=True_lbda=0test.json'
name_model_gamlss_best_fit_rfore = 'l_-6_tpred_15_text_15_train_5_tinter_3_furain_True_rfore_True_cumr_6_intunknown=True_shrunken_fmse=10_niters=20000_warm=True_lbda=0test.json'
name_model_gamlss_no_thresholds_oracles = 'l_-6_tpred_1_text_1_train_1_tinter_0_furain_True_rfore_False_cumr_None_intunknown=True_shrunken_fmse=10_niters=20000_warm=True_lbda=0test.json'
name_model_gamlss_3_thresholds_oracles = 'l_-6_tpred_3_text_3_train_3_tinter_0_furain_True_rfore_False_cumr_None_intunknown=True_shrunken_fmse=10_niters=20000_warm=True_lbda=0test.json'
name_model_gamlss_only_wtp_oracles = 'l_-6_tpred_1_text_0_train_0_tinter_0_furain_False_rfore_False_cumr_None_intunknown=True_shrunken_fmse=10_niters=20000_warm=True_lbda=0test.json'
name_model_gamlss_lstm_oracles = 'l_-6_furain_True_rfore_False_mse_10.0_units_10_dopout_0.0001test.json'
name_model_gamlss_sarx_oracles = 'sarx_l_-6_tpred_1_text_1_train_1_tinter_0_furain_True_rfore_False_cumr_None_intunknown=Truetest.json'
name_model_gamlss_lstm_rfore = 'l_-6_furain_True_rfore_True_mse_10.0_units_10_dopout_0.0001test.json'
name_model_gamlss_sarx_rfore = 'sarx_l_-6_tpred_1_text_1_train_1_tinter_0_furain_True_rfore_True_cumr_None_intunknown=Truetest.json'


for var in [
    dir_results,
    dir_raster_radar_historical,
    path_csv_forecast_rain_radar,
    path_csv_data_wtp_network_rain,
    name_model_gamlss_best_fit_oracles,
    name_model_gamlss_best_fit_rfore,
    name_model_gamlss_no_thresholds_oracles,
    name_model_gamlss_3_thresholds_oracles,
    name_model_gamlss_only_wtp_oracles,
    name_model_gamlss_lstm_oracles,
    name_model_gamlss_sarx_oracles,
    name_model_gamlss_lstm_rfore,
    name_model_gamlss_sarx_rfore
]:
    if var is None:
        raise ValueError(f'Please set all variables in the configuration file "config_paths.py"')

dir_scores = os.path.join(dir_results, NAME_DIR_SCORES)
dir_samples = os.path.join(dir_results, NAME_DIR_SAMPLES)
dir_criteria = os.path.join(dir_results, NAME_DIR_CRITERIA)
dir_datetimes_x = os.path.join(dir_results, NAME_DIR_DATETIMES_X)
dir_models = os.path.join(dir_results, NAME_DIR_MODELS)
dir_dm = os.path.join(dir_results, NAME_DIR_DM)
dir_grid_search = os.path.join(dir_results, NAME_DIR_GRID_SEARCH)
dir_truth = os.path.join(dir_results, NAME_DIR_TRUTH)
dir_indices_valid = os.path.join(dir_results, NAME_DIR_INDICES_VALID)
dir_indices_subsets = os.path.join(dir_results, NAME_DIR_INDICES_SUBSETS)
dir_hyperparams_benchmarks = os.path.join(dir_results, NAME_DIR_HYPERPARAMS)
dir_tables = os.path.join(dir_results, NAME_DIR_TABLES)
dir_plots = os.path.join(dir_results, NAME_DIR_PLOTS)