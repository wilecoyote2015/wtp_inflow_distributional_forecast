import shutil
from sklearn.linear_model import LinearRegression
import math
import numpy as np
import matplotlib.patches as mpatches
import os
from code.common_functions.misc import load_json_gzip, json_to_params, make_dir_if_missing
import pickle
import json
from tensorflow_graphics.math.interpolation import bspline
from src.config_paths import *
from src.config_variables import *
from src.inflow_forecast.model.gamlss.models_gamlss import ModelGamlssJsu
from src.inflow_forecast.model.point_and_sarx.model_point import ModelPointBase
from src.inflow_forecast.model.benchmark.model_rnn_separate_steps import ModelRnnSeparateTimestepsJsu


from matplotlib import pyplot as plt
import matplotlib

import pandas as pd
from src.common_functions.get_data import get_data
from datetime import datetime, timedelta
from src.constants.params_preprocessing import PARAMS_PREPROCESSING_WEIDEN, \
    PARAMS_PREPROCESSING_WEIDEN_NANS

datetimes_example_phases  = [
    datetime(2019, 5, 2, 16, 30),
    datetime(2018, 8, 8, 5, 15,)
]

limits_phases = [
    [1.25, 2.25, 9.25, 11.5],
    [1.25, 2.25, 6.5, 9.5],

]

#### SET THE FOLOWING VARIABLES
# Filepath to output directory
make_dir_if_missing(dir_plots)
####


params_preprocessing = {**PARAMS_PREPROCESSING_WEIDEN, 'thresholds_columns_clip': {}}
params_preprocessing_nans = {**PARAMS_PREPROCESSING_WEIDEN_NANS, 'thresholds_columns_clip': {}}

plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['pdf.fonttype'] = 42

extensions = ['png', 'eps', 'pdf']

color_measurement = 'tab:blue'

color_gamlss = 'tab:blue'
color_lstm = 'tab:orange'
color_sarx = 'tab:green'


cls_model_gamlss = ModelGamlssJsu.__name__
cls_model_sarx = ModelPointBase.__name__
cls_model_lstm = ModelRnnSeparateTimestepsJsu.__name__

path_predictions_best_fit_test = os.path.join(dir_samples, cls_model_gamlss, name_model_gamlss_best_fit_oracles + '.gz')
path_predictions_best_fit_rfore_test = os.path.join(dir_samples, cls_model_gamlss, name_model_gamlss_best_fit_rfore + '.gz')
path_predictions_no_thresholds_test = os.path.join(dir_samples, cls_model_gamlss, name_model_gamlss_no_thresholds_oracles + '.gz')
path_predictions_3_thresholds_test = os.path.join(dir_samples, cls_model_gamlss, name_model_gamlss_3_thresholds_oracles + '.gz')
path_predictions_only_wtp_test = os.path.join(dir_samples, cls_model_gamlss, name_model_gamlss_only_wtp_oracles + '.gz')
path_predictions_lstm_test = os.path.join(dir_samples, cls_model_gamlss, name_model_gamlss_lstm_oracles + '.gz')
path_predictions_sarx_test = os.path.join(dir_samples, cls_model_gamlss, name_model_gamlss_sarx_oracles + '.gz')

path_truth_best_fit_test = os.path.join(dir_truth, cls_model_gamlss, name_model_gamlss_best_fit_oracles + '.gz')
path_truth_best_fit_rfore_test = os.path.join(dir_truth, cls_model_gamlss, name_model_gamlss_best_fit_rfore + '.gz')
path_truth_no_thresholds_test = os.path.join(dir_truth, cls_model_gamlss, name_model_gamlss_no_thresholds_oracles + '.gz')
path_truth_3_thresholds_test = os.path.join(dir_truth, cls_model_gamlss, name_model_gamlss_3_thresholds_oracles + '.gz')
path_truth_only_wtp_test = os.path.join(dir_truth, cls_model_gamlss, name_model_gamlss_only_wtp_oracles + '.gz')
path_truth_lstm_test = os.path.join(dir_truth, cls_model_gamlss, name_model_gamlss_lstm_oracles + '.gz')
path_truth_sarx_test = os.path.join(dir_truth, cls_model_gamlss, name_model_gamlss_sarx_oracles + '.gz')

path_datetimes_best_fit_test = os.path.join(dir_datetimes_x, cls_model_gamlss, name_model_gamlss_best_fit_oracles )
path_datetimes_best_fit_rfore_test = os.path.join(dir_datetimes_x, cls_model_gamlss, name_model_gamlss_best_fit_rfore )
path_datetimes_no_thresholds_test = os.path.join(dir_datetimes_x, cls_model_gamlss, name_model_gamlss_no_thresholds_oracles )
path_datetimes_3_thresholds_test = os.path.join(dir_datetimes_x, cls_model_gamlss, name_model_gamlss_3_thresholds_oracles )
path_datetimes_only_wtp_test = os.path.join(dir_datetimes_x, cls_model_gamlss, name_model_gamlss_only_wtp_oracles )
path_datetimes_lstm_test = os.path.join(dir_datetimes_x, cls_model_gamlss, name_model_gamlss_lstm_oracles )
path_datetimes_sarx_test = os.path.join(dir_datetimes_x, cls_model_gamlss, name_model_gamlss_sarx_oracles )

path_scores_rfore_timesteps_best_fit_test = os.path.join(dir_scores, cls_model_gamlss, NAME_DIR_SCORES_TIMESTEPS, name_model_gamlss_best_fit_rfore )
path_scores_rfore_datapoints_best_fit_test = os.path.join(dir_scores, cls_model_gamlss, NAME_DIR_SCORES_DATAPOINTS, name_model_gamlss_best_fit_rfore )
path_scores_rfore_timesteps_lstm_test = os.path.join(dir_scores, cls_model_gamlss, NAME_DIR_SCORES_TIMESTEPS, name_model_gamlss_lstm_rfore )
path_scores_rfore_datapoints_lstm_test = os.path.join(dir_scores, cls_model_gamlss, NAME_DIR_SCORES_DATAPOINTS, name_model_gamlss_lstm_rfore )
path_scores_rfore_timesteps_sarx_test = os.path.join(dir_scores, cls_model_gamlss, NAME_DIR_SCORES_TIMESTEPS, name_model_gamlss_sarx_rfore )
path_scores_rfore_datapoints_sarx_test = os.path.join(dir_scores, cls_model_gamlss, NAME_DIR_SCORES_DATAPOINTS, name_model_gamlss_sarx_rfore )

path_scores_roracle_timesteps_best_fit_test = os.path.join(dir_scores, cls_model_gamlss, NAME_DIR_SCORES_TIMESTEPS, name_model_gamlss_best_fit_oracles )
path_scores_roracle_datapoints_best_fit_test = os.path.join(dir_scores, cls_model_gamlss, NAME_DIR_SCORES_DATAPOINTS, name_model_gamlss_best_fit_oracles )
path_scores_roracle_timesteps_lstm_test = os.path.join(dir_scores, cls_model_gamlss, NAME_DIR_SCORES_TIMESTEPS, name_model_gamlss_lstm_oracles )
path_scores_roracle_datapoints_lstm_test = os.path.join(dir_scores, cls_model_gamlss, NAME_DIR_SCORES_DATAPOINTS, name_model_gamlss_lstm_oracles )
path_scores_roracle_timesteps_sarx_test = os.path.join(dir_scores, cls_model_gamlss, NAME_DIR_SCORES_TIMESTEPS, name_model_gamlss_sarx_oracles )
path_scores_roracle_datapoints_sarx_test = os.path.join(dir_scores, cls_model_gamlss, NAME_DIR_SCORES_DATAPOINTS, name_model_gamlss_sarx_oracles )


def load_pickle(filepath):
    print(f'Loading {filepath}')
    path_pickle = f'{os.path.basename(filepath)}.pickle'
    if not os.path.exists(path_pickle):
        result = np.asarray(load_json_gzip(filepath))
        pickle.dump(result, open(path_pickle, 'wb'))
    else:
        result = pickle.load(open(path_pickle, 'rb'))
    return result

predictions_best_fit_test = load_pickle(path_predictions_best_fit_test)
predictions_best_fit_rfore_test = load_pickle(path_predictions_best_fit_rfore_test)
predictions_no_thresholds_test = load_pickle(path_predictions_no_thresholds_test)
predictions_3_thresholds_test = load_pickle(path_predictions_3_thresholds_test)
predictions_only_wtp_test = load_pickle(path_predictions_only_wtp_test)
predictions_lstm_test = load_pickle(path_predictions_lstm_test)
predictions_sarx_test = load_pickle(path_predictions_sarx_test)
# predictions_no_thresholds_test = np.asarray(load_json_gzip(path_predictions_no_thresholds_test))
# predictions_3_thresholds_test = np.asarray(load_json_gzip(path_predictions_3_thresholds_test))
truth_best_fit_test = np.asarray(load_json_gzip(path_truth_best_fit_test))
truth_best_fit_rfore_test = np.asarray(load_json_gzip(path_truth_best_fit_rfore_test))
truth_no_thresholds_test = np.asarray(load_json_gzip(path_truth_no_thresholds_test))
truth_3_thresholds_test = np.asarray(load_json_gzip(path_truth_3_thresholds_test))
truth_only_wtp_test = np.asarray(load_json_gzip(path_truth_only_wtp_test))
truth_lstm_test = np.asarray(load_json_gzip(path_truth_lstm_test))
truth_sarx_test = np.asarray(load_json_gzip(path_truth_sarx_test))

delta_step = timedelta(minutes=15)

make_dtarray = lambda x: np.asarray(x, dtype='datetime64[ns]')
with open(path_datetimes_best_fit_test) as f:
    j = json.load(f)
    dates_best_fit_test = json_to_params(j, make_dtarray)
with open(path_datetimes_best_fit_rfore_test) as f:
    j = json.load(f)
    dates_best_fit_rfore_test = json_to_params(j, make_dtarray)


with open(path_datetimes_no_thresholds_test) as f:
    j = json.load(f)
    dates_no_thresholds_test = json_to_params(j, make_dtarray)
with open(path_datetimes_3_thresholds_test) as f:
    j = json.load(f)
    dates_3_thresholds_test = json_to_params(j, make_dtarray)
with open(path_datetimes_only_wtp_test) as f:
    j = json.load(f)
    dates_only_wtp_test = json_to_params(j, make_dtarray)
with open(path_datetimes_lstm_test) as f:
    j = json.load(f)
    dates_lstm_test = json_to_params(j, make_dtarray)
with open(path_datetimes_sarx_test) as f:
    j = json.load(f)
    dates_sarx_test = json_to_params(j, make_dtarray)


scores_rfore_datapoints_best_fit_test = load_json_gzip(path_scores_rfore_datapoints_best_fit_test)
with open(path_scores_rfore_timesteps_best_fit_test) as f:
    j = json.load(f)
    scores_rfore_timesteps_best_fit_test = json_to_params(j, np.asarray)
scores_rfore_datapoints_lstm_test = load_json_gzip(path_scores_rfore_datapoints_lstm_test)
with open(path_scores_rfore_timesteps_lstm_test) as f:
    j = json.load(f)
    scores_rfore_timesteps_lstm_test = json_to_params(j, np.asarray)
scores_rfore_datapoints_sarx_test = load_json_gzip(path_scores_rfore_datapoints_sarx_test)
with open(path_scores_rfore_timesteps_sarx_test) as f:
    j = json.load(f)
    scores_rfore_timesteps_sarx_test = json_to_params(j, np.asarray)
    
scores_roracle_datapoints_best_fit_test = load_json_gzip(path_scores_roracle_datapoints_best_fit_test)
with open(path_scores_roracle_timesteps_best_fit_test) as f:
    j = json.load(f)
    scores_roracle_timesteps_best_fit_test = json_to_params(j, np.asarray)
scores_roracle_datapoints_lstm_test = load_json_gzip(path_scores_roracle_datapoints_lstm_test)
with open(path_scores_roracle_timesteps_lstm_test) as f:
    j = json.load(f)
    scores_roracle_timesteps_lstm_test = json_to_params(j, np.asarray)
scores_roracle_datapoints_sarx_test = load_json_gzip(path_scores_roracle_datapoints_sarx_test)
with open(path_scores_roracle_timesteps_sarx_test) as f:
    j = json.load(f)
    scores_roracle_timesteps_sarx_test = json_to_params(j, np.asarray)


def save_fig(fig, filename):
    for extension in extensions:
        fig.savefig(os.path.join(dir_plots, f'{filename}.{extension}'), dpi=1200)

steps_hour = 4
n_points_day = steps_hour * 24
n_points_week = n_points_day*7

x_hours_day = np.linspace(0, 24, n_points_day)

str_format_label_unit = "{name} [{unit}]"

format_label_unit = lambda name_, unit_: str_format_label_unit.format(name=name_, unit=unit_)

unit_venturi = 'l/s'

name_inflow = 'WTP inflow'
name_level = 'water level'
name_rain = 'rain rate'
name_time_day = 'time of day'
unit_time_day = 'h'
label_rain = format_label_unit(name_rain, 'mm/min')
label_water_level = format_label_unit('level above NHN', 'm')

label_venturi = format_label_unit(name_inflow, unit_venturi)
label_residuals = format_label_unit('forecast error', unit_venturi)

date_start = date_start_train
date_end = date_end_test

variables_use = variables_external + [name_column_inflow]

data, _, _ = get_data(
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        False,
        n_steps_predict,
        variables_use,
        [name_column_rain_history],
        True,
        params_preprocessing,
        path_csv_data_wtp_network_rain,
        path_csv_forecast_rain_radar,
        steps_cumulate_rain=6,
        keep_raw_rain=True,
    )
data_nans, _, _ = get_data(
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        False,
        n_steps_predict,
        variables_use,
        [name_column_rain_history],
        True,
        params_preprocessing_nans,
        path_csv_data_wtp_network_rain,
        path_csv_forecast_rain_radar,
        steps_cumulate_rain=6,
        keep_raw_rain=True,
    )

########## Plot params
textwidth_mm = 140
cm = 1/2.54  # centimeters in inches
mm = cm / 10.
size_point = 0.353  #mm
px_per_mm = 10

mm_to_px = lambda mm: mm*px_per_mm
points_to_px = lambda points: points * size_point * px_per_mm

## sizes in mm
width_mm_quarter = textwidth_mm/2
height_mm_quarter = width_mm_quarter * 3/4

height_px_quarter = mm_to_px(height_mm_quarter)
width_px_quarter = mm_to_px(width_mm_quarter)

width_in_quarter = width_mm_quarter*mm
height_in_quarter = height_mm_quarter*mm

figsize_plt_quarter = (width_mm_quarter*mm, height_mm_quarter*mm)
figsize_plt_quarter_two_columns = (width_mm_quarter*2*mm, height_mm_quarter*mm)
figsize_plt_quarter_two_columns_two_rows = (width_mm_quarter*2*mm, height_mm_quarter*2*mm)

## text
font_size = 8  # points
font_size_subscript = 6 # points

font = {'family' : 'Liberation Serif',
        # 'weight' : 'n',
        'size': font_size}
matplotlib.rc('font', **font)
matplotlib.rc('legend', fontsize=font_size_subscript)

def make_label_x(axis, text):
    axis.set_xlabel(text, fontsize=font_size)
def make_label_y(axis, text):
    axis.set_ylabel(text, fontsize=font_size)
##########


#### descriptives ####

### Seasonals ###

def make_data_seasonal(datasets):
    list_reshaped = []
    for data_ in datasets:
        index_weekday = data_.index.weekday  # 0 is mondayy

        # aggregate daily data
        datetime_start = data_.index[(index_weekday == 0) & (data_.index.hour == 0) & (data_.index.minute == 0)][0]
        data_start_aligned = data_[name_column_inflow][datetime_start:]
        n_weeks = int(math.floor(len(data_start_aligned) / n_points_week))
        n_datapoints = n_weeks * n_points_week
        data_aligned = data_start_aligned.iloc[:n_datapoints]

        # [week, day, intraday]
        list_reshaped.append(np.reshape(data_aligned.to_numpy(), (n_weeks, 7, n_points_day)))

    data_reshaped = np.concatenate(list_reshaped, axis=0)

    medians_saturday = np.median(data_reshaped[:, 5], axis=0)
    medians_sunday = np.median(data_reshaped[:, 6], axis=0)
    medians_weekday = np.median(data_reshaped[:, :5], axis=[0,1])

    return medians_weekday, medians_saturday, medians_sunday

is_winter = ((data.index.month==12) |
    (data.index.month==1) |
    (data.index.month==2))

is_summer = ((data.index.month==6) |
    (data.index.month==7) |
    (data.index.month==8))

data_winter_2017 = data[
    is_winter & (data.index.year==2017)
]
data_winter_2018 = data[
    is_winter & (data.index.year==2018)
]
data_winter_2019 = data[
    is_winter & (data.index.year==2019)
]

data_summer_2017 = data[
    is_summer & (data.index.year==2017)
]
data_summer_2018 = data[
    is_summer & (data.index.year==2018)
]
data_summer_2019 = data[
    is_summer & (data.index.year==2019)
]

medians_weekday, medians_saturday, medians_sunday = make_data_seasonal([data])
medians_weekday_winter, medians_saturday_winter, medians_sunday_winter = make_data_seasonal([data_winter_2017, data_winter_2018, data_winter_2019])
medians_weekday_summer, medians_saturday_summer, medians_sunday_summer = make_data_seasonal([data_summer_2017, data_summer_2018, data_summer_2019])

fig, axes = plt.subplots(ncols=2, figsize=figsize_plt_quarter_two_columns)

axes[0].plot(x_hours_day, medians_weekday, color=color_measurement, label='working day')
axes[0].plot(x_hours_day, medians_saturday, color='tab:green', label='saturday')
axes[0].plot(x_hours_day, medians_sunday, color='tab:orange', label='sunday')
axes[0].set(xlim=[0, 24])

axes[1].plot(x_hours_day, medians_weekday_summer, color='tab:green', label='summer')
axes[1].plot(x_hours_day, medians_weekday_winter, color='tab:red', label='winter')
axes[1].set(xlim=[0, 24])

axes[0].set_title('a)')
axes[1].set_title('b)')

axes[0].legend(loc='upper left', frameon=True, framealpha=1)
axes[1].legend(loc='upper left', frameon=True, framealpha=1)


label_x = format_label_unit('time of day', 'h')
make_label_y(axes[0], label_venturi)
make_label_x(axes[0], label_x)
make_label_y(axes[1], label_venturi)
make_label_x(axes[1], label_x)
axes[1].set_xlabel(format_label_unit('time of day', 'h'))

fig.tight_layout()
save_fig(fig, 'seasonalities')

##### scatters
print('Scatters...')
fig, axes = plt.subplots(ncols=2, figsize=figsize_plt_quarter_two_columns)
# axes[0].scatter(data_nans[col_ext], data_nans[col_venturi], s=0.01)

def get_is_dry(data_venturi):
    return data_venturi < 250

def get_is_rise(data_venturi):
    return (data_venturi.shift(-5) > 400) & (data_venturi.shift(5) < 400)

is_dry = get_is_dry(data_nans[name_column_inflow])

# reduce number of datapoints to plot to avoid bloated vector graphic
frac_use_dry = 0.1
use_dry = np.logical_and(
    is_dry,
    np.random.choice(a=[False, True], size=is_dry.shape, p=[1-frac_use_dry, frac_use_dry])
)

color_15 = 'tab:blue'
color_75 = 'tab:orange'
axes[0].scatter(data_nans[column_level_plot].shift(1)[use_dry], data_nans[name_column_inflow][use_dry], s=0.05, color=color_15)
axes[0].scatter(data_nans[column_level_plot].shift(5)[use_dry], data_nans[name_column_inflow][use_dry], s=0.05, color=color_75)

# cut off outliers
axes[0].set(xlim=[54.1, 54.3], ylim=[25, 250])

data_rain_cumulated = data_nans[name_column_rain_history].rolling(6).sum().shift(2)
data_venturi = data_nans[name_column_inflow]
is_rise = get_is_rise(data_venturi)

# has_rain = (data_rain_cumulated > 0.001) & (data_rain_cumulated < 0.075)
axes[1].scatter(data_nans[column_level_plot].shift(1)[is_rise], data_nans[name_column_inflow][is_rise], s=0.05, color=color_15)
axes[1].scatter(data_nans[column_level_plot].shift(5)[is_rise], data_nans[name_column_inflow][is_rise], s=0.05, color=color_75)


make_label_x(axes[0], label_water_level)
make_label_y(axes[0], label_venturi)
axes[0].set_title('a)')

make_label_x(axes[1], label_water_level)
make_label_y(axes[1], label_venturi)
axes[1].set_title('b)')

path_15 = mpatches.Patch(color=color_15, label='15 min')
path_75 = mpatches.Patch(color=color_75, label='75 min')

axes[0].legend(loc='lower right', handles=[path_15, path_75], frameon=True, framealpha=1)
axes[1].legend(loc='lower right',  handles=[path_15, path_75], frameon=True, framealpha=1)

fig.tight_layout()
save_fig(fig, 'scatter_venturi_vs_level_dry_rise')


#### Histogram wtp
print('Hist WTP')
fig, axis = plt.subplots(figsize=figsize_plt_quarter_two_columns)
axis.hist(data[name_column_inflow], bins=50)
make_label_y(axis, label_venturi)
make_label_x(axis, 'count')
fig.tight_layout()
save_fig(fig, 'histogram_wtp')

#### Histogram rain
print('Hist rain')
fig, axis = plt.subplots(figsize=figsize_plt_quarter_two_columns)
axis.hist(data[name_column_rain_history], bins=20)
make_label_y(axis, label_rain)
make_label_x(axis, 'count')
fig.tight_layout()
axis.semilogy()
save_fig(fig, 'histogram_rain')

#### Example rain event
for datetime_example_phases, limits_phases_ in zip(datetimes_example_phases, limits_phases):
    print('rain event phases')
    n_hours = 15
    timedelta_duration_phases = timedelta(hours=n_hours)
    x = np.linspace(0, n_hours, n_hours*steps_hour+1, endpoint=True)

    data_phases = data[datetime_example_phases:datetime_example_phases+timedelta_duration_phases]
    data_phases_rain = data_phases[name_column_rain_history].to_numpy()
    data_phases_wtp = data_phases[name_column_inflow].to_numpy()
    data_phases_ext = data_phases[column_level_plot].to_numpy()
    data_phases_sku = data_phases[column_sku_plot].to_numpy()

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(width_in_quarter*2, height_in_quarter*1.2))

    axes[0,0].plot(x, data_phases_wtp, color=color_measurement)
    axes[0,0].set(xlim=[0, x[-1]])

    axes[0,1].plot(x, data_phases_rain, color=color_measurement)
    axes[0,1].set(xlim=[0, x[-1]])
    axes[1,0].plot(x, data_phases_ext, color=color_measurement)
    axes[1,0].set(xlim=[0, x[-1]])
    axes[1,1].plot(x, data_phases_sku, color=color_measurement)
    axes[1,1].set(xlim=[0, x[-1]])

    axes[0,0].axvspan(limits_phases_[0], limits_phases_[1], alpha=0.2, color='tab:red')
    axes[0,0].axvspan(limits_phases_[1], limits_phases_[2], alpha=0.2, color='tab:green')
    axes[0,0].axvspan(limits_phases_[2], limits_phases_[3], alpha=0.2, color='tab:orange')

    label_x = format_label_unit('time', 'h')
    make_label_x(axes[0,0], label_x)
    make_label_x(axes[0,1], label_x)
    make_label_x(axes[1,0], label_x)
    make_label_x(axes[1,1], label_x)

    make_label_y(axes[0,0], label_venturi)
    make_label_y(axes[0,1], label_rain)
    make_label_y(axes[1,0], label_water_level)
    make_label_y(axes[1,1], label_water_level)

    axes[0,0].set_title('a)')
    axes[0,1].set_title('b)')
    axes[1,0].set_title('c)')
    axes[1,1].set_title('d)')

    fig.tight_layout()
    save_fig(fig, f'example_phases_{datetime_example_phases.isoformat()}')

    #### Example rain event, one plot
    print('rain event phases unified')
    n_hours = 15
    timedelta_duration_phases = timedelta(hours=n_hours)
    x = np.linspace(0, n_hours, n_hours*steps_hour+1, endpoint=True)

    data_phases = data[datetime_example_phases:datetime_example_phases+timedelta_duration_phases]
    data_phases_rain = data_phases[name_column_rain_history].to_numpy()
    data_phases_wtp = data_phases[name_column_inflow].to_numpy()
    data_phases_ext = data_phases[column_level_plot].to_numpy()
    data_phases_sku = data_phases[column_sku_plot].to_numpy()

    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(width_in_quarter*2, height_in_quarter*1.2))

    def scale_minmax(data_):
        return (data_ - data_.min()) / (data_.max() - data_.min())

    axis.plot(x, scale_minmax(data_phases_wtp), color=color_measurement, label=name_inflow)
    axis.set(xlim=[0, x[-1]])
    axis.plot(x, scale_minmax(data_phases_rain), color='tab:green', label=name_rain)
    axis.set(xlim=[0, x[-1]])
    axis.plot(x, scale_minmax(data_phases_ext), color='tab:orange', label='level: network')
    axis.set(xlim=[0, x[-1]])
    axis.plot(x, scale_minmax(data_phases_sku), color='tab:purple', label='level: storage sewer')
    axis.set(xlim=[0, x[-1]])

    axis.axvspan(limits_phases_[0], limits_phases_[1], alpha=0.1, color='tab:red')
    axis.axvspan(limits_phases_[1], limits_phases_[2], alpha=0.1, color='tab:green')
    axis.axvspan(limits_phases_[2], limits_phases_[3], alpha=0.1, color='tab:orange')

    axis.legend(loc=('upper right'), frameon=True, framealpha=1)

    label_x = format_label_unit('time', 'h')
    make_label_x(axis, label_x)

    make_label_y(axis, 'normalized value')

    fig.tight_layout()
    save_fig(fig, f'example_phases_unified_{datetime_example_phases.isoformat()}')


#### Illustration of linear splines
print('Scatters...')
fig, axes = plt.subplots(ncols=2, figsize=figsize_plt_quarter_two_columns)
# axes[0].scatter(data_nans[col_ext], data_nans[col_venturi], s=0.01)

color_points = 'tab:blue'
color_line = 'tab:orange'

data_rain_cumulated = data_nans[name_column_rain_history].rolling(6).sum().shift(2)
data_venturi = data_nans[name_column_inflow]
is_rise = get_is_rise(data_venturi)

# has_rain = (data_rain_cumulated > 0.001) & (data_rain_cumulated < 0.075)
x, y = data_nans[column_level_plot].shift(1)[is_rise].to_numpy(), data_nans[name_column_inflow][is_rise].to_numpy()
is_nan = np.logical_or(np.isnan(x), np.isnan(y))
not_nan = np.logical_not(is_nan)
x, y = x[not_nan], y[not_nan]
axes[0].scatter(x, y, s=0.05, color=color_points)
axes[1].scatter(x, y, s=0.05, color=color_points)

x_spaced = np.linspace(np.min(x), np.max(x), 100)
regression_linear = LinearRegression().fit(x[..., np.newaxis], y)
prediction_linear = regression_linear.predict(x_spaced[..., np.newaxis])

n_thresholds = 5
thresholds = np.append(np.linspace(np.min(x), np.max(x), n_thresholds, endpoint=False)[1:], [-np.inf])
x_thresholded = np.stack([
    np.maximum(x, threshold_) for threshold_ in thresholds
], axis=1)

x_spaced_thresholded = np.stack([
    np.maximum(x_spaced, threshold_) for threshold_ in thresholds
], axis=1)
regression_splines = LinearRegression().fit(x_thresholded, y)
prediction_splines = regression_splines.predict(x_spaced_thresholded)

axes[0].plot(x_spaced, prediction_linear, color=color_line)
axes[1].plot(x_spaced, prediction_splines, color=color_line)



# y_regime_breaks = regression_splines.predict(
#     np.stack([
#         np.maximum(thresholds[:-1], threshold_) for threshold_ in thresholds
#     ], axis=1)
# )
#
#
#
# [axes[1].axvline(x_, color='tab:grey') for x_ in thresholds[:-1]]


axes[0].set(ylim=[np.min(y), np.max(y)])
axes[1].set(ylim=[np.min(y), np.max(y)])

make_label_x(axes[0], label_water_level)
make_label_y(axes[0], label_venturi)
axes[0].set_title('a)')

make_label_x(axes[1], label_water_level)
make_label_y(axes[1], label_venturi)
axes[1].set_title('b)')

fig.tight_layout()
save_fig(fig, 'linear_splines')

#### Prediction: with vs. without thresholds
def plot_prediction(truth, samples, dates, date_sample, axis, fig, label_x, label_y, title=None, n_steps_before_prediction=10, pos_legend='upper left'):
    print(f'plot prediction {label_x} {label_y}')

    n_steps_prediction = samples.shape[-1]
    n_steps_all = n_steps_prediction + n_steps_before_prediction
    x = np.linspace(-n_steps_before_prediction/steps_hour, n_steps_prediction/steps_hour, n_steps_all+1, endpoint=True)[1:]

    idx_first_prediction = np.argmin(np.abs(dates- np.datetime64(date_sample)))
    truth = truth[idx_first_prediction-n_steps_before_prediction:idx_first_prediction+n_steps_prediction, 0]

    predictions_data = samples[:, idx_first_prediction]

    # quantiles = 0.25, 0.75
    n_quantiles = 9
    quantiles = np.linspace(0.1, 0.9, n_quantiles, endpoint=True)

    quantiles_data = np.quantile(predictions_data, quantiles, axis=0)

    x_prediction = x[-n_steps_prediction:]

    n_ranges = int(quantiles.shape[0]/2)
    color_map = matplotlib.cm.get_cmap('autumn')
    for idx_quantile in range(n_ranges):
        idx_upper = quantiles.shape[0] - idx_quantile - 1
        if idx_upper == idx_quantile:
            continue
        pos_color = idx_quantile / (n_ranges-1)
        axis.fill_between(
            x_prediction,
            quantiles_data[idx_quantile],
            quantiles_data[idx_upper],
            alpha=0.3,
            color=color_map(pos_color),
            label=f'forecast quantiles {round(quantiles[idx_quantile], 1)} - {round(quantiles[idx_upper], 1)}'
        )

    axis.plot(
        x,
        truth,
        color=color_measurement,
        label='truth'
    )

    axis.set(xlim=[x[0], x[-1]])
    axis.legend(loc=pos_legend, frameon=True, framealpha=1)


    make_label_x(axis, label_x)
    make_label_y(axis, label_y)

    if title is not None:
        axis.set_title(title)

label_x_prediction = format_label_unit('time after last observation', 'h')

date_sample_prediction = datetime(2019, 5, 2, 17, 0)
fig, axes = plt.subplots(ncols=2, figsize=figsize_plt_quarter_two_columns)
plot_prediction(
    truth_best_fit_test,
    predictions_best_fit_test,
    dates_best_fit_test,
    date_sample_prediction,
    axes[0],
    fig,
    label_x_prediction,
    label_venturi,
    'a)'
)
plot_prediction(
    truth_3_thresholds_test,
    predictions_3_thresholds_test,
    dates_3_thresholds_test,
    date_sample_prediction,
    axes[1],
    fig,
    label_x_prediction,
    label_venturi,
    'b)'
)
fig.tight_layout()
save_fig(fig, 'forecast_best_fit_vs_3_thresholds')

fig, axes = plt.subplots(ncols=2, figsize=figsize_plt_quarter_two_columns)
plot_prediction(
    truth_best_fit_test,
    predictions_best_fit_test,
    dates_best_fit_test,
    date_sample_prediction,
    axes[0],
    fig,
    label_x_prediction,
    label_venturi,
    'a)'
)
plot_prediction(
    truth_no_thresholds_test,
    predictions_no_thresholds_test,
    dates_no_thresholds_test,
    date_sample_prediction,
    axes[1],
    fig,
    label_x_prediction,
    label_venturi,
    'b)'
)
fig.tight_layout()
save_fig(fig, 'forecast_best_fit_vs_no_thresholds')

date_sample_prediction = datetime(2019, 7, 8, 15)
fig, axes = plt.subplots(ncols=2, figsize=figsize_plt_quarter_two_columns)
plot_prediction(
    truth_best_fit_test,
    predictions_best_fit_test,
    dates_best_fit_test,
    date_sample_prediction,
    axes[0],
    fig,
    label_x_prediction,
    label_venturi,
    'a)'
)
plot_prediction(
    truth_only_wtp_test,
    predictions_only_wtp_test,
    dates_only_wtp_test,
    date_sample_prediction,
    axes[1],
    fig,
    label_x_prediction,
    label_venturi,
    'b)'
)
fig.tight_layout()
save_fig(fig, 'forecast_dry_spike_best_fit_vs_no_ext')

date_sample_prediction = datetime(2019, 5, 7, 10, 45)
fig, axis = plt.subplots(figsize=figsize_plt_quarter_two_columns)
plot_prediction(
    truth_best_fit_test,
    predictions_best_fit_test,
    dates_best_fit_test,
    date_sample_prediction,
    axis,
    fig,
    label_x_prediction,
    label_venturi,
    n_steps_before_prediction=86,
    pos_legend='lower left'
)

fig.tight_layout()
save_fig(fig, 'forecast_dry')

# Oracle vs Forecasts

date_sample_prediction = datetime(2019, 10, 6, 9, 30)
fig, axes = plt.subplots(ncols=2, figsize=figsize_plt_quarter_two_columns)
plot_prediction(
    truth_best_fit_test,
    predictions_best_fit_test,
    dates_best_fit_test,
    date_sample_prediction,
    axes[0],
    fig,
    label_x_prediction,
    label_venturi,
    'a)'
)
plot_prediction(
    truth_best_fit_rfore_test,
    predictions_best_fit_rfore_test,
    dates_best_fit_rfore_test,
    date_sample_prediction,
    axes[1],
    fig,
    label_x_prediction,
    label_venturi,
    'b)'
)
fig.tight_layout()
save_fig(fig, 'forecast_oracle_vs_rfore')

date_sample_prediction = datetime(2019, 5, 21, 5, 45)
fig, axes = plt.subplots(ncols=2, figsize=figsize_plt_quarter_two_columns)
plot_prediction(
    truth_best_fit_test,
    predictions_best_fit_test,
    dates_best_fit_test,
    date_sample_prediction,
    axes[0],
    fig,
    label_x_prediction,
    label_venturi,
    'a)'
)
plot_prediction(
    truth_best_fit_rfore_test,
    predictions_best_fit_rfore_test,
    dates_best_fit_rfore_test,
    date_sample_prediction,
    axes[1],
    fig,
    label_x_prediction,
    label_venturi,
    'b)'
)
fig.tight_layout()
save_fig(fig, 'forecast_oracle_vs_rfore_2')


date_sample_prediction = datetime(2019, 1, 13, 7, 15)
fig, axes = plt.subplots(ncols=2, figsize=figsize_plt_quarter_two_columns)
plot_prediction(
    truth_best_fit_test,
    predictions_best_fit_test,
    dates_best_fit_test,
    date_sample_prediction,
    axes[0],
    fig,
    label_x_prediction,
    label_venturi,
    'a)'
)
plot_prediction(
    truth_best_fit_rfore_test,
    predictions_best_fit_rfore_test,
    dates_best_fit_rfore_test,
    date_sample_prediction,
    axes[1],
    fig,
    label_x_prediction,
    label_venturi,
    'b)'
)
fig.tight_layout()
save_fig(fig, 'forecast_oracle_vs_rfore_3')


##### RMSE Versus Timesteps
fig, axes = plt.subplots(ncols=2, figsize=figsize_plt_quarter_two_columns)

x = np.linspace(0.25, 2.5, 10)
axes[0].plot(x, scores_roracle_timesteps_best_fit_test['raise']['rmse_timesteps'], label='GAMLSS', color=color_gamlss)
axes[0].plot(x, scores_roracle_timesteps_lstm_test['raise']['rmse_timesteps'], label='LSTM', color=color_lstm)
axes[0].plot(x, scores_roracle_timesteps_sarx_test['raise']['rmse_timesteps'], label='SARX', color=color_sarx)

axes[1].plot(x, scores_rfore_timesteps_best_fit_test['raise']['rmse_timesteps'], label='GAMLSS', color=color_gamlss)
axes[1].plot(x, scores_rfore_timesteps_lstm_test['raise']['rmse_timesteps'], label='LSTM', color=color_lstm)
axes[1].plot(x, scores_rfore_timesteps_sarx_test['raise']['rmse_timesteps'], label='SARX', color=color_sarx)

axes[0].set(xlim=[x[0], x[-1]])
axes[1].set(xlim=[x[0], x[-1]])

axes[0].legend(loc='upper left', frameon=True, framealpha=1)
axes[1].legend(loc='upper left', frameon=True, framealpha=1)

make_label_y(axes[0], format_label_unit('RMSE', unit_venturi))
make_label_x(axes[0], format_label_unit('forecast time', 'h'))
axes[0].set_title('a)')

make_label_y(axes[1], format_label_unit('RMSE', unit_venturi))
make_label_x(axes[1], format_label_unit('forecast time', 'h'))
axes[1].set_title('b)')

fig.tight_layout()


save_fig(fig, 'rmse_vs_time_rforecast')

##### Residuals

## All
timestep = 4
fig, axes = plt.subplots(ncols=2, figsize=figsize_plt_quarter_two_columns)

frac_use = 0.1
use = np.random.choice(a=[False, True], size=is_dry.shape, p=[1-frac_use, frac_use])

residuals_no_threshold =  np.mean(predictions_no_thresholds_test, axis=0)[..., timestep] - truth_no_thresholds_test[..., timestep]
residuals_no_threshold_df = pd.DataFrame(residuals_no_threshold, columns=['residuals'], index=pd.DatetimeIndex(dates_no_thresholds_test, tz=data.index.tz))
df_no_threshold_w_level = residuals_no_threshold_df.join(data[column_level_plot], how='inner')
use = np.random.choice(a=[False, True], size=len(df_no_threshold_w_level), p=[1-frac_use, frac_use])

residuals_best_fit = np.mean(predictions_best_fit_test, axis=0)[..., timestep] - truth_best_fit_test[..., timestep]
residuals_best_fit_df = pd.DataFrame(residuals_best_fit, columns=['residuals'], index=pd.DatetimeIndex(dates_best_fit_test, tz=data.index.tz))
df_best_fit_w_level = residuals_best_fit_df.join(data[column_level_plot], how='inner')
use = np.random.choice(a=[False, True], size=len(df_best_fit_w_level), p=[1-frac_use, frac_use])

axes[0].scatter(df_best_fit_w_level[column_level_plot], df_best_fit_w_level['residuals'], s=0.1)
axes[1].scatter(df_no_threshold_w_level[column_level_plot], df_no_threshold_w_level['residuals'], s=0.1)



make_label_x(axes[0], label_water_level)
make_label_y(axes[0], label_residuals)
axes[0].set_title('a)')

make_label_x(axes[1], label_water_level)
make_label_y(axes[1], label_residuals)
axes[1].set_title('b)')

fig.tight_layout()
save_fig(fig, 'residuals_no_threshold_vs_best_fit')


## Rise
fig, axes = plt.subplots(ncols=2, figsize=figsize_plt_quarter_two_columns)
residuals_no_threshold =  np.mean(predictions_no_thresholds_test, axis=0)[..., timestep] - truth_no_thresholds_test[..., timestep]
residuals_no_threshold_df = pd.DataFrame(residuals_no_threshold, columns=['residuals'], index=pd.DatetimeIndex(dates_no_thresholds_test, tz=data.index.tz))
df_no_threshold_w_level = residuals_no_threshold_df.join(data[[column_level_plot, name_column_inflow]], how='inner')
df_no_threshold_w_level = df_no_threshold_w_level[get_is_rise(df_no_threshold_w_level[name_column_inflow])]


residuals_best_fit = np.mean(predictions_best_fit_test, axis=0)[..., timestep] - truth_best_fit_test[..., timestep]
residuals_best_fit_df = pd.DataFrame(residuals_best_fit, columns=['residuals'], index=pd.DatetimeIndex(dates_best_fit_test, tz=data.index.tz))
df_best_fit_w_level = residuals_best_fit_df.join(data[[column_level_plot, name_column_inflow]], how='inner')
df_best_fit_w_level = df_best_fit_w_level[get_is_rise(df_best_fit_w_level[name_column_inflow])]

axes[0].scatter(df_no_threshold_w_level[column_level_plot], df_no_threshold_w_level['residuals'], s=0.1)
axes[1].scatter(df_best_fit_w_level[column_level_plot], df_best_fit_w_level['residuals'], s=0.1)


make_label_x(axes[0], label_water_level)
make_label_y(axes[0], label_residuals)
axes[0].set_title('a)')

make_label_x(axes[1], label_water_level)
make_label_y(axes[1], label_residuals)
axes[1].set_title('b)')

fig.tight_layout()
save_fig(fig, 'residuals_no_threshold_vs_best_fit_rise')


# For presentation: Bsplines
fig, axis = plt.subplots(figsize=figsize_plt_quarter_two_columns)
n_knots = 4
degree = 2

max_state = 365
states = np.linspace(0, max_state, 100, endpoint=True)

def states_to_position(states, n_knots, degree, max_state):
    max_position = n_knots - degree
    return states*0.99999 / max_state * max_position*2
positions = states_to_position(states, n_knots, degree, max_state)

weights = bspline.knot_weights(
    positions,
    n_knots,
    degree,
    True,
    False,
)

for idx in range(n_knots):
    axis.plot(states, weights[..., idx], label=f'season {idx}')
axis.plot(states, np.sum(weights, axis=-1), label='sum')
make_label_x(axis, 'time')
make_label_y(axis, 'weight')
axis.legend()

fig.tight_layout()
save_fig(fig, 'bsplines')
