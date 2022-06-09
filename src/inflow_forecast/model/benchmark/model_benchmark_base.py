import logging

import numpy as np
from datetime import timedelta
import tensorflow as tf

from src.inflow_forecast.model.model_base import ModelBase

logging.basicConfig(level=20)

class ModelBenchmarkBase(ModelBase):
    """ Base class providing a framework for defining models. """
    # DTYPE = np.float64
    n_params_pdf = 1
    idx_param_pdf_loc = 0
    name_model_base = 'base_benchmark'

    def __init__(
            self,
            variable_target: str,
            n_steps_forecast_horizon: int,
            min_lag: int,
            # for each column: [timestep[param_pdf[threshold[lag]]]]
            #   convention: timesteps relative to time of prediciton
            *args,
            means_scaling_x: np.ndarray = None,
            stds_scaling_x: np.ndarray = None,
            mean_scaling_y: np.ndarray = None,
            std_scaling_y: np.ndarray = None,
            columns_needed=None,
            **kwargs
    ):
        # remove kwargs that may be passed when loading from file
        for key in [
            'lags_columns',
            'means_scaling_design_matrices_base',
            'stds_scaling_design_matrices_base',
            'means_scaling_design_matrices_prediction',
            'stds_scaling_design_matrices_prediction',
            'means_scaling_design_matrices_interaction',
            'stds_scaling_design_matrices_interaction',
            'n_seasons_annual',
            'seasons_days_weeks',
            'thresholds',
            'params'
        ]:
            if key in kwargs:
                kwargs.pop(key)

        ModelBase.__init__(
            self,
            variable_target=variable_target,
            n_steps_forecast_horizon=n_steps_forecast_horizon,
            lags_columns={variable_target: []},
            thresholds={variable_target: []},
            seasons_days_weeks=[],
            n_seasons_annual=1,
            lambdas_lasso=0,
            *args,
            **kwargs
        )

        self.min_lag = min_lag

        self.columns_needed_ = columns_needed

        self.means_scaling_x = means_scaling_x
        self.stds_scaling_x = stds_scaling_x
        self.mean_scaling_y = mean_scaling_y
        self.std_scaling_y = std_scaling_y

    @property
    def kwargs_store_model_base(self):
        return dict(
            variable_target=self.variable_target,
            n_steps_forecast_horizon=self.n_steps_forecast_horizon,
            min_lag=self.min_lag,
            seasons_days_weeks=self.seasons_days_weeks,
            n_seasons_annual=self.n_seasons_annual,
            means_scaling_x=self.means_scaling_x,
            stds_scaling_x=self.stds_scaling_x,
            mean_scaling_y=self.mean_scaling_y,
            std_scaling_y=self.std_scaling_y,
            columns_needed=self.columns_needed_
        )

    def set_up_other_variables(self, data, *args, **kwargs):
        self.columns_needed_ = sorted(list(data.columns))

    @property
    def columns_needed(self):
        return self.columns_needed_

    @property
    def n_datapoints_history_needed(self):
        return -self.min_lag

    @property
    def min_lag_wrt_first_step_prediction(self):
        return self.min_lag

    def set_params_scaling_x(self, x_in_sample, datetimes_x_in_sample, y_fit, indices_y_fit_valid):
        self.means_scaling_x = tf.reduce_mean(x_in_sample, axis=0)
        self.stds_scaling_x = tf.math.reduce_std(x_in_sample, axis=0)

    @classmethod
    def encode_seasons(cls, x, datetimes_x):
        datetimes_x_days = cls.floor_days_datetimes(datetimes_x)
        datetimes_x_weeks = cls.floor_datetimes(datetimes_x, 'W')
        datetimes_x_years = cls.floor_datetimes(datetimes_x, 'AS')  # TODO: correct?

        phase_day = (datetimes_x - datetimes_x_days).astype(int) / np.timedelta64(timedelta(days=1), 'ns').astype(int)
        phase_week = (datetimes_x - datetimes_x_weeks).astype(int) / np.timedelta64(timedelta(days=7), 'ns').astype(int)
        phase_year = (datetimes_x - datetimes_x_years).astype(int) / np.timedelta64(timedelta(days=365.25), 'ns').astype(int) # TODO: okay this way?

        logging.warning('Check if the phases are correct! in hypercube script, there were problems.')

        seasons = np.concatenate(
            [
                tf.math.sin(
                    phase_day * 2 * np.pi
                )[:, tf.newaxis],
                tf.math.cos(
                    phase_day * 2 * np.pi
                )[:, tf.newaxis],
                tf.math.sin(
                    phase_week * 2 * np.pi
                )[:, tf.newaxis],
                tf.math.cos(
                    phase_week * 2 * np.pi
                )[:, tf.newaxis],
                tf.math.sin(
                    phase_year * 2 * np.pi
                )[:, tf.newaxis],
                tf.math.cos(
                    phase_year * 2 * np.pi
                )[:, tf.newaxis],
            ],
            axis=1
        )

        return seasons

    def set_indices_columns_variables_design_matrices_base(self):
        pass

    def make_design_matrix_timestep_base(self, x, datetimes_x, idx_timestep, do_scaling):
        logging.warning('Check if the phases are correct! in hypercube script, there were problems.')

        seasons = self.encode_seasons(x, datetimes_x)
        x_w_seasons = np.concatenate(
            [
                (x - self.means_scaling_x) / self.stds_scaling_x if do_scaling else x,
                seasons
            ],
            axis=1
        )

        nans_x = tf.reduce_any(
            tf.math.is_nan(x_w_seasons)
        )
        if nans_x:
            raise ValueError('Nans in x')

        result = np.ndarray(
            (
                self.get_n_datapoints_prediction(x),
                self.n_datapoints_history_needed,
                x_w_seasons.shape[1]
            )
        )

        # shift back the target variable.
        #   external regressors are always with lags w.r.t. first prediction timestep.
        #   target variable is w.r.t. idx_timestep,
        #   so that there are min_lags datapoints of each regressor.
        x_w_seasons[:, self.idx_column_x_target] = np.roll(x_w_seasons[:, self.idx_column_x_target], -idx_timestep)

        # fill all the datasets in result
        for idx_datapoint_prediction in range(self.get_n_datapoints_prediction(x)):
            # traverse from reverse to ensure alignment regardless how many first datapoints of x
            #   are sliced away.
            idx_datapoint_reverse = -idx_datapoint_prediction-1
            idx_stop = x_w_seasons.shape[0] - idx_datapoint_prediction
            idx_start = idx_stop - self.n_datapoints_history_needed
            result[idx_datapoint_reverse] = x_w_seasons[
                idx_start
                :idx_stop  #+ 1
            ]

        return result

    def make_design_matrix_timestep(self, x, datetimes_x, predictions_scaled, idx_timestep, do_scaling, design_matrix_base=None):
        """
        Design matrix with [datapoint, lag, regressor], so that each element
        of axis 0 is a dataset.
        """
        if design_matrix_base is None:
            result = self.make_design_matrix_timestep_base(x, datetimes_x, idx_timestep, do_scaling)
        else:
            result = np.copy(design_matrix_base)

        # insert the predictions
        n_steps_predictions = len(predictions_scaled)
        first_lag_needs_prediction = max(self.min_lag, -n_steps_predictions)
        for lag in range(first_lag_needs_prediction, 0):
            # predictions are [timestep[datapoint]] and timestep is ascending. hence, last one is most recent
            prediction_lag = predictions_scaled[lag]
            result[..., lag, self.idx_column_x_target] = prediction_lag

        return result
