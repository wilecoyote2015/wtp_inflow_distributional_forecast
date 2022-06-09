import logging

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except:
    pass

import numpy as np
from time import time

import pandas as pd
from sklearn import linear_model
import sklearn
from tqdm import tqdm

import tensorflow as tf
from scipy import sparse
from tensorflow_probability import distributions as tfd

logging.basicConfig(level=logging.DEBUG)

from src.constants.misc import *

from src.inflow_forecast.model.model_base import ModelBase


class ModelPointBase(ModelBase):
    n_params_pdf = 1
    name_model_base = 'point_base'

    def __init__(self, *args, params=None, stds_timesteps_scaled=None, **kwargs):
        ModelBase.__init__(self, *args, **kwargs)
        self.stds_timesteps_scaled = [None for idx in range(
            self.n_steps_forecast_horizon)] if stds_timesteps_scaled is None else stds_timesteps_scaled

        self.setup_optimizers(params)

    def setup_optimizers(self, params):
        optimizers_timesteps = []
        for lambdas_timestep in self.lambdas_lasso:
            optimizers_params = []
            for lambda_param_pdf in lambdas_timestep:
                if lambda_param_pdf > 0:
                    optimizers_params.append(
                        linear_model.Lasso(
                            fit_intercept=False,
                            normalize=False,
                            alpha=lambda_param_pdf,
                            max_iter=100000,
                            # selection='random',
                            precompute=not self.make_design_matrix_sparse,
                            copy_X=True,
                            # fit_path=True,
                            # eps=1e-6
                            tol=1e-7,  # 1e-7
                            # jitter=0.01
                            selection='random',
                            # n_jobs=4
                        )
                    )
                elif lambda_param_pdf == 0:
                    optimizers_params.append(
                        linear_model.LinearRegression(fit_intercept=False, copy_X=False, n_jobs=4),
                    )
                else:
                    raise ValueError(f'Lambda {lambda_param_pdf} < 0.')
            optimizers_timesteps.append(optimizers_params)

            if params is not None:
                for optimizers_timestep, params_model_timestep in zip(optimizers_timesteps, params):
                    for optimizer_param_pdf, params_model_param_pdf in zip(optimizers_timestep, params_model_timestep):
                        optimizer_param_pdf.coef_ = params_model_param_pdf
                        # Need to set intercept because Linear model is not initializable properly without fitting.
                        optimizer_param_pdf.intercept_ = 0.

            self.optimizers_timesteps = optimizers_timesteps

    def get_log_likelihood(self, data: pd.DataFrame, bool_datapoints_valid=None):
        # predict the means
        expected_values, x, truth, datetimes_x = self.predict_get_truth(
            data,
            True,
            None,
            bool_datapoints_valid=bool_datapoints_valid
        )

        # for each timestep, predict log-prob ob data
        log_probs_timesteps = []
        for idx_timestep in range(self.n_steps_forecast_horizon):
            stds = self.stds_timesteps_scaled[idx_timestep] * self.std_scaling_y
            distribution = tfd.Normal(
                expected_values[0, :, idx_timestep],  # [0] because first axis is samples
                stds
            )
            log_probs_timesteps.append(distribution.log_prob(truth[..., idx_timestep]))

        result = np.sum(log_probs_timesteps)

        return result

    def get_params_fitted_timestep_param_pdf(self, idx_timestep, idx_param_pdf):
        return self.optimizers_timesteps[idx_timestep][idx_param_pdf].coef_

    def predict_design_matrix(self, design_matrix: np.ndarray, idx_timestep):
        with sklearn.utils.parallel_backend('loky', n_jobs=-1, inner_max_num_threads=None):
            result = self.optimizers_timesteps[idx_timestep][0].predict(design_matrix)

        return result.astype(self.DTYPE)

    @property
    def kwargs_store_model_subclass(self):
        return {'stds_timesteps_scaled': self.stds_timesteps_scaled}

    def check_model_fitted(self):
        return not self.optimizers_timesteps[0][0].coef_ is None

    def fit_after_setup(self, x, datetimes_x, y_fit, indices_y_fit_valid_data, warm_start, params_pdf_fit, *args,
                        **kwargs):
        predictions_previous_scaled = []
        y_fit_scaled = self.scale_y(y_fit)

        for idx_timestep_predict in range(self.n_steps_forecast_horizon):
            # make design matrix with previous predictions
            logging.info(f'timestep {idx_timestep_predict}: creating desing matrix')
            design_matrix = self.make_design_matrix_timestep_param_pdf(
                x,
                predictions_previous_scaled,
                datetimes_x,
                idx_timestep_predict,
                0,  # only one pdf param for the point forecast.
            )
            design_matrix = self.postprocess_design_matrix(
                design_matrix,
                x,
                predictions_previous_scaled,
                datetimes_x,
                idx_timestep_predict,
                0,  # only one pdf param for the point forecast.
            )
            # design_matrices.append(design_matrix)
            logging.info(f'Fitting timestep {idx_timestep_predict} using created design matrix.')

            y_valid_scaled = self.slice_tensor_valid(y_fit_scaled, indices_y_fit_valid_data)
            design_matrix_valid = self.slice_tensor_valid(design_matrix, indices_y_fit_valid_data)

            with sklearn.utils.parallel_backend('loky', n_jobs=-1, inner_max_num_threads=None):
                if self.make_design_matrix_sparse:
                    design_matrix_valid_sparse = sparse.coo_matrix(
                        (design_matrix_valid[design_matrix_valid != 0], np.nonzero(design_matrix_valid)))
                    design_matrix_valid_sparse = design_matrix_valid_sparse.tocsc()

                    logging.debug(
                        f'sparse matrix density: {design_matrix_valid_sparse.nnz / float(np.size(design_matrix_valid))}')

                    design_matrix_valid = design_matrix_valid_sparse

                t0 = time()
                self.optimizers_timesteps[idx_timestep_predict][0].fit(design_matrix_valid,
                                                                       y_valid_scaled[:, idx_timestep_predict])
                logging.debug(f'fitted in {time() - t0}')

                # predict for all data (also invalid) to have valid input for building design matrix
                #   for next timestep.
                prediction_timestep_scaled = self.predict_design_matrix(design_matrix, idx_timestep_predict)
                predictions_previous_scaled.append(prediction_timestep_scaled)

                # make std dev for probabilistic sampling, assuming i.i. distribution
                prediction_timestep_scaled_valid = self.slice_tensor_valid(prediction_timestep_scaled,
                                                                           indices_y_fit_valid_data)
                errors = y_valid_scaled[:, idx_timestep_predict] - prediction_timestep_scaled_valid
                std_timestep = np.std(errors)

                self.stds_timesteps_scaled[idx_timestep_predict] = std_timestep

            # delete the design matrix
            del design_matrix_valid
            del design_matrix
            tf.keras.backend.clear_session()

    def predict_x_scaled(self, x, datetimes_x, n_samples, *args, **kwargs):
        expectations_timesteps = []

        logging.info('Predicting')
        for idx_timestep in tqdm(range(self.n_steps_forecast_horizon)):
            design_matrix = self.make_design_matrix_timestep_param_pdf(
                x,
                expectations_timesteps,
                datetimes_x,
                idx_timestep,
                0,  # only one pdf param for the point forecast.
            )
            design_matrix = self.postprocess_design_matrix(
                design_matrix,
                x,
                expectations_timesteps,
                datetimes_x,
                idx_timestep,
                0,  # only one pdf param for the point forecast.
            )
            prediction_expectation = self.predict_design_matrix(design_matrix, idx_timestep)

            expectations_timesteps.append(prediction_expectation)

        result_timesteps = []
        for idx_timestep in range(self.n_steps_forecast_horizon):

            # if n_samples is None, the expected value is returned as only sample
            expectations_timestep = expectations_timesteps[idx_timestep]
            if n_samples is None:
                result_timesteps.append(expectations_timestep[np.newaxis, ...])
            else:
                # sample from normal with bootstrapped std
                samples = np.random.normal(
                    expectations_timestep,
                    self.stds_timesteps_scaled[idx_timestep],
                    [n_samples, expectations_timestep.shape[0]])
                result_timesteps.append(samples)

        # make predictions with [sample, datapoint, timestep]
        predictions_array = np.stack(result_timesteps, axis=-1)

        # result is [sample, datapoint, timeste]
        return predictions_array

    def postprocess_design_matrix(
            self,
            design_matrix,
            x: np.ndarray,
            predictions_timesteps_prev: [np.ndarray],
            datetimes_x: np.ndarray,
            idx_timestep: int,
            idx_param_pdf: int,
    ) -> np.ndarray:
        return design_matrix
