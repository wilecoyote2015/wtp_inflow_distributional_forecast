from src.inflow_forecast.model.benchmark.model_benchmark_base import ModelBenchmarkBase
import tensorflow as tf

from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import logging
import os
import numpy as np
import pandas as pd


class ModelRnnSeparateTimesteps(ModelBenchmarkBase):
    distribution = tfd.Normal
    n_params_pdf = 2
    idx_param_pdf_loc = 0
    params_pdf_constant = []  # TODO
    link_functions_default = [
        lambda x: x,
        lambda x: tf.math.softplus(x*5.)/5.,  # factor for softness control
    ]
    def __init__(self, n_units, *args, dropout=0., **kwargs):
        self.n_units = n_units
        ModelBenchmarkBase.__init__(self, *args, **kwargs)
        self.dropout = dropout
        self.models_timesteps_fitted = [None for idx in range(self.n_steps_forecast_horizon)]

    def make_model_timestep(self, idx_timestep, output_mean=False, output_pdf=False):
        # columns + 6 because of seasonal columns
        if output_pdf:
            fn_tensor = lambda s: s
        elif output_mean:
            fn_tensor = lambda s: s.mean()
        else:
            fn_tensor = lambda s: s.sample()

        # get positions of all params in concatenation of rnn and constant params
        positions_params_concat = []
        idx_current_variable, idx_current_constant = 0, self.n_params_pdf - len(self.params_pdf_constant)
        for idx_param in range(self.n_params_pdf):
            is_constant = idx_param in self.params_pdf_constant
            idx_in_concat = idx_current_constant if is_constant else idx_current_variable
            positions_params_concat.append((idx_param, idx_in_concat))
            if is_constant:
                idx_current_constant += 1
            else:
                idx_current_variable += 1

        input = tf.keras.layers.Input(shape=(self.n_datapoints_history_needed, len(self.columns_needed) + 6), dtype=self.DTYPE)
        rnn = tf.keras.layers.LSTM(self.n_units, return_sequences=True, dtype=self.DTYPE, dropout=self.dropout)(input)
        flatten = tf.keras.layers.Flatten(dtype=self.DTYPE)(rnn)
        # layer for constant parameters
        one_constant = tf.keras.layers.Lambda(lambda x: x[..., :1, 0] * 0. + 1., dtype=self.DTYPE)(input)
        dense_params_constant = tf.keras.layers.Dense(
            units=len(self.params_pdf_constant),
            use_bias=False,
            dtype=self.DTYPE
        )(one_constant)
        dense_params_pdf = tf.keras.layers.Dense(
            units=self.n_params_pdf - len(self.params_pdf_constant),
            dtype=self.DTYPE
        )(flatten)

        concat_all_params = tf.keras.layers.Concatenate(axis=-1)([dense_params_pdf, dense_params_constant])
        pdf = tfp.layers.DistributionLambda(
            lambda params: self.distribution(
                *[
                    self.link_functions_default[idx_param](
                        params[..., position_param]
                    )
                    for idx_param, position_param in positions_params_concat
                ],
            ),
            convert_to_tensor_fn=fn_tensor,
            dtype=self.DTYPE
        )(concat_all_params)

        model = tf.keras.Model(inputs=input, outputs=pdf)

        return model

    @property
    def kwargs_store_model_subclass(self):
        return{
            'n_units': self.n_units,
        }

    def fit_after_setup(self, x, datetimes_x, y_fit, indices_y_fit_valid_data, warm_start, params_pdf_fit, n_epochs, *args, path_save_model=None,
                        interval_save=50,
                        data_validate=None,
                        weight_mse=0.,
                        **kwargs):
        y_fit_scaled = self.scale_y(y_fit)

        if not isinstance(n_epochs, (list, tuple)):
            # n_iters must be same lenth as
            n_epochs_use = [n_epochs] * self.n_steps_forecast_horizon
        else:
            n_epochs_use = n_epochs

        predictions_previous = tf.unstack(
            y_fit_scaled,
            axis=1
        )

        if data_validate is not None:
            x_validate, y_validate, datetimes_validate = self.make_x_y(data_validate, True)
            y_validate_scaled = (y_validate - self.mean_scaling_y) / self.std_scaling_y
            predictions_validate = tf.unstack(
                self.slice_tensor_sample_predictions(y_validate_scaled),
                axis=1
            )

        for idx_timestep_predict in range(self.n_steps_forecast_horizon):
            # make design matrix with previous predictions
            logging.info(f'timestep {idx_timestep_predict}: creating design matrices')

            design_matrix_train = self.make_design_matrix_timestep(
                x,
                datetimes_x,
                predictions_previous[:idx_timestep_predict],
                idx_timestep_predict,
                True,
            )

            if data_validate is not None:
                design_matrix_validate = self.make_design_matrix_timestep(
                    x_validate,
                    datetimes_validate,
                    predictions_validate[:idx_timestep_predict],
                    idx_timestep_predict,
                    True,
                )
                y_validate_timestep = predictions_validate[idx_timestep_predict]
                data_validate_timestep = (design_matrix_validate, y_validate_timestep)
            else:
                data_validate_timestep = None

            design_matrix_valid = design_matrix_train[indices_y_fit_valid_data]
            y_valid = y_fit_scaled[indices_y_fit_valid_data, idx_timestep_predict]

            # design_matrices.append(design_matrix)
            logging.info(f'Fitting timestep {idx_timestep_predict} using created design matrix.')

            if warm_start and self.check_model_fitted():
                logging.info('Using pre-fitted model for warm start.')
                model_timestep = self.models_timesteps_fitted[idx_timestep_predict]
            else:
                model_timestep = self.make_model_timestep(idx_timestep_predict, output_mean=False)

            logging.warning('With new tf version, it maybe that slicing y is not needed')

            def loss(y, rv_y):
                log_prob = rv_y.log_prob(y)
                mean = rv_y.mean()
                mse = tf.reduce_mean((y - mean) ** 2)
                return -log_prob + mse*weight_mse

            model_timestep.compile(
                loss=loss,
                # loss='rmse',
                          optimizer=tf.optimizers.Adam(learning_rate=0.005, amsgrad=False),
                          metrics=[  # tf.metrics.MeanAbsoluteError()
                          ],
                          run_eagerly=True
                )

            history = model_timestep.fit(
                        x=design_matrix_valid,
                        y=y_valid,
                    epochs=n_epochs_use[idx_timestep_predict],
                            validation_data=data_validate_timestep
                )

            self.models_timesteps_fitted[idx_timestep_predict] = model_timestep


    def predict_x_scaled(self, x, datetimes_x, n_samples, *args, return_params=False, **kwargs):
        logging.info('Sampling: making model')
        # result is [sample, datapoint, timestep]

        design_matrices_base = [
            self.make_design_matrix_timestep_base(
                x,
                datetimes_x,
                idx_timestep,
                True
            ) for idx_timestep in range(self.n_steps_forecast_horizon)
        ]

        models = self.models_timesteps_fitted
        @tf.function(experimental_follow_type_hints=True, jit_compile=False)
        def sample_model(idx_timestep: int, design_matrix: tf.Tensor):
            return models[idx_timestep](design_matrix)

        logging.info(f'Making interaction and prediction pdf params')
        paths_sampled_list = []
        for idx_sample in range(n_samples):
            logging.info(f'sampling sample {idx_sample+1} of {n_samples}')
            samples_timesteps = []
            for idx_timestep, design_matrix_base_timestep in enumerate(design_matrices_base):
                logging.debug(f'making design matrix for timestep {idx_timestep+1} of {self.n_steps_forecast_horizon}')
                design_matrix = self.make_design_matrix_timestep(
                    x,
                    datetimes_x,
                    samples_timesteps,
                    idx_timestep,
                    True,
                    design_matrix_base=design_matrix_base_timestep
                )
                logging.debug(f'sampling from model')
                sample_timestep = sample_model(idx_timestep, design_matrix)
                samples_timesteps.append(sample_timestep)

            paths_sampled_list.append(tf.stack(samples_timesteps, axis=1))
        result = tf.stack(paths_sampled_list, axis=0)

        return result

    def save_model_additional_subclass(self, path_output, save_data):
        logging.error('SAVING RNN: For some reason, loaded model performs significantly worse than fitted model.')
        for idx_timestep, model_timestep in enumerate(self.models_timesteps_fitted):
            model_timestep.save_weights(self.get_path_store_model_weights(path_output, idx_timestep), overwrite=True)

    @staticmethod
    def get_path_store_model_weights(path_store_model, idx_timestep):
        path_wo_ext = os.path.splitext(path_store_model)[0]

        return f'{path_wo_ext}_MODEL_TF_{idx_timestep}'

    def setup_loading_file_subclass(self, filepath):
        for idx_timestep, model_timestep in enumerate(self.models_timesteps_fitted):
            path_model = self.get_path_store_model_weights(filepath, idx_timestep)
            model_timestep = self.make_model_timestep(idx_timestep)
            model_timestep.load_weights(path_model)
            self.models_timesteps_fitted[idx_timestep] = model_timestep

    def get_log_likelihood(self, data: pd.DataFrame, bool_datapoints_valid=None):

        x, y, datetimes_x = self.make_x_y(data, True)
        x_prediction, truth = self.slice_x_y_sample_predictions(x, y)
        truth_scaled = (truth-self.mean_scaling_y) / self.std_scaling_y

        predictions_previous = tf.unstack(
            truth_scaled,
            axis=1
        )

        if bool_datapoints_valid is not None:
            bool_x_valid, y_for_valid, datetimes_x_ = self.make_x_y(bool_datapoints_valid, True)
            indices_valid = self.get_indices_datapoints_history_valid(bool_x_valid, True)
            self.logger.info(f'Omitted {truth.shape[0] - indices_valid.shape[0]} predictions with invalid datapoints')
            truth_scaled = tf.gather(truth_scaled, indices_valid)
        else:
            indices_valid = self.get_indices_datapoints_history_valid(tf.ones_like(data, dtype=tf.bool), True)


        log_prob = 0.
        for idx_timestep_predict in range(self.n_steps_forecast_horizon):
            # make design matrix with previous predictions
            logging.info(f'timestep {idx_timestep_predict}: creating design matrices')

            design_matrix_train = self.make_design_matrix_timestep(
                x,
                datetimes_x,
                predictions_previous[:idx_timestep_predict],
                idx_timestep_predict,
                True,
            )


            design_matrix_valid = design_matrix_train[indices_valid]

            # design_matrices.append(design_matrix)
            logging.info(f'Fitting timestep {idx_timestep_predict} using created design matrix.')
            model_timestep = self.models_timesteps_fitted[idx_timestep_predict]

            logging.warning('With new tf version, it maybe that slicing y is not needed')

            rv_timestep = model_timestep(design_matrix_valid)
            log_probs_timestep = rv_timestep.log_prob(truth_scaled[..., idx_timestep_predict])

            log_prob += tf.reduce_sum(log_probs_timestep)

        return log_prob

    @property
    def n_params_fitted_active(self):
        n_params = 0
        for model_timestep in self.models_timesteps_fitted:
            n_params_timestep = np.sum([np.prod(v.get_shape()) for v in model_timestep.trainable_weights])
            n_params += n_params_timestep
        return n_params

class ModelRnnSeparateTimestepsJsu(ModelRnnSeparateTimesteps):
    distribution = tfd.JohnsonSU
    n_params_pdf = 4
    idx_param_pdf_loc = 2
    params_pdf_constant = [1]  # TODO
    link_functions_default = [
        # skew, curt, loc, scale
        lambda x: (tf.math.sigmoid(x) - 0.5)*3,
        lambda x: tf.math.softplus(x),
        lambda x: x,
        lambda x: tf.math.softplus(x),  # factor for softness control
    ]

