import logging
import tensorflow as tf

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from tensorflow_probability import distributions as tfd
import tempfile
import tensorboard as tb
from copy import deepcopy
import os

from src.inflow_forecast.model.model_base import ModelBase
from src.common_functions.misc import json_to_params
from src.constants.misc import *

class ModelGamlssBase(ModelBase):

    n_params_pdf = 4
    idx_param_pdf_loc = 2
    params_pdf_constant = []  # parameters fitted with scalar constant value

    link_functions_default = [
        # skewness, tailweight, loc, scale for JSU
        lambda x: (tf.math.sigmoid(x) - 0.5),
        lambda x: tf.math.sigmoid(x)*3 + 0.05,
        lambda x: x,
        lambda x: tf.math.softplus(x*5.)/5.,  # factor for softness control
    ]

    def __init__(
            self,
            # n_iters_fit: int,
            *args,
            # link functions for each pdf parameter.
            # Must be as many as n_params_pdf
            # for each param pdf, the function takes one argument and maps it into appropriate space.
            params=None,
            class_optimizer=tf.optimizers.Adam,
            # Remark: Learning rate must be quite low. Order of magnitude: 1e-4
            kwargs_optimizer: dict=None,
            n_iters_per_param=1,
            use_mean_interaction=True,
            **kwargs
    ):
        ModelBase.__init__(self, *args, **kwargs)

        self.use_mean_interaction = use_mean_interaction

        if kwargs_optimizer is None:
            kwargs_optimizer = {
            'learning_rate': tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.0002, decay_steps=1000,
                                                           decay_rate=1, staircase=False),
            'amsgrad': True
        }


        self.n_iters_per_param = n_iters_per_param
        if not isinstance(class_optimizer, (tuple, list)):
            class_optimizer = [class_optimizer] * self.n_params_pdf
        elif len(class_optimizer) != self.n_params_pdf:
            raise ValueError(f'class_optimizer must be same length as self.n_params_pdf: {len(class_optimizer)} != {self.n_params_pdf}')

        if not isinstance(kwargs_optimizer, (tuple, list)):
            kwargs_optimizer = [kwargs_optimizer] * self.n_params_pdf
        elif len(kwargs_optimizer) != self.n_params_pdf:
            raise ValueError(
                f'kwargs_optimizer must be same length as self.n_params_pdf: {len(kwargs_optimizer)} != {self.n_params_pdf}')

        # store class and start args of optimizer so that it can be reset.
        self.class_optimizer = class_optimizer
        self.kwargs_optimizer = kwargs_optimizer if kwargs_optimizer is not None else {
            'learning_rate': tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.0002, decay_steps=1000,
                                                           decay_rate=1, staircase=False),
            'amsgrad': True
        }

        self.init_optimizer()

        self.link_functions = self.link_functions_default

        if len(self.link_functions) != self.n_params_pdf:
            raise ValueError(f'Number of link functions ({len(self.link_functions)}) != n_params_pdf ({self.n_params_pdf}).')

        if params is not None:
            params_tf = []
            # ensure that the params are variables.
            for idx_timestep in range(self.n_steps_forecast_horizon):
                params_tf_timestep = []
                for idx_param_pdf in range(self.n_params_pdf):
                    params_tf_timestep.append(
                        tf.Variable(params[idx_timestep][idx_param_pdf], dtype=self.DTYPE)
                    )
                params_tf.append(params_tf_timestep)

            self.params_tf = params_tf

        else:
            self.params_tf = None

    def init_optimizer(self):
        self.optimizer = [
            class_optimizer(**kwargs)
            for class_optimizer, kwargs in zip(self.class_optimizer, self.kwargs_optimizer)
        ]

    def check_model_fitted(self):
        return not self.params_tf is None

    @property
    def params_pdf_ar(self):
        return [idx for idx in range(self.n_params_pdf) if idx not in self.params_pdf_constant]

    @classmethod
    def model_from_point_forecast_file(cls,filepath, *args, lags_other_params=None, thresholds_other_params=None, **kwargs):
        """Pass the class-specific args not saved in point model file as args and kwargs!"""

        with open(filepath) as f:
            model_json = json.load(f)

        if cls.idx_param_pdf_loc is None:
            raise ValueError('Cannot use point for cast as location is not parameterized by a pdf param.')

        model_deserialized = json_to_params(model_json, np.asarray)
        kwargs_processed = cls.process_kwargs_from_file(model_deserialized, *args, **kwargs)

        # remove params because initializer would need params for all params pdf
        kwargs_processed[KWARGS_MODEL].pop('params')

        # If other lags and thresholds are not given, use the ones from point model for all pdf params.
        lags, thresholds = cls.make_lags_thresholds_from_point_model(
            kwargs_processed[KWARGS_MODEL]['lags_columns'],
            kwargs_processed[KWARGS_MODEL]['thresholds'],
            lags_other_params,
            thresholds_other_params
        )
        kwargs_processed[KWARGS_MODEL]['lags_columns'] = lags
        kwargs_processed[KWARGS_MODEL]['thresholds'] = thresholds

        # if indices_steps_day_include are given, they are only given for one param pdf.
        #   translate to all params pdf.
        #   there are constant parameters, but design matrices are always created for all params, so
        #   it is necessary to do this here, too.
        n_params_pdf = cls.n_params_pdf
        indices_steps_day_include = kwargs_processed[KWARGS_MODEL]['indices_steps_day_include']
        for idx_timestep, indices_steps_day_timestep in enumerate(indices_steps_day_include):
            indices_steps_day_include[idx_timestep] = deepcopy(indices_steps_day_timestep * n_params_pdf)
        kwargs_processed[KWARGS_MODEL]['indices_steps_day_include'] = indices_steps_day_include

        # scaling params are also duplicated for all params pdf.
        #   there are constant parameters, but design matrices are always created for all params, so
        #   it is necessary to do this here, too.
        for idx_timestep in range(kwargs_processed[KWARGS_MODEL]['n_steps_forecast_horizon']):
            kwargs_processed[KWARGS_MODEL]['means_scaling_design_matrices_base'][idx_timestep] *= n_params_pdf
            kwargs_processed[KWARGS_MODEL]['stds_scaling_design_matrices_base'][idx_timestep] *= n_params_pdf
            kwargs_processed[KWARGS_MODEL]['means_scaling_design_matrices_prediction'][idx_timestep] *= n_params_pdf
            kwargs_processed[KWARGS_MODEL]['stds_scaling_design_matrices_prediction'][idx_timestep] *= n_params_pdf
            kwargs_processed[KWARGS_MODEL]['means_scaling_design_matrices_interaction'][idx_timestep] *= n_params_pdf
            kwargs_processed[KWARGS_MODEL]['stds_scaling_design_matrices_interaction'][idx_timestep] *= n_params_pdf

        model = cls(**kwargs_processed[KWARGS_MODEL])


        # Setup must be done to get new scaling params.
        model.set_params_from_point_forecast(filepath)
        model.setup_loading_file_subclass(filepath)

        return model

    @classmethod
    def get_n_params_pdf_not_constant(cls):
        return cls.n_params_pdf - len(cls.params_pdf_constant)

    @classmethod
    def make_lags_thresholds_from_point_model(
            cls,
            lags_point,
            thresholds_point,
            lags_other_params,
            thresholds_other_params
    ):
        if lags_other_params is None and thresholds_other_params is None:
            logging.debug('Loading from point model: using lags and thresholds from point for all params pdf')
            lags_result, thresholds_result = {}, {}
            for name_column, lags_point_column in lags_point.items():
                thresholds_point_column = thresholds_point[name_column]
                lags_result_column, thresholds_result_column = [], []
                for lags_timestep_point, thresholds_timestep_point in zip(lags_point_column, thresholds_point_column):
                    # duplicate lags and threshold for all pdf params
                    # REMARK: also needed for constant params, as design matrices are created for all.
                    lags_result_column.append(
                        deepcopy(lags_timestep_point * cls.n_params_pdf)
                    )
                    thresholds_result_column.append(
                        deepcopy(thresholds_timestep_point * cls.n_params_pdf)
                    )
                lags_result[name_column] = lags_result_column
                thresholds_result[name_column] = thresholds_result_column

        elif lags_other_params is None or thresholds_other_params is None:
            raise ValueError('Either none or both of lags and thresholds must be given.')
        else:
            raise NotImplementedError('Need to updae this.')
            # TODO: check that provided lags and thresholds are compatible with model.
            logging.debug('Loading from point model: inserting provided lags and thresholds for the additional params pdf')
            lags_result, thresholds_result = deepcopy(lags_other_params), deepcopy(thresholds_other_params)
            for name_column, lags_point_column in lags_point.items():
                thresholds_point_column = thresholds_point[name_column]
                for idx_timestep, lags_timestep_point, thresholds_timestep_point in enumerate(zip(lags_point_column, thresholds_point_column)):
                    # duplicate lags and threshold for all pdf params
                    # REMARK: also needed for constant params, as design matrices are created for all.
                    lags_result[name_column][idx_timestep].insert(deepcopy(lags_timestep_point), cls.idx_param_pdf_loc)
                    thresholds_result[name_column][idx_timestep].insert(deepcopy(thresholds_timestep_point), cls.idx_param_pdf_loc)

        return lags_result, thresholds_result

    def setup_set_params_from_point_forecast(self, path_model):
        # Do setup with data from point forecast.
        #   this is important because scaling params must correspond to the train data used for point forecast.
        data_train = self.load_data_df(self.get_path_store_model_data(path_model))
        self.do_setup(data_train)
        self.set_params_from_point_forecast(path_model)

    def set_params_from_point_forecast(self, path_model_point_forecast):
        with open(path_model_point_forecast) as f:
            model_json = json.load(f)

        if self.idx_param_pdf_loc is None:
            raise ValueError('Cannot use point forecast as location is not parameterized by a pdf param.')

        lags_point = json_to_params(model_json[KWARGS_MODEL]['lags_columns'], lambda x: tf.constant(x, dtype=tf.int32))
        thresholds_point = json_to_params(model_json[KWARGS_MODEL]['thresholds'], lambda x: tf.constant(x, dtype=self.DTYPE))

        for idx_timestep in range(self.n_steps_forecast_horizon):
            for name_column in self.lags_columns.keys():
                if name_column not in lags_point:
                    raise ValueError(f'column {name_column} not in point lags.')
                if name_column not in thresholds_point:
                    raise ValueError(f'column {name_column} not in point thresholds.')

                for idx_threshold in range(len(list(thresholds_point[name_column][idx_timestep][0]))):
                    if not np.all(
                        np.equal(
                            lags_point[name_column][idx_timestep][0][idx_threshold],
                            self.lags_columns[name_column][idx_timestep][self.idx_param_pdf_loc][idx_threshold]
                        )
                    ):
                        raise ValueError(f'lags differ for column {name_column} at idx timestep {idx_timestep} and threshold {idx_threshold}')

                if not np.all(thresholds_point[name_column][idx_timestep][0] == self.thresholds[name_column][idx_timestep][self.idx_param_pdf_loc]):
                    raise ValueError(f'thresholds differ for column {name_column} at idx timestep {idx_timestep}')

        # deserialize params as variables
        params_deserialized_point = json_to_params(model_json[KWARGS_MODEL]['params'], lambda x: tf.Variable(x, dtype=self.DTYPE))

        self.set_default_params_fitted()

        # insert new params
        for idx_timestep in range(self.n_steps_forecast_horizon):
            self.params_tf[idx_timestep][self.idx_param_pdf_loc] = params_deserialized_point[idx_timestep][0]

    def launch_tensorboard(self):
        """
        Runs tensorboard with the given log_dir and wait for user input to kill the app.
        :param log_dir:
        :param clear_on_exit: If True Clears the log_dir on exit and kills the tensorboard app
        :return:
        """
        tb_ = tb.program.TensorBoard()
        tb_.configure(argv=[None, '--logdir', self.path_tensorboard])
        self.logger.info("Launching Tensorboard ")
        url = tb_.launch()
        self.logger.info(f'tb url: {url}')

    def get_params_fitted_timestep_param_pdf(self, idx_timestep, idx_param_pdf):
        return self.params_tf[idx_timestep][idx_param_pdf]

    @staticmethod
    def fn_make_distribution(*args, **kwargs):
        return tfd.JohnsonSU(*args, **kwargs)

    def get_default_params_fitted(self):
        params_tf = []
        for idx_timestep in range(self.n_steps_forecast_horizon):
            params_tf_timestep = []
            for idx_param_pdf in range(self.n_params_pdf):
                params_tf_timestep.append(
                    tf.Variable(self.get_default_params_timestep_param_pdf(
                        idx_timestep,
                        idx_param_pdf
                    ), dtype=self.DTYPE,
                        name=f'params_timestep_{idx_timestep}_param_pdf_{idx_param_pdf}'
                    )
                )
            params_tf.append(params_tf_timestep)

        return params_tf

    def set_default_params_fitted(self):
        self.params_tf = self.get_default_params_fitted()

    @property
    def kwargs_store_model_subclass(self):
        return {}

    def get_default_params_timestep_param_pdf(
            self,
            idx_timestep,
            idx_param_pdf
    ):
        # number of columns in the design matrix: simply use the
        if idx_param_pdf not in self.params_pdf_constant:
            n_params = self.get_n_params_needed_timestep_param_pdf(
                idx_timestep,
                idx_param_pdf
            )
            result = tf.random.normal((n_params, ), dtype=self.DTYPE) / n_params
        else:
            # if hack with tf.cond in make_params_distribution_timestep() is used,
            #   make constant tensor with 1 axis instead of scalar.
            result = tf.constant(0., dtype=self.DTYPE)


        return result


    def fit_after_setup(self, x, datetimes_x, y_fit, indices_y_fit_valid_data, warm_start, params_pdf_fit, n_iters, *args, fit_consecutively=False, path_save_model=None,
                        interval_save=50,
                        # TODO: changed for testing
                        weight_penalty=10.,  # 100 is good for normal, 10 for jsu.
                        **kwargs):
        # predictions_previous = np.ndarray((self.y_in_sample_predictions.shape[0], 0), dtype=self.DTYPE)
        params_pdf_fit = params_pdf_fit if params_pdf_fit is not None else range(self.n_params_pdf)

        y_fit_scaled = self.scale_y(y_fit)

        if not isinstance(n_iters, (list, tuple)):
            # n_iters must be same lenth as
            n_iters_use = [n_iters] * self.n_steps_forecast_horizon
        else:
            n_iters_use = n_iters

        if self.path_tensorboard is not None:
            self.launch_tensorboard()

        if not warm_start or self.params_tf is None:
            self.set_default_params_fitted()
        else:
            self.logger.info('Fitting with warm start.')

        predictions_previous = tf.unstack(
            y_fit_scaled,
            axis=1)

        for idx_timestep_predict in range(self.n_steps_forecast_horizon):
            # make design matrix with previous predictions
            self.logger.info(f'timestep {idx_timestep_predict}: creating design matrices')

            design_matrices_params = []
            for idx_param_pdf in range(self.n_params_pdf):
                design_matrix = self.make_design_matrix_timestep_param_pdf(
                    x,
                    predictions_previous[:idx_timestep_predict],
                    datetimes_x,
                    idx_timestep_predict,
                    idx_param_pdf,  # only one pdf param for the point forecast.
                )

                design_matrices_params.append(design_matrix)
            # design_matrices.append(design_matrix)
            self.logger.info(f'Fitting timestep {idx_timestep_predict} using created design matrix.')

            dtype = self.DTYPE

            optimizers_params = [
                self.class_optimizer[idx_param_pdf](**self.kwargs_optimizer[idx_param_pdf])
                for idx_param_pdf in range(self.n_params_pdf)
            ]

            get_loss_grads = self.make_fn_for_optimizer(y_fit_scaled, indices_y_fit_valid_data, design_matrices_params, idx_timestep_predict, weight_penalty, *args, **kwargs)

            n_iters_per_param = self.n_iters_per_param
            params_pdf_fit = tf.constant(params_pdf_fit)

            def fn_base(idx_param_pdf_):
                def fn():
                    loss_, grads_, params_ = get_loss_grads(idx_param_pdf_)

                    # optimizer = tf.switch_case(idx_param_pdf, fns_get_optimizer)
                    optimizer = optimizers_params[idx_param_pdf_]
                    optimizer.apply_gradients(
                        [(grads_, params_)]
                    )
                    return loss_
                return fn

            fns_get_loss_grads_params = [
                fn_base(0),
                fn_base(1),
                fn_base(2),
                fn_base(3)
            ][:self.n_params_pdf]


            def fit_iter_param(iter, iter_param, idx_param_pdf):
                loss_ = tf.switch_case(
                    idx_param_pdf,
                    fns_get_loss_grads_params,
                )

                return loss_

            # @tf.function(jit_compile=True)
            def fit_param_pdf(iter, idx_param_pdf):
                return tf.map_fn(
                    lambda iter_param: fit_iter_param(iter, iter_param, idx_param_pdf),
                    tf.range(n_iters_per_param),
                    fn_output_signature=dtype
                )

            def fit_params(iter):
                # TODO: avoid using for-loops to prevent extensive retracing
                losses = tf.map_fn(
                    lambda idx_param: fit_param_pdf(iter, idx_param),
                    params_pdf_fit,
                    fn_output_signature=dtype
                )
                tf.print('iter: ', iter, 'loss: ', losses[:, -1])
                return losses

            @tf.function(jit_compile=False)
            def fit_timestep():
                tf.map_fn(
                    fit_params,
                    tf.range(n_iters_use[idx_timestep_predict]),
                    fn_output_signature=dtype,
                )

            fit_timestep()

            # delete the design matrix
            del design_matrices_params
            tf.keras.backend.clear_session()

    @staticmethod
    def make_params_distribution_timestep(design_matrices, params_timestep, indices_params_pdf_ar, indices_params_pdf_constant, fns_link=None):
        params_pdf = []
        # TODO: tf map_fn to avoid retracing
        for idx_param_pdf in range(len(indices_params_pdf_ar)+ len(indices_params_pdf_constant)):
            if idx_param_pdf in indices_params_pdf_constant:
                if fns_link is not None:
                    params_pdf.append(fns_link[idx_param_pdf](params_timestep[idx_param_pdf]))
                else:
                    params_pdf.append(params_timestep[idx_param_pdf])
            else:
                params_matrix = params_timestep[idx_param_pdf][..., tf.newaxis]
                param_pdf_raw = tf.linalg.matmul(
                    design_matrices[idx_param_pdf],
                    params_matrix,
                )

                if fns_link is not None:
                    param_pdf = fns_link[idx_param_pdf](param_pdf_raw[..., 0])
                else:
                    param_pdf = param_pdf_raw[..., 0]
                params_pdf.append(param_pdf)

        return params_pdf

    def make_fn_for_optimizer(
            self,
            y_fit_scaled,
            indices_y_fit_valid_data,
            design_matrices_timestep,
            idx_timestep_predict,
            weight_mse,
            *args,
            **kwargs

    ):
        params_timestep = self.params_tf[idx_timestep_predict]
        fn_loss = self.make_fn_loss(y_fit_scaled, indices_y_fit_valid_data, design_matrices_timestep, idx_timestep_predict, weight_mse, *args, **kwargs)

        def get_loss_grads(idx_param):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                params_ = params_timestep[idx_param]

                tape.watch(params_)
                loss_ = fn_loss()
                grads = tape.gradient(loss_, params_)

            return loss_, grads, params_
        return get_loss_grads

    def make_fn_penalty(self):
        def penalty(weight_penalty, y_valid_scaled, distribution):
            mean = distribution.mean()
            mse = tf.reduce_mean((y_valid_scaled - mean) ** 2)
            return mse * weight_penalty

        return penalty


    def make_fn_loss(self, y_fit_scaled, indices_y_fit_valid_data, design_matrices_timestep, idx_timestep_predict, weight_penalty, *args, **kwargs):
        y_valid_scaled = tf.gather(y_fit_scaled, indices_y_fit_valid_data)[:, idx_timestep_predict]

        design_matrices_valid = [
            tf.gather(design_matrix_param, indices_y_fit_valid_data)
            for design_matrix_param in design_matrices_timestep
        ]
        params_timestep = self.params_tf[idx_timestep_predict]
        fns_link = self.link_functions
        make_params_distribution_timestep = self.make_params_distribution_timestep
        fn_make_distribution = self.fn_make_distribution
        indices_params_constant = self.params_pdf_constant
        indices_params_ar = [idx for idx in range(self.n_params_pdf) if idx not in self.params_pdf_constant]
        lambdas_lasso_timestep = self.lambdas_lasso[idx_timestep_predict]
        fn_penalty = self.make_fn_penalty()

        def fn_loss():
            params_pdf = make_params_distribution_timestep(
                design_matrices_valid,
                params_timestep,
                indices_params_ar,
                indices_params_constant,
                fns_link
            )
            distribution = fn_make_distribution(*params_pdf)
            penalty = fn_penalty(weight_penalty, y_valid_scaled, distribution)

            lassos_params_pdf = []
            for params_param_pdf, lambda_param_pdf in zip(params_timestep, lambdas_lasso_timestep):
                lassos_params_pdf.append(lambdas_lasso_timestep * tf.reduce_sum(tf.abs(params_param_pdf)))

            lasso = tf.reduce_sum(lassos_params_pdf)

            return - tf.math.reduce_mean(distribution.log_prob(y_valid_scaled)) + penalty + lasso

        return fn_loss

    def make_fn_loss_grads(self, params_pdf_fit, n_timesteps, *args, fit_only_last_timestep=False, **kwargs):
        model = self.make_model_tf(
            self.x_in_sample,
            self.datetimes_x_in_sample,
            tf.range(self.get_n_datapoints_prediction(self.x_in_sample)),
            n_timesteps=n_timesteps
        )

        y_tuple = self.make_y_tuple(self.y_fit_scaled)
        if n_timesteps is not None:
            y_tuple = y_tuple[:n_timesteps]
        indices_y_fit_valid_data = self.indices_y_fit_valid_data

        if params_pdf_fit is None:
            params_tf = self.params_tf
        else:
            self.logger.info(f'Fitting pdf params {params_pdf_fit}')
            params_tf = [
                [params_timestep[idx_param_pdf] for idx_param_pdf in params_pdf_fit]
                for params_timestep in self.params_tf
            ]

        params_tf_flattened = []
        if fit_only_last_timestep:
            params_tf_flattened.extend(params_tf[n_timesteps-1])
        else:
            for idx_timestep in range((n_timesteps if n_timesteps is not None else self.n_steps_forecast_horizon)):
                params_tf_flattened.extend(params_tf[idx_timestep])
        params_tf_flattened.extend(self.get_additional_params_fit(params_pdf_fit, n_timesteps, fit_only_last_timestep, *args, **kwargs))

        def get_penalty_lasso_(n_timesteps_):
            # if lambdas_lasso_ar is not None:
            result = 0.

            for params_timestep, lambdas_timestep in zip(params_tf[:n_timesteps_], self.lambdas_lasso[:n_timesteps_]):
                for params_param_pdf, lambda_param_pdf in zip(params_timestep, lambdas_timestep):
                    if lambda_param_pdf != 0:
                        result += tf.reduce_sum(tf.abs(params_param_pdf)) * lambda_param_pdf

            return result

        def loss():
            if fit_only_last_timestep:
                log_prob = model.log_prob_parts(y_tuple)[n_timesteps-1]
            else:
                log_prob = model.log_prob(y_tuple)
            log_prob = tf.gather(log_prob, indices_y_fit_valid_data)

            log_likelihood = tf.reduce_sum(log_prob)

            penalty_lasso = get_penalty_lasso_(
                n_timesteps if n_timesteps is not None else self.n_steps_forecast_horizon
            )
            normalization_constant = log_prob.shape[0] if fit_only_last_timestep else log_prob.shape[0] * len(y_tuple)

            log_likelihood_normed = log_likelihood / normalization_constant

            result = -log_likelihood_normed + penalty_lasso

            tf.summary.scalar('loss', result)
            tf.summary.scalar('log_likelihood', log_likelihood)
            tf.summary.scalar('log_likelihood_normed', log_likelihood_normed)

            return result

        @tf.function(jit_compile=False)
        def get_loss_grads():
            with tf.GradientTape() as tape:
                loss_ = loss()
                grads = tape.gradient(loss_, params_tf_flattened)

            return loss_, grads

        return get_loss_grads, params_tf_flattened

    def get_additional_params_fit(self, params_pdf_fit, n_timesteps, fit_only_last_timestep,  *args, **kwargs):
        return []


    def get_log_likelihood(self, data: pd.DataFrame, bool_datapoints_valid=None):
        self.logger.warning('get_log_likelihood(): MUST RESPECT DATA NANS INVALID!!!')

        x, y, datetimes_x = self.make_x_y(data, True)
        y = self.slice_tensor_sample_predictions(y)
        y_scaled = (y-self.mean_scaling_y) / self.std_scaling_y

        bool_x_valid, y_for_valid, datetimes_x = self.make_x_y(bool_datapoints_valid,
                                                               omit_unknown_future_truth=True)
        indices_valid = self.get_indices_datapoints_history_valid(bool_x_valid, True)


        # FIXME: self.y is 1d, right?
        predictions_previous = tf.unstack(
            y_scaled,
            axis=1)

        indices_params_ar = [idx for idx in range(self.n_params_pdf) if idx not in self.params_pdf_constant]

        log_probs_datapoints_timesteps = []
        for idx_timestep_predict in range(self.n_steps_forecast_horizon):
            # make design matrix with previous predictions
            self.logger.info(f'timestep {idx_timestep_predict}: creating design matrices')

            design_matrices_params = []
            for idx_param_pdf in range(self.n_params_pdf):
                design_matrix = self.make_design_matrix_timestep_param_pdf(
                    x,
                    predictions_previous[:idx_timestep_predict],
                    datetimes_x,
                    idx_timestep_predict,
                    idx_param_pdf,  # only one pdf param for the point forecast.
                )

                design_matrices_params.append(design_matrix)

            self.logger.info(f'Getting log prob for timestep {idx_timestep_predict} using created design matrix.')
            params_pdf_timestep = self.make_params_distribution_timestep(
                design_matrices_params,
                self.params_tf[idx_timestep_predict],
                indices_params_ar,
                self.params_pdf_constant,
                self.link_functions
            )
            distribution = self.fn_make_distribution(*params_pdf_timestep)

            print(y_scaled.shape)
            print(params_pdf_timestep[0].shape)

            log_prob_datapoints_timestep = distribution.log_prob(y_scaled[:, idx_timestep_predict])
            log_prob_datapoints_timestep_valid = tf.gather(log_prob_datapoints_timestep, indices_valid)
            log_probs_datapoints_timesteps.append(log_prob_datapoints_timestep_valid)


            del design_matrices_params
            tf.keras.backend.clear_session()

        log_probs_datapoints_timesteps_tf = tf.stack(log_probs_datapoints_timesteps, axis=-1)
        result = tf.reduce_sum(log_probs_datapoints_timesteps_tf)

        return result

    def make_point_forecast_distribution(self, distribution):
        return distribution.mean()

    def predict_x_scaled(self, x, datetimes_x, n_samples, *args, return_params=False, use_mean_predictions_previous=False, **kwargs):
        self.logger.info('Sampling: making model')
        n_datapoints_prediction = self.get_n_datapoints_prediction(x)

        # result is [sample, datapoint, timestep]
        result = tf.zeros((n_samples, n_datapoints_prediction, 0), dtype=self.DTYPE)
        point_forecasts = tf.zeros((n_samples, n_datapoints_prediction, 0), dtype=self.DTYPE)

        self.logger.info('Making design matrices base')
        design_matrices_base = self.make_design_matrices_base(
            x,
            datetimes_x,
        )

        indices_params_ar = [idx for idx in range(self.n_params_pdf) if idx not in self.params_pdf_constant]

        self.logger.info(f'Making base pdf parameters for all timesteps')
        params_pdf_base = []
        for design_matrices_base_timestep, params_timestep in tqdm(zip(design_matrices_base, self.params_tf)):
            params_base = []
            for idx_param in range(self.n_params_pdf):
                n_params_base = design_matrices_base_timestep[idx_param].shape[1]
                param_sliced = params_timestep[idx_param][-n_params_base:] if idx_param in indices_params_ar else params_timestep[idx_param]
                params_base.append(param_sliced)

            params_pdf_base.append(
                self.make_params_distribution_timestep(
                    design_matrices_base_timestep,
                    params_base,
                    indices_params_ar,
                    self.params_pdf_constant,
                    None
                )
            )

        self.logger.info(f'Making interaction and prediction pdf params')
        for idx_timestep_predict in range(self.n_steps_forecast_horizon):
            # make design matrix with previous predictions
            self.logger.info(f'timestep {idx_timestep_predict}: Sampling')
            samples_timestep = []
            point_forecasts_timestep = []

            # make design matrices for all samples

            for idx_sample in tqdm(range(n_samples)):
                design_matrices_params_interact_predictions = []
                logging.info('Making design matrices of interaction and predictions')
                for idx_param_pdf in range(self.n_params_pdf):

                    # TODO: vectorication
                    t1 = timeit.default_timer()
                    design_matrix_sample_interact_prediction = np.ndarray((n_datapoints_prediction, 0), dtype=self.DTYPE)
                    predictions_prev = tf.unstack(result[idx_sample, ...], axis=-1)
                    points_forecasts_prev = tf.unstack(point_forecasts[idx_sample, ...], axis=-1)
                    t2 = timeit.default_timer()
                    logging.info(f'Created empty data arrays in {t2-t1}')

                    if self.pairs_interactions:
                        # s = time.time()
                        t1 = timeit.default_timer()
                        design_matrix_interactions = self.make_design_matrix_interactions_timestep_param_pdf(
                            x,
                            # list [timestep], with tensor elements of [datapoint[
                            points_forecasts_prev if self.use_mean_interaction else predictions_prev,
                            datetimes_x,
                            idx_timestep_predict,
                            idx_param_pdf,
                        )
                        t2 = timeit.default_timer()
                        logging.info(f'Created interaction design matrix in {t2 - t1}')

                        design_matrix_sample_interact_prediction = tf.concat((design_matrix_interactions, design_matrix_sample_interact_prediction), axis=-1)
                    if idx_timestep_predict > 0:
                        t1 = timeit.default_timer()
                        design_matrix_predictions = self.make_design_matrix_predicions_timestep_param_pdf(
                            x,
                            points_forecasts_prev if use_mean_predictions_previous else predictions_prev,
                            datetimes_x,
                            idx_timestep_predict,
                            idx_param_pdf,
                        )
                        t2 = timeit.default_timer()
                        logging.info(f'Created prediction design matrix in {t2 - t1}')
                        design_matrix_sample_interact_prediction = tf.concat((design_matrix_predictions, design_matrix_sample_interact_prediction), axis=-1)
                    design_matrices_params_interact_predictions.append(design_matrix_sample_interact_prediction)

                params_interactions_predictions = []
                logging.info('getting model params')

                for idx_param in range(self.n_params_pdf):
                    n_params = design_matrices_params_interact_predictions[idx_param].shape[-1]
                    params_all = self.params_tf[idx_timestep_predict][idx_param]
                    param_sliced = params_all[:n_params] if idx_param in indices_params_ar else params_all
                    params_interactions_predictions.append(param_sliced)

                self.logger.info(f'making pdf params interaction predictions from design matrices')
                params_pdf_interactions_predictions = self.make_params_distribution_timestep(
                    design_matrices_params_interact_predictions,
                    params_interactions_predictions,
                    indices_params_ar,
                    self.params_pdf_constant,
                    None
                )

                params_pdf = [
                    fn_link(param_base + param_interaction_prediction) for fn_link, param_base, param_interaction_prediction
                    in zip(self.link_functions, params_pdf_base[idx_timestep_predict], params_pdf_interactions_predictions)
                ]

                self.logger.info(f'making and sampling from distribution')
                distribution = self.fn_make_distribution(*params_pdf)

                logging.info('sampling from distribution')
                sample = distribution.sample()
                point_forecast_sample = self.make_point_forecast_distribution(distribution)
                samples_timestep.append(sample)
                point_forecasts_timestep.append(point_forecast_sample)
                logging.info('sampled')

            samples_timestep_tf = tf.stack(samples_timestep, axis=0)
            point_forecasts_timestep_tf = tf.stack(point_forecasts_timestep, axis=0)
            result = tf.concat([result, samples_timestep_tf[..., tf.newaxis]], axis=-1)
            point_forecasts = tf.concat([point_forecasts, point_forecasts_timestep_tf[..., tf.newaxis]], axis=-1)

            if self.pairs_interactions:
                del design_matrix_interactions
            if idx_timestep_predict > 0:
                del design_matrix_predictions

            del distribution
            del params_pdf_interactions_predictions
            del params_all
            del param_sliced
            del params_interactions_predictions
            del design_matrix_sample_interact_prediction
            del design_matrices_params_interact_predictions

            del predictions_prev
            del points_forecasts_prev
            tf.keras.backend.clear_session()

            if return_params:
                raise NotImplementedError
                params_pdf_timesteps.append(params_pdf)
            else:
                del params_pdf


        return result

