import logging
import numbers
from collections import OrderedDict
from functools import lru_cache
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.sparse import csr_matrix
from tqdm import tqdm
import src.common_functions.scores as scores
import json
from src.common_functions.misc import json_to_params, params_to_json, get_pbas, dump_json_gzip, load_json_gzip
import tensorflow as tf
import os
from copy import deepcopy
import hashlib
from src.constants.misc import *


class ModelBase:
    """ Base class providing a framework for defining models. """
    DTYPE = np.float64
    n_params_pdf = 1
    idx_param_pdf_loc = 0
    name_model_base = 'base'
    suffix_data_nans_invalid = 'w_nans_invalid'

    def __init__(
            self,
            variable_target: str,
            n_steps_forecast_horizon: int,
            # for each column: [timestep[param_pdf[threshold[lag]]]]
            #   convention: timesteps relative to time of prediciton
            lags_columns: {str: [[[[int]]]]},
            # [timestep[param_pdf[threshold]]]
            thresholds: {str: [[[float]]]},
            seasons_days_weeks: [[int]],
            n_seasons_annual: int,
            # TODO: this should be in derived forecast classes.
            # either float for all or [timestep[param_pdf]]
            lambdas_lasso: [[float]] or float,
            means_scaling_design_matrices_base: np.ndarray = None,
            stds_scaling_design_matrices_base: np.ndarray = None,
            means_scaling_design_matrices_prediction: np.ndarray = None,
            stds_scaling_design_matrices_prediction: np.ndarray = None,
            means_scaling_design_matrices_interaction: np.ndarray = None,
            stds_scaling_design_matrices_interaction: np.ndarray = None,
            mean_scaling_y: np.ndarray = None,
            std_scaling_y: np.ndarray = None,
            make_design_matrix_sparse=False,
            timedelta_data=None,
            date_reference_seasonals_year=None,
            # For each group of days given in seasons_days_weeks
            #   the indices of intraday seasons to include.
            #   can be used for model shrinkage, so that only
            #   relevant seasons are included in the model.
            # [timestep[param_pdf[annual season[group_days[season in day]]]]]
            indices_steps_day_include: [[[[[int]]]]] = None,
            model_description=None,
            logger=None,
            **kwargs

    ):
        self.model_description = model_description
        if indices_steps_day_include is not None:
            if len(indices_steps_day_include[0][0]) != n_seasons_annual:
                raise ValueError(f'indices_steps_day_include has {len(indices_steps_day_include)}'
                                 f'annual seasons, but model has {n_seasons_annual}')
            # TODO: also assess num of day groups consistent with seasons_days_weeks

        self.logger = logger if logger is not None else logging.getLogger('Model')

        # Remark: cannot insert dummies here because self.n_steps_day is needed but unknown until fitting.

        self.indices_steps_day_include = indices_steps_day_include

        # Used in GAMLSS
        self.indices_columns_variables_design_matrices_base = None

        # Remark: zero-indexed, starting at monday
        self.make_design_matrix_sparse = make_design_matrix_sparse
        self.n_seasons_annual = n_seasons_annual

        self.get_min_lag_wrt_first_step_predict_timestep_column.cache_clear()
        self.get_min_lag_wrt_first_step_predict_timestep.cache_clear()

        self.timedelta_data = timedelta_data

        # ensure that thresholds of each variable and lags for each threshold are sorted.
        #   this is important for saving and loading the model, as human readalbe params are
        #   stored as dicts, so that order is not preserved. Hence, on loading the model, the order is reconstructed
        #   by sorting.
        thresholds_sorted = self.sort_lags_or_thresholds(thresholds)
        lags_columns_sorted = self.sort_lags_or_thresholds(lags_columns)

        # # convert the threshold sets to tensors / arrays
        self.thresholds = self.nested_iterables_to_array(thresholds_sorted, lambda x: tf.constant(x, dtype=self.DTYPE))
        self.thresholds = self.convert_dtype_nested(thresholds_sorted, self.DTYPE)
        self.lags_columns = lags_columns_sorted

        self.n_steps_forecast_horizon = n_steps_forecast_horizon
        self.seasons_days_weeks = seasons_days_weeks

        # target variable
        self.variable_target = variable_target

        if isinstance(lambdas_lasso, (float, int)):
            self.logger.info('Lambda given for all timesteps and params pdf. Converting.')
            self.lambdas_lasso = [
                [lambdas_lasso] * self.n_params_pdf
                for idx_timestep in range(self.n_steps_forecast_horizon)
            ]
        else:
            self.lambdas_lasso = lambdas_lasso

        # will be filled with first datetime of training data:
        #   use beginning of year to ensure alignment
        self.date_reference_seasonals_year = date_reference_seasonals_year

        if (
                mean_scaling_y is not None
                and std_scaling_y is not None
                and means_scaling_design_matrices_base is not None
                and stds_scaling_design_matrices_base is not None
                and means_scaling_design_matrices_prediction is not None
                and stds_scaling_design_matrices_prediction is not None
                and means_scaling_design_matrices_interaction is not None
                and stds_scaling_design_matrices_interaction is not None
        ):
            self.mean_scaling_y = mean_scaling_y
            self.std_scaling_y = std_scaling_y

            self.means_scaling_design_matrices_base = self.nested_iterables_to_array(
                means_scaling_design_matrices_base, lambda x: tf.constant(x, dtype=self.DTYPE)
            )
            self.stds_scaling_design_matrices_base = self.nested_iterables_to_array(
                stds_scaling_design_matrices_base, lambda x: tf.constant(x, dtype=self.DTYPE)
            )
            self.means_scaling_design_matrices_prediction = self.nested_iterables_to_array(
                means_scaling_design_matrices_prediction, lambda x: tf.constant(x, dtype=self.DTYPE)
            )
            self.stds_scaling_design_matrices_prediction = self.nested_iterables_to_array(
                stds_scaling_design_matrices_prediction, lambda x: tf.constant(x, dtype=self.DTYPE)
            )
            self.means_scaling_design_matrices_interaction = self.nested_iterables_to_array(
                means_scaling_design_matrices_interaction, lambda x: tf.constant(x, dtype=self.DTYPE)
            )
            self.stds_scaling_design_matrices_interaction = self.nested_iterables_to_array(
                stds_scaling_design_matrices_interaction, lambda x: tf.constant(x, dtype=self.DTYPE)
            )
        else:
            # different scalings for each timestep and param pdf
            self.means_scaling_design_matrices_base = [
                [None] * self.n_params_pdf
                for idx_timestep in range(self.n_steps_forecast_horizon)
            ]
            self.stds_scaling_design_matrices_base = [
                [None] * self.n_params_pdf
                for idx_timestep in range(self.n_steps_forecast_horizon)
            ]
            self.means_scaling_design_matrices_prediction = [
                [None] * self.n_params_pdf
                for idx_timestep in range(self.n_steps_forecast_horizon)
            ]
            self.stds_scaling_design_matrices_prediction = [
                [None] * self.n_params_pdf
                for idx_timestep in range(self.n_steps_forecast_horizon)
            ]
            self.means_scaling_design_matrices_interaction = [
                [None] * self.n_params_pdf
                for idx_timestep in range(self.n_steps_forecast_horizon)
            ]
            self.stds_scaling_design_matrices_interaction = [
                [None] * self.n_params_pdf
                for idx_timestep in range(self.n_steps_forecast_horizon)
            ]
            self.mean_scaling_y = None
            self.std_scaling_y = None

        if set(self.lags_columns.keys()).union([self.variable_target]) != set(self.thresholds.keys()):
            raise ValueError(f'Lags_columns and thresholds do not have same columns.')

    @property
    def pairs_interactions(self):
        pairs_unsorted = [key for key in self.lags_columns.keys() if isinstance(key, (list, tuple))]

        return sorted(pairs_unsorted, key=lambda pair: pair[0] + pair[1])

    @property
    def hash_wo_lambda(self):
        info = self.model_info
        info.pop(PARAM_LAMBDA)
        str_info = json.dumps(params_to_json(info), sort_keys=True, ensure_ascii=True)
        str_info_enc = str_info.encode('utf-8')
        return hashlib.md5(
            str_info_enc
        ).hexdigest()

    @property
    def model_info(self):
        return {
            NAME_MODEL: self.name_model_base,
            PARAM_LAMBDA: self.lambdas_lasso,
            PARAM_LAGS: self.lags_columns,
            PARAM_THRESHOLDS: self.thresholds,
            PARAM_SEASONS_DAYS_WEEK: self.seasons_days_weeks,
            PARAM_N_SEASONS_YEAR: self.n_seasons_annual,
            PARAM_N_STEPS_FORECAST_HORIZON: self.n_steps_forecast_horizon,
            PARAM_VARIABLE_TARGET: self.variable_target,
        }

    @property
    def n_params_fitted(self):
        params_fitted = self.params_fitted
        n_params = 0
        for params_timestep in params_fitted:
            for params_param_pdf in params_timestep:
                n_params += np.size(params_param_pdf)
        return n_params

    @property
    def n_params_fitted_active(self):
        params_fitted = self.params_fitted
        n_params = 0
        for params_timestep in params_fitted:
            for params_param_pdf in params_timestep:
                params_param_pdf_np = np.asarray(params_param_pdf)
                n_params += np.size(params_param_pdf_np[np.abs(params_param_pdf_np) > 1e-6])
        return n_params

    @property
    def kwargs_store_model_subclass(self):
        return {}

    @property
    def kwargs_store_model_base(self):
        return dict(
            class_model=type(self).__name__,
            variable_target=self.variable_target,
            n_steps_forecast_horizon=self.n_steps_forecast_horizon,
            lags_columns=self.lags_columns,
            thresholds=self.thresholds,
            seasons_days_weeks=self.seasons_days_weeks,
            date_reference_seasonals_year=self.date_reference_seasonals_year,
            n_seasons_annual=self.n_seasons_annual,
            means_scaling_design_matrices_base=self.means_scaling_design_matrices_base,
            stds_scaling_design_matrices_base=self.stds_scaling_design_matrices_base,
            means_scaling_design_matrices_prediction=self.means_scaling_design_matrices_prediction,
            stds_scaling_design_matrices_prediction=self.stds_scaling_design_matrices_prediction,
            means_scaling_design_matrices_interaction=self.means_scaling_design_matrices_interaction,
            stds_scaling_design_matrices_interaction=self.stds_scaling_design_matrices_interaction,
            mean_scaling_y=self.mean_scaling_y,
            std_scaling_y=self.std_scaling_y,
            make_design_matrix_sparse=self.make_design_matrix_sparse,
            timedelta_data=self.timedelta_data,
            lambdas_lasso=self.lambdas_lasso,
            params=self.params_fitted,
            indices_steps_day_include=self.indices_steps_day_include
        )

    @property
    def kwargs_store_model(self):

        result = {**self.kwargs_store_model_base, **self.kwargs_store_model_subclass}

        return result

    @property
    def columns_lags_no_interaction(self):
        return sorted([key for key in self.lags_columns.keys() if not isinstance(key, (list, tuple))])

    @property
    def params_fitted_human_readable(self):
        raise NotImplementedError('Still buggy; dont use')
        params_fitted = self.params_fitted

        self.logger.warning(
            'getting human readable params could be wrong due to merging od prediction and base matrix!')

        def get_params(columns, fn_lag_valid, indices_params_start):
            result_columns_ = {}
            indices_params = indices_params_start
            # params_fitted = self.params_fitted
            for name_column in columns:
                lags_column_timesteps = self.lags_columns[name_column]
                result_timesteps_ = []
                for idx_timestep, lags_column_timestep in enumerate(lags_column_timesteps):
                    result_params_pdf_ = []
                    for idx_param_pdf, lags_column_param_pdf in enumerate(lags_column_timestep):
                        result_thresholds_ = []
                        for lags_column_threshold in lags_column_param_pdf:
                            result_lags_ = []
                            for lag in lags_column_threshold:
                                if not fn_lag_valid(idx_timestep, lag):
                                    continue
                                idx_param = indices_params[idx_timestep][idx_param_pdf]
                                result_lags_.append(params_fitted[idx_timestep][idx_param_pdf][idx_param])
                                indices_params[idx_timestep][idx_param_pdf] += 1

                            result_thresholds_.append(result_lags_)
                        result_params_pdf_.append(result_thresholds_)
                    result_timesteps_.append(result_params_pdf_)
                result_columns_[name_column] = result_timesteps_

            return result_columns_

        indices_params = [
            [0 for idx_param_pdf in range(self.n_params_pdf)]
            for idx_timestep in range(self.n_steps_forecast_horizon)
        ]

        # first is predictions
        params_base_predictions = get_params(
            [self.variable_target],
            lambda idx_timestep_, lag_: lag_ >= -idx_timestep_,
            indices_params
        )

        # then, interactions
        params_base_interactions = get_params(
            self.pairs_interactions,
            lambda idx_timestep_, lag_: True,
            indices_params
        )

        # base lags
        params_base_lags = get_params(
            self.columns_needed,
            lambda idx_timestep_, lag_: lag_ < -idx_timestep_,
            indices_params
        )

        # seasonals
        params_seasonals = []
        params_fitted = self.params_fitted
        for idx_timestep in range(self.n_steps_forecast_horizon):
            seasonals_timestep = []
            for idx_param_pdf in range(self.n_params_pdf):
                seasonals_param_pdf = []
                for indices_use_season_year in self.indices_steps_day_include[idx_timestep][idx_param_pdf]:
                    seasonals_season_year = []
                    for indices_use_group_days in indices_use_season_year:
                        seasonals_groups_days = []
                        for index_use_step_day in indices_use_group_days:
                            idx_param = indices_params[idx_timestep][idx_param_pdf]
                            seasonals_groups_days.append(params_fitted[idx_timestep][idx_param_pdf][idx_param])
                            indices_params[idx_timestep][idx_param_pdf] += 1
                        seasonals_season_year.append(seasonals_groups_days)
                    seasonals_param_pdf.append(seasonals_season_year)
                seasonals_timestep.append(seasonals_param_pdf)
            params_seasonals.append(seasonals_timestep)

        return params_base_lags, params_base_predictions, params_base_interactions, params_seasonals

    @property
    def name_model(self):
        return f'{self.name_model_base}_COLS_{sorted(self.indices_columns_x.keys())}'

    @property
    def idx_slice_first_datapoints(self):
        """Index for slicing first datapoint of first day"""
        raise NotImplementedError

    @property
    def columns_needed(self):
        """Columns for whcih lags are defined"""

        set_vars = {self.variable_target}
        for key in self.lags_columns.keys():
            # decompose intercation terms
            if isinstance(key, (tuple, list)):
                for name_column in key:
                    set_vars.add(self.get_name_column_clean(name_column))
            else:
                set_vars.add(self.get_name_column_clean(key))

        return sorted(list(set_vars))

    @property
    def indices_columns_x(self):
        return {name_column: idx for idx, name_column in enumerate(self.columns_needed)}

    @property
    def idx_column_x_target(self):
        return self.columns_needed.index(self.variable_target)

    @property
    def n_datapoints_history_needed(self):
        # self.logger.warning('n_datapoints_history_needed(): NEED TO CONSIDER cumulation!')
        """Number of days needed in data do do prediction with given model specification"""
        min_lag_wrt_first_step_prediction = self.min_lag_wrt_first_step_prediction
        # min lag is negative if history is needed.
        return np.maximum(
            0,
            -min_lag_wrt_first_step_prediction
        )

    @property
    def min_lag_wrt_first_step_prediction(self):
        return np.min(
            [
                self.get_min_lag_wrt_first_step_predict_timestep(idx_timestep)
                for idx_timestep in range(self.n_steps_forecast_horizon)
            ]
        )

    def scale_y(self, y):
        return (y - self.mean_scaling_y) / self.std_scaling_y

    def rescale_y(self, y_scaled):
        return y_scaled * self.std_scaling_y + self.mean_scaling_y

    def make_y_fit(self, y_in_sample_wo_unknown_future):
        return self.slice_tensor_sample_predictions(y_in_sample_wo_unknown_future)

    def make_y_fit_vector(self, y_fit):
        return tf.flatten(y_fit)

    @property
    def index_start_design_matrix_seasonal(self):
        return -(len(self.seasons_days_weeks) * self.n_steps_day * self.n_seasons_annual)

    @property
    def lags__thresholds_target_prediction(self):
        """ Used for building design matrix for predictions. Sort out lags not covered by past samples
            and also sort out thresholds for which lags are not present then."""
        lags = []
        thresholds = []
        for idx_timestep, (lags_timestep, thresholds_timestep) in enumerate(
                zip(self.lags_columns[self.variable_target], self.thresholds[self.variable_target])):
            lags_filtered_timestep, thresholds_filtered_timestep = [], []
            for lags_param_pdf, thresholds_param_pdf in zip(lags_timestep, thresholds_timestep):
                thresholds_param_pdf_filtered_list = []
                lags_param_pdf_filtered = []
                for idx_threshold, lags_threshold in enumerate(lags_param_pdf):
                    indices_lags_valid = tf.where(tf.greater_equal(lags_threshold, -idx_timestep))
                    if tf.greater(tf.size(indices_lags_valid), 0):
                        thresholds_param_pdf_filtered_list.append(thresholds_param_pdf[idx_threshold])
                        lags_threshold_filtered = tf.gather(
                            lags_threshold, indices_lags_valid
                        )[:, 0]
                        lags_param_pdf_filtered.append(lags_threshold_filtered)
                thresholds_filtered_timestep.append(tf.stack(thresholds_param_pdf_filtered_list))
                lags_filtered_timestep.append(lags_param_pdf_filtered)
            lags.append(lags_filtered_timestep)
            thresholds.append(thresholds_filtered_timestep)

        return lags, thresholds

    @property
    def params_fitted(self):
        if not self.check_model_fitted():
            raise ValueError('Model is not fitted.')
        result = []

        for idx_timestep in range(self.n_steps_forecast_horizon):
            result_timestep = []
            for idx_param_pdf in range(self.n_params_pdf):
                result_timestep.append(self.get_params_fitted_timestep_param_pdf(idx_timestep, idx_param_pdf))
            result.append(result_timestep)

        return result

    def get_timedelta_data(self, datetimes_x):
        # TODO: more sophisitcated and error if incosistent timedelta
        return np.timedelta64(datetimes_x[1] - datetimes_x[0], 'ns')

    @property
    def n_steps_day(self):
        return int(np.timedelta64(timedelta(days=1)) / self.timedelta_data)

    @staticmethod
    def fn_transform_data(x):
        # TODO: verify that this works.
        return x
        # return tf.where(tf.greater(x, 0), tf.sqrt(x), tf.zeros_like(x))

    @staticmethod
    def sort_lags_or_thresholds(value):
        keys_sorted = sorted(value.keys(), key=lambda x: x[0] + x[1] if isinstance(x, tuple) else x)

        result = OrderedDict(
            [(key, value[key]) for key in keys_sorted]
        )
        return result

    @staticmethod
    def get_path_store_model_data(path_store_model, suffix=''):
        path_wo_ext = os.path.splitext(path_store_model)[0]

        return f'{path_wo_ext}_data_{suffix}.json'

    @classmethod
    def process_kwargs_from_file(cls, model_deserialized, *args, **kwargs):
        """ For loading model from file of different model class: add missing params for initilizing this class"""
        result = deepcopy(model_deserialized)
        result[KWARGS_MODEL] = {**result[KWARGS_MODEL], **kwargs}

        return result

    @classmethod
    def model_from_file(cls, filepath, *args, **kwargs):
        def all_subclasses(cls):
            result = {}
            for subclass in cls.__subclasses__():
                result[subclass.__name__] = subclass
                result = {
                    **result,
                    subclass.__name__: subclass,
                    **all_subclasses(subclass)
                }

            return result

        with open(filepath) as f:
            model_json = json.load(f)

        name_class = model_json[KWARGS_MODEL]['class_model']
        class_model = all_subclasses(cls)[name_class]

        model_deserialized = json_to_params(model_json, np.asarray)
        model_deserialized_w_kwargs = class_model.process_kwargs_from_file(model_deserialized, *args, **kwargs)
        model = class_model(**model_deserialized_w_kwargs[KWARGS_MODEL])
        model.setup_loading_file_subclass(filepath)

        return model

    def setup_loading_file_subclass(self, filepath):
        pass

    @classmethod
    def params_human_readable_to_params(cls, params):
        raise NotImplementedError

    @staticmethod
    def make_y_tuple(y, indices_valid_data=None):
        """Make y as tuple """
        if indices_valid_data is None:
            return tf.unstack(y, axis=1)
        else:
            return tf.unstack(
                tf.gather(y, indices_valid_data),
                axis=1
            )

    @staticmethod
    def sort_items_params(params: dict):
        """Get sorted items of dict where keys are the form name_number"""
        return sorted(params.items(), key=lambda key, value: int(key.split('_')[-1]))

    def get_params_fitted_timestep_param_pdf(self, idx_timestep, idx_param_pdf):
        raise NotImplementedError

    def sort_numeric_args(self, args):
        if isinstance(args, np.ndarray):
            return np.sort(args)
        if isinstance(args, tf.Tensor):
            return tf.sort(args)
        if not isinstance(args, (list, tuple, dict)):
            raise ValueError(f'args must be list, tuple or dict. args: {args}')
        if len(args) == 0:
            return args
        if isinstance(args, (tuple, list)):
            if isinstance(args[0], (tuple, list, dict, np.ndarray, tf.Tensor)):
                return [self.sort_numeric_args(arg) for arg in args]
            elif isinstance(args[0], (float, int)):
                return sorted(args)
            else:
                raise ValueError(f'type of first element {args[0]} not known sortable type')
        elif isinstance(args, dict):
            if isinstance(list(args.values())[0], (tuple, list, dict, np.ndarray, tf.Tensor)):
                return {key: self.sort_numeric_args(arg) for key, arg in args.items()}
            else:
                raise ValueError(f'type of first element {args[0]} not known sortable type')

    def convert_dtype_nested(self, iterables, dtype):
        # TODO: catch case of mixed iterable and non-iterable elements
        if (
                isinstance(iterables, tf.Tensor)
        ):
            return tf.cast(iterables, dtype)
        elif isinstance(iterables, np.ndarray):
            return iterables.astype(dtype)
        elif iterables is None:
            return iterables
        elif isinstance(iterables, list):
            return [self.convert_dtype_nested(element, dtype) for element in iterables]
        elif isinstance(iterables, tuple):
            return (self.convert_dtype_nested(element, dtype) for element in iterables)
        elif isinstance(iterables, numbers.Number):
            return dtype(iterables)
        elif isinstance(iterables, dict):
            return {
                key: self.convert_dtype_nested(element, dtype)
                for key, element in iterables.items()
            }
        else:
            return iterables

    def nested_iterables_to_array(self, iterables, constructor_array):
        # TODO: catch case of mixed iterable and non-iterable elements.
        if (
                isinstance(iterables, (tf.Tensor, np.ndarray))
                or isinstance(iterables, (list, tuple)) and len(iterables) == 0
                or isinstance(iterables, (list, tuple)) and isinstance(iterables[0], (int, float))
        ):
            return constructor_array(iterables)
        elif iterables is None:
            return iterables
        elif isinstance(iterables, list):
            return [
                self.nested_iterables_to_array(element, constructor_array)
                for element in iterables
            ]
        elif isinstance(iterables, tuple):
            return (
                self.nested_iterables_to_array(element, constructor_array)
                for element in iterables
            )
        elif isinstance(iterables, dict):
            return {
                key: self.nested_iterables_to_array(element, constructor_array)
                for key, element in iterables.items()
            }
        else:
            raise ValueError(f'Unrecognized type of element {iterables}')

    def get_params_lags_thresholds_shrunken(self, threshold_lasso):
        # raise ValueError('Shrinking does not work with cumulating in base matrix anymore.')
        logging.warning('get_params_lags_thresholds_shrunken(): Check if it works with cumulation in base matrix!')
        # must be in base part, as error occurs already for timestep 0.

        #####
        n_dropped = 0
        n_params = self.n_params_fitted

        indices_params_current = [
            [0 for idx_param_pdf in range(self.n_params_pdf)]
            for idx_timestep in range(self.n_steps_forecast_horizon)
        ]

        lags_filled = {
            col_: [
                [
                    [[] for threshold in self.thresholds[col_][idx_timestep][idx_param_pdf]]
                    for idx_param_pdf in range(self.n_params_pdf)
                ]
                for idx_timestep in range(self.n_steps_forecast_horizon)
            ] for col_ in self.lags_columns.keys()
        }

        params_shrunken = [
            [np.ndarray([0], dtype=self.DTYPE) for idx_param_pdf in range(self.n_params_pdf)]
            for idx_timestep in range(self.n_steps_forecast_horizon)
        ]

        def fill_lags_non_zero(
                params_shrunken_,
                columns,
                fn_lag_valid,
                indices_params_start,
                lags_filled_,
                n_dropped,
                params_scaling
        ):

            params_scaling_cleaned = [
                [
                    [np.ndarray([0], dtype=self.DTYPE) for idx_param_pdf in range(self.n_params_pdf)]
                    for idx_timestep in range(self.n_steps_forecast_horizon)
                ] for set_params in params_scaling
            ]

            idx_column_design_matrix = [
                [0 for idx_param_pdf in range(self.n_params_pdf)]
                for idx_timestep in range(self.n_steps_forecast_horizon)
            ]

            indices_params = indices_params_start
            params_fitted = self.params_fitted
            for name_column in columns:
                for idx_timestep, lags_column_timestep in enumerate(self.lags_columns[name_column]):
                    for idx_param_pdf, lags_column_param_pdf in enumerate(lags_column_timestep):
                        for idx_threshold, lags_column_threshold in enumerate(lags_column_param_pdf):
                            lags_filled_threshold = lags_filled_[name_column][idx_timestep][idx_param_pdf][
                                idx_threshold]
                            for lag in lags_column_threshold:
                                if not fn_lag_valid(idx_timestep, lag):
                                    continue
                                idx_param = indices_params[idx_timestep][idx_param_pdf]
                                param = params_fitted[idx_timestep][idx_param_pdf][idx_param]
                                if abs(param) >= threshold_lasso:
                                    lags_filled_threshold.append(lag)
                                    params_shrunken_[idx_timestep][idx_param_pdf] = np.append(
                                        params_shrunken_[idx_timestep][idx_param_pdf], param)

                                    idx_column_design_matrix_ = idx_column_design_matrix[idx_timestep][idx_param_pdf]
                                    for idx_set_scaling, set_scaling in enumerate(params_scaling):
                                        params_scaling_cleaned[idx_set_scaling][idx_timestep][
                                            idx_param_pdf] = np.append(
                                            params_scaling_cleaned[idx_set_scaling][idx_timestep][idx_param_pdf],
                                            params_scaling[idx_set_scaling][idx_timestep][idx_param_pdf][
                                                idx_column_design_matrix_]
                                        )
                                else:
                                    n_dropped += 1
                                idx_column_design_matrix[idx_timestep][idx_param_pdf] += 1
                                indices_params[idx_timestep][idx_param_pdf] += 1

            return n_dropped, params_scaling_cleaned, idx_column_design_matrix

        # first is predictions
        n_dropped, (means_predictions, stds_predictions), _ = fill_lags_non_zero(
            params_shrunken,
            [self.variable_target],
            lambda idx_timestep_, lag_: lag_ >= -idx_timestep_,
            indices_params_current,
            lags_filled,
            n_dropped,
            (self.means_scaling_design_matrices_prediction, self.stds_scaling_design_matrices_prediction)
        )

        # then, interactions
        n_dropped, (means_interactions, stds_interactions), _ = fill_lags_non_zero(
            params_shrunken,
            self.pairs_interactions,
            lambda idx_timestep_, lag_: True,
            indices_params_current,
            lags_filled,
            n_dropped,
            (self.means_scaling_design_matrices_interaction, self.stds_scaling_design_matrices_interaction)
        )

        # base lags
        n_dropped, (means_base, stds_base), idx_column_design_matrix_base = fill_lags_non_zero(
            params_shrunken,
            self.columns_lags_no_interaction,
            lambda idx_timestep_, lag_: lag_ < -idx_timestep_,
            indices_params_current,
            lags_filled,
            n_dropped,
            (self.means_scaling_design_matrices_base, self.stds_scaling_design_matrices_base)
        )

        # build indices_steps_day_include
        indices_steps_day_include = []

        params_scaling_raw = [self.means_scaling_design_matrices_base, self.stds_scaling_design_matrices_base]

        params_fitted = self.params_fitted
        for idx_timestep in range(self.n_steps_forecast_horizon):
            indices_timestep = []
            for idx_param_pdf in range(self.n_params_pdf):
                indices_param_pdf = []
                for indices_use_season_year in self.indices_steps_day_include[idx_timestep][idx_param_pdf]:
                    indices_season_annual = []
                    for indices_use_group_days in indices_use_season_year:
                        indices_group_days = []
                        for index_use_step_day in indices_use_group_days:
                            idx_param = indices_params_current[idx_timestep][idx_param_pdf]
                            param = params_fitted[idx_timestep][idx_param_pdf][idx_param]
                            if abs(param) >= threshold_lasso:
                                indices_group_days.append(index_use_step_day)
                                params_shrunken[idx_timestep][idx_param_pdf] = np.append(
                                    params_shrunken[idx_timestep][idx_param_pdf],
                                    param
                                )
                                for idx_set_scaling, set_scaling in enumerate((means_base, stds_base)):
                                    set_scaling[idx_timestep][idx_param_pdf] = np.append(
                                        set_scaling[idx_timestep][idx_param_pdf],
                                        params_scaling_raw[idx_set_scaling][idx_timestep][idx_param_pdf][
                                            idx_column_design_matrix_base[idx_timestep][idx_param_pdf]
                                        ]
                                    )
                                idx_column_design_matrix_base[idx_timestep][idx_param_pdf] += 1

                            indices_params_current[idx_timestep][idx_param_pdf] += 1
                        indices_season_annual.append(indices_group_days)
                    indices_param_pdf.append(indices_season_annual)
                indices_timestep.append(indices_param_pdf)
            indices_steps_day_include.append(indices_timestep)

        self.logger.info(f'Dropped {n_dropped} of {n_params} params.')

        return (params_shrunken,
                lags_filled,
                self.thresholds,  # TODO
                indices_steps_day_include,
                means_predictions,
                stds_predictions,
                means_interactions,
                stds_interactions,
                means_base,
                stds_base
                )

    def save_model_shrunken(self, filepath_save, threshold_lasso):

        (params_shrunken,
         lags_shrunken,
         thresholds_shrunken,  # TODO
         indices_steps_day_include,
         means_predictions,
         stds_predictions,
         means_interactions,
         stds_interactions,
         means_base,
         stds_base
         ) = self.get_params_lags_thresholds_shrunken(threshold_lasso)

        kwargs = {}
        kwargs['lags_columns'] = lags_shrunken
        kwargs['thresholds'] = thresholds_shrunken
        kwargs['params'] = params_shrunken
        kwargs['indices_steps_day_include'] = indices_steps_day_include

        kwargs['means_scaling_design_matrices_interaction'] = means_interactions
        kwargs['stds_scaling_design_matrices_interaction'] = stds_interactions

        kwargs['means_scaling_design_matrices_prediction'] = means_predictions
        kwargs['stds_scaling_design_matrices_prediction'] = stds_predictions
        kwargs['means_scaling_design_matrices_base'] = means_base
        kwargs['stds_scaling_design_matrices_base'] = stds_base

        self.save_model(filepath_save, kwargs)

    def save_data_train(self, path_output, columns_save=None):
        self.save_data_df(self.data_train, path_output, columns_save)

    def save_data_df(self, df, path_output, columns_save=None):
        if columns_save is None:
            dump_json_gzip(path_output, params_to_json(df.to_dict()))
        else:
            df[columns_save].to_json(path_output)
            dump_json_gzip(path_output, params_to_json(df[columns_save].to_dict()))

    @staticmethod
    def load_data_df(path_data):
        dict_data = json_to_params(load_json_gzip(path_data), lambda x: list(x))
        result = pd.DataFrame.from_dict(dict_data)

        return result

    def save_model_additional_subclass(self, path_output, save_data):
        pass

    def save_model(self, path_output, kwargs_additional=None):

        if kwargs_additional is None:
            kwargs_additional = {}

        kwargs = {**self.kwargs_store_model, **kwargs_additional}
        kwargs_serializable = params_to_json(kwargs)

        result = {
            INFO_HUMAN_READABME: {
                'description': self.model_description if self.model_description is not None else {}
            },
            KWARGS_MODEL: kwargs_serializable
        }

        with open(path_output, 'w') as f:
            json.dump(result, f, indent=2)

        self.save_model_additional_subclass(path_output, True)

    @classmethod
    def floor_days_datetimes(cls, datetimes: np.ndarray):
        return cls.floor_datetimes(datetimes, 'D')

    @classmethod
    def floor_datetimes(cls, datetimes: np.ndarray, interval):
        # copy is important because otherwise datetimes is modified in place
        df = pd.DataFrame(datetimes.copy())

        for name_column in df.columns:
            if interval == 'W':
                df[name_column] = df[name_column].dt.to_period('W-SUN').dt.start_time
            elif interval == 'AS':
                df[name_column] = df[name_column].apply(lambda x: x.replace(hour=12, minute=0, day=1, month=1))
            else:
                df[name_column] = df[name_column].dt.floor(interval)

        return df.to_numpy(dtype=datetimes.dtype) if datetimes.ndim != 1 else df.to_numpy(dtype=datetimes.dtype)[:, 0]

    def get_datetimes_y(self, datetimes_x):
        result = np.ndarray((datetimes_x.shape[0], self.n_steps_forecast_horizon), datetimes_x.dtype)
        for idx_timestep in range(self.n_steps_forecast_horizon):
            result[:, idx_timestep] = datetimes_x + self.timedelta_data * (idx_timestep + 1)
        return result

    def check_model_fitted(self):
        raise NotImplementedError

    def get_n_params_needed_timestep_param_pdf(
            self,
            idx_timestep,
            idx_param_pdf
    ):
        # obtain the number of params from the scaling parameters because there is one for each design matrix column.
        result = (
                np.size(self.means_scaling_design_matrices_base[idx_timestep][idx_param_pdf])
                + np.size(self.means_scaling_design_matrices_prediction[idx_timestep][idx_param_pdf])
        )
        if self.pairs_interactions:
            result += np.size(self.means_scaling_design_matrices_interaction[idx_timestep][idx_param_pdf])

        return result

    @lru_cache()
    def get_min_lag_wrt_first_step_predict_timestep_column(self, idx_timestep, name_column):
        if not isinstance(name_column, (list, tuple)) and name_column.startswith(CUMULATE):
            n_steps_cumulate = self.get_n_steps_cumulate(name_column)
        else:
            n_steps_cumulate = 0
        min_lag_wrt_timestep = 10000000000000  # TODO
        for lags_timestep in self.lags_columns[name_column][idx_timestep]:
            for lags_param_pdf in lags_timestep:
                for lags_threshold in lags_param_pdf:
                    if tf.size(lags_threshold) != 0:
                        min_lag_wrt_timestep = np.minimum(np.min(lags_threshold), min_lag_wrt_timestep)

        # cumulation means that n_steps_cumulate previous steps are needed.
        min_lag_wrt_timestep -= n_steps_cumulate

        return self.get_lag_wrt_first_step_predict(min_lag_wrt_timestep, idx_timestep)

    def get_lag_wrt_first_step_predict(self, lag, idx_timestep):
        return lag + idx_timestep

    @lru_cache()
    def get_min_lag_wrt_first_step_predict_timestep(self, idx_timestep):
        # lags are [timestep[param_pdf[threshold[lag]]]]
        return np.min([
            self.get_min_lag_wrt_first_step_predict_timestep_column(idx_timestep, name_column)
            for name_column in self.lags_columns.keys()
        ])

    @lru_cache()
    def get_lags_columns_wrt_first_step_predict(self):

        result = {}
        # TODO: consider interactions

        for name_column, lags_column in self.lags_columns.items():
            result_column = set()
            for idx_timestep, lags_thresholds in enumerate(lags_column):
                for lags_threshold in lags_thresholds:
                    for lag in lags_threshold:
                        result_column.add(self.get_lag_wrt_first_step_predict(lag, idx_timestep))

            result[name_column] = sorted(result_column)

        return result

    def get_n_regressors_timestep_param_pdf(self, idx_timestep, idx_param_pdf):
        means_base = self.means_scaling_design_matrices_base[idx_timestep][idx_param_pdf]
        means_interaction = self.means_scaling_design_matrices_interaction[idx_timestep][idx_param_pdf]
        means_prediction = self.means_scaling_design_matrices_prediction[idx_timestep][idx_param_pdf]
        result = (
                means_base.shape[-1] +
                (means_interaction.shape[-1] if means_interaction is not None else 0) +
                means_prediction.shape[-1]
        )

        return result

    def slice_tensor_sample_predictions(self, t):
        idx_datapoint_first = self.get_idx_datapoint_first_prediction()

        return t[idx_datapoint_first:]

    def slice_x_y_sample_predictions(self, x, y):
        """ From x and y used for prediction, obtain for each prediction
            the x of last day before prediction and the truth according to prediction.

        :param x:
        :param y:
        :return:
        """
        return [
            self.slice_tensor_sample_predictions(x),
            self.slice_tensor_sample_predictions(y),
        ]

    def get_weekdays(self, datetimes: np.ndarray):
        df = pd.DataFrame(datetimes)

        for name_col in df.columns:
            df[name_col] = df[name_col].dt.dayofweek

        numpy_ = df.to_numpy()

        return numpy_ if datetimes.ndim > 1 else numpy_[:, 0]

    def set_up_other_variables(self, data, *args, **kwargs):
        """ Perform setup for subclasses (e.g. store quantiles for rectifiers)

        :return: nothing
        """
        pass

    def make_x_y(self, data: pd.DataFrame, omit_unknown_future_truth: bool) -> (np.ndarray, np.ndarray, np.ndarray):
        """ Make x array with [datapoint, column]
        """
        data_columns_needed = data[self.columns_needed] if self.columns_needed is not None else data
        x = data_columns_needed.to_numpy(dtype=self.DTYPE)

        # remark: important to ensure nanoseconds unit.
        #   especially when working with time deltas, which must be ns too.
        datetimes_x = data_columns_needed.index.to_numpy().astype('datetime64[ns]')

        # y is x of target column for future
        y = np.ndarray((x.shape[0], self.n_steps_forecast_horizon), dtype=self.DTYPE)
        for idx_step_predict in range(self.n_steps_forecast_horizon):
            n_step_predict = idx_step_predict + 1
            y[:, idx_step_predict] = np.roll(
                x[:, self.idx_column_x_target],
                -n_step_predict,
                0
            )
            y[-n_step_predict:, idx_step_predict] = np.nan

        # for the forecast horizon, no y values are present, but x history is known.
        n_steps_unknown_future = self.n_steps_forecast_horizon
        return (
            x[:-n_steps_unknown_future],
            y[:-n_steps_unknown_future],
            datetimes_x[:-n_steps_unknown_future]
        ) if omit_unknown_future_truth else (x, y, datetimes_x)

    def get_days_week_y(self, datetimes_x: np.ndarray):
        """Weekdays for y from weekdays of X"""
        datetimes_y = self.get_datetimes_y(datetimes_x)

        return self.get_weekdays(datetimes_y)

    def get_difference_time_reference_x(self, datetimes_x: np.ndarray,
                                        datetime_reference: datetime):
        reference = np.datetime64(datetime_reference)
        difference_times = datetimes_x - reference

        return difference_times

    def get_n_timesteps_after_reference_x(self, datetimes_x: np.ndarray,
                                          datetime_reference: datetime):
        """Days after an (arbitrarily selected) reference date to ensure consistency of annual
            seasonals for different datasets
        """
        difference = self.get_difference_time_reference_x(datetimes_x, datetime_reference)
        timedelta_reference = np.timedelta64(self.timedelta_data, 'ns')
        result = (difference / timedelta_reference).astype(int)

        return result

    def get_n_timesteps_after_reference_y(self, datetimes_x: np.ndarray,
                                          datetime_reference: datetime):
        n_steps_x = self.get_n_timesteps_after_reference_x(datetimes_x, datetime_reference)
        result = np.ndarray((n_steps_x.shape[0], self.n_steps_forecast_horizon), dtype=int)
        for idx_step_predict in range(self.n_steps_forecast_horizon):
            result[:, idx_step_predict] = n_steps_x + idx_step_predict + 1

        return result

    @staticmethod
    def make_x_column_lagged_day_wrt_prediction_time(x_column, lag_day):
        # remark: convention is that lags must be negative (or zero if forecast of variable is used).
        #   rolling with -lag then rolls data forward in time
        # remark: -1 because x is d-1 relative to day of predicted value. Hence, shift is
        #   one less than lag.
        return np.roll(x_column, -lag_day - 1, axis=0)

    def get_n_datapoints_before_first_prediction(self):
        return self.n_datapoints_history_needed - 1

    def get_idx_datapoint_first_prediction(self):
        return max(self.get_n_datapoints_before_first_prediction(), 0)

    def get_n_datapoints_prediction(self, x):
        self.logger.error('get_n_datapoints_prediction() works incorrectly for realtime predictions,'
                          'where slice_n_datapoints_first_prediction is False! Hence, functions to make design matrices'
                          ' timestep must also be reworked! At least for benchmarks. Maybe, simply remove slice_n_datapoints_first_prediction '
                          'to make everything simpler and clearer.')
        return x.shape[0] - self.get_idx_datapoint_first_prediction()

    @classmethod
    def lag_data(cls, data, lag, idx_timestep):
        """ Lag data representing last known values by a lag given
            relative to idx_timestep in future.

        """
        # # Roll column for lagging. +1 because x corresponds to lag: -1 already.
        lag_wrt_timestep_predict_0 = cls.get_lag_wrt_first_timestep_predict(lag, idx_timestep)

        return np.roll(data, -(lag_wrt_timestep_predict_0 + 1), axis=-1)

    def make_part_design_matrix_base_lags_timestep_param_pdf(
            self,
            x,
            datetimes_x,
            idx_timestep,
            idx_param_pdf
    ):
        result_list = []
        # make lags
        # remark: iterating over column names  from columns needed so that order is always preserved.

        # remark: interactions are not in columns_needed.
        for name_column in self.columns_lags_no_interaction:
            x_column = self.get_data_column_from_x(x, name_column)
            # x_column = x[:, idx_column]
            lags_column_thresholds = self.lags_columns[name_column][idx_timestep][idx_param_pdf]
            thresholds_column = self.thresholds[name_column][idx_timestep][idx_param_pdf]

            for lags_column_threshold, threshold in zip(lags_column_thresholds, thresholds_column):
                x_thresholded = self.threshold_x_lagged(x_column, threshold)
                for lag in lags_column_threshold:
                    # skip lags of target variable that must be taken from prediction design matrix
                    #   no lags of unknown data for other vars allowed.
                    if lag >= -idx_timestep:
                        idx_column = self.indices_columns_x[name_column]

                        if idx_column == self.idx_column_x_target:
                            continue
                        else:
                            raise ValueError(f'lag column {name_column}: {lag} for idx_timestep {idx_timestep} '
                                             f'would need unknown data.')

                    x_lagged_thresholded = self.lag_data(x_thresholded, lag, idx_timestep)

                    result_list.append(x_lagged_thresholded)

        result = np.stack(result_list, axis=1) if result_list else np.ndarray((x.shape[0], 0), dtype=self.DTYPE)
        result_sliced = self.slice_design_matrix_n_datapoints_first_prediction(result)

        result_sliced = self.fn_transform_data(result_sliced)

        return result_sliced

    @staticmethod
    def get_lag_wrt_first_timestep_predict(lag, idx_timestep):
        # lags are given w.r.t. idx_timestep, but
        #   x is correspomding to lag: -1  w.r.t. idx_timestep=0
        #   hence, if e.g. idx_timestep is 2, the lag w.r.t. idx_timestep=0
        #   is lag+idx_timestep (considering that lag is defined negative for past information)
        return lag + idx_timestep

    def threshold_x_lagged(self, x_lagged, threshold):
        return self.threshold_x_lagged_static(
            x_lagged,
            threshold,
            self.make_design_matrix_sparse
        )

    @staticmethod
    @tf.function(experimental_follow_type_hints=True)
    def threshold_x_lagged_static(x_lagged: tf.Tensor, threshold: tf.Tensor, make_design_matrix_sparse: tf.Tensor):
        # should work with x_lagged matrix out of the box? no, as multiple thresholds for each column exist.
        result = tf.minimum(x_lagged, threshold)

        result = tf.cond(
            tf.logical_and(make_design_matrix_sparse, tf.math.is_finite(threshold)),
            lambda: result - threshold,
            lambda: result
        )

        return result

    def get_name_column_interaction(self, column_a, column_b, lag_a, lag_b):
        return f'{INTERACTION}_{column_a}_{column_b}_lag_{lag_a}_lag_{lag_b}'

    def get_name_column_clean(self, name_column):
        if name_column.startswith(CUMULATE):
            str_n_steps = name_column[len(CUMULATE) + 1:].split('_')[0]

            return name_column[len(CUMULATE) + 1 + len(str_n_steps) + 1:]
        else:
            return name_column

    def get_n_steps_cumulate(self, name_column):
        # +1 because underscore
        str_n_steps = name_column[len(CUMULATE) + 1:].split('_')[0]
        return int(str_n_steps)

    def get_data_column_from_x(self, x, name_column: str):
        name_column_clean = self.get_name_column_clean(name_column)
        data = x[..., self.indices_columns_x[name_column_clean]]

        if name_column.startswith(CUMULATE):
            result = np.zeros_like(data)

            n_steps = self.get_n_steps_cumulate(name_column)
            # +1 because step of 0 would mean only the original data
            for idx_step in range(n_steps + 1):
                rolled = np.roll(data, idx_step, -1)
                result += rolled
        else:
            result = data

        return result

    def make_data_lagged_interaction(self, x, name_column, idx_timestep, lag, predictions):

        if lag >= -idx_timestep:
            if name_column != self.variable_target or lag > -1:
                raise ValueError(f'lag_a for interaction with {name_column}: {lag} for idx_timestep {idx_timestep} '
                                 f'would need unknown data.')
            else:
                idx_prediction = len(predictions) + lag
                return predictions[idx_prediction]
        else:
            data = self.get_data_column_from_x(x, name_column)
            data_lagged = self.lag_data(data, lag, idx_timestep)

            # if it is normal column, it has all rows of x, also the ones that must be sliced till first prediction
            return self.slice_design_matrix_n_datapoints_first_prediction(data_lagged)

    def make_design_matrix_interactions_timestep_param_pdf(
            self,
            x,
            # list [timestep], with tensor elements of [datapoint[
            predictions_timesteps_prev_scaled,
            datetimes_x,
            idx_timestep,
            idx_param_pdf,
            predictions_are_scaled=True,
            do_scaling=True
    ):

        # FIXME
        self.logger.warning('Interactions: need to consider max lag in self.max_lag!')

        if predictions_are_scaled:
            predictions = [prediction * self.std_scaling_y + self.mean_scaling_y for prediction in
                           predictions_timesteps_prev_scaled]
        else:
            predictions = predictions_timesteps_prev_scaled

        # FIXME: will not work when loading shrunken model
        result_list = []
        for pair_interaction in self.pairs_interactions:
            lags_pair_thresholds = self.lags_columns[pair_interaction][idx_timestep][idx_param_pdf]
            thresholds_pair = self.thresholds[pair_interaction][idx_timestep][idx_param_pdf]

            # build all interactions of column a with lags of column b
            for lags_a_b_threshold, threshold in zip(lags_pair_thresholds, thresholds_pair):
                for lag_a_b in lags_a_b_threshold:
                    a_lagged = self.make_data_lagged_interaction(x, pair_interaction[0], idx_timestep, lag_a_b[0],
                                                                 predictions)
                    b_lagged = self.make_data_lagged_interaction(x, pair_interaction[1], idx_timestep, lag_a_b[1],
                                                                 predictions)

                    # interaction = np.sqrt(a_lagged * b_lagged)
                    interaction = a_lagged * b_lagged

                    interaction_thresholded = self.threshold_x_lagged(interaction, threshold)
                    result_list.append(interaction_thresholded)

        result = np.stack(result_list, axis=-1) if result_list else np.ndarray((x.shape[0], 0), dtype=self.DTYPE)

        if do_scaling:
            result = (
                             result -
                             self.means_scaling_design_matrices_interaction[idx_timestep][idx_param_pdf][
                                 tf.newaxis, ...]
                     ) / self.stds_scaling_design_matrices_interaction[idx_timestep][idx_param_pdf][
                         tf.newaxis, ...]

        return result

    def make_design_matrix_predicions_timestep_param_pdf(
            self,
            x,
            # list [timestep], with tensor elements of [datapoint[
            predictions_timesteps_prev_scaled,
            datetimes_x,
            idx_timestep,
            idx_param_pdf,
            scale_matrix=True,
            predictions_are_scaled=True
    ):

        if predictions_are_scaled:
            predictions = [prediction * self.std_scaling_y + self.mean_scaling_y for prediction in
                           predictions_timesteps_prev_scaled]
        else:
            predictions = predictions_timesteps_prev_scaled

        lags_target, thresholds_target = self.lags__thresholds_target_prediction

        lags_column_thresholds = lags_target[idx_timestep][idx_param_pdf]
        thresholds_column = thresholds_target[idx_timestep][idx_param_pdf]

        result_list = []
        for idx_threshold in range(thresholds_column.shape[0]):
            threshold = thresholds_column[idx_threshold]
            lags_threshold = lags_column_thresholds[idx_threshold]

            for lag in lags_threshold:
                prediction_lagged = predictions[lag]
                result_list.append(self.threshold_x_lagged(prediction_lagged, threshold))

        result = np.stack(
            result_list,
            axis=-1
        )

        if scale_matrix:
            return (
                           result - self.means_scaling_design_matrices_prediction[idx_timestep][idx_param_pdf][
                       tf.newaxis, ...]
                   ) / self.stds_scaling_design_matrices_prediction[idx_timestep][idx_param_pdf][tf.newaxis, ...]
        else:
            return result

    @staticmethod
    # @tf.function
    def make_design_matrix_predicions_timestep_param_pdf_static(
            dtype,
            # list [timestep], with tensor elements of [datapoint] or possibly [sample, datapoint]
            predictions_timesteps_prev_scaled,
            idx_timestep,
            idx_param_pdf,
            # lags_columns[var_target] with only lags valid for prediction matrix (hence, lag >= idx_timestep.
            #   cannot be calculated in this function because tf.where does not work with xla.
            lags_target_prediction,
            variable_target,
            # same as lags_columns: only thresholds with lags are allowed to be present.
            thresholds_target_prediction,
            means_scaling_design_matrices_prediction,
            stds_scaling_design_matrices_prediction,
            mean_scaling_y,
            std_scaling_y,
            make_design_matrix_sparse,
            fn_threshold_x_lagged,
            n_datapoints_prediction,
            fn_transform_data,
            scale_matrix=True,
            predictions_are_scaled=tf.constant(True)
            # predictions_are_scaled=True
    ):
        lags_column_thresholds = lags_target_prediction[idx_timestep][idx_param_pdf]
        thresholds_column = thresholds_target_prediction[idx_timestep][idx_param_pdf]

        # zeros to catch thae case that no threshold is present.
        #   cannot use cond to catch the case because then len of resulting tensor can be 0, which cannot be
        #   concat. tensorfuck.
        results_thresholds = [
            tf.zeros((0, n_datapoints_prediction), dtype)
        ]

        for idx_threshold in range(thresholds_column.shape[0]):
            lags_column_threshold = lags_column_thresholds[idx_threshold]
            threshold = thresholds_column[idx_threshold]

            # skip lags of target variable that must be taken from prediction design matrix
            #   no lags of unknown data for other vars allowed.
            def fn_lag(lag):
                # tf.print(idx_lag)
                # yes, tf gather cannot deal with negative args... Hence, construct positive one...
                lag_reverse = len(predictions_timesteps_prev_scaled) + lag
                prediction_lag = tf.gather(predictions_timesteps_prev_scaled, lag_reverse)
                # incoming prediction is scaled (which is necessary because the function gets predicitons from
                #   in tf joint distribution).
                #   but rescaling is not needed if the input is not scaled (when using this function to get scaling params)
                prediction_lag_rescaled = tf.cond(
                    predictions_are_scaled,
                    lambda: prediction_lag * std_scaling_y + mean_scaling_y,
                    lambda: prediction_lag
                )
                return fn_threshold_x_lagged(
                    prediction_lag_rescaled,
                    threshold,
                    make_design_matrix_sparse
                )

            result_threshold = tf.map_fn(
                fn_lag,
                lags_column_threshold,
                # remark: output signature is important to prevent obtaining tensor with empty shape if lags are empty.
                fn_output_signature=tf.TensorSpec(
                    shape=(n_datapoints_prediction,),
                    dtype=dtype
                ),
                name='map_lags_make_design_matrix_prediction'
            )

            results_thresholds.append(result_threshold)

        result = tf.transpose(tf.concat(
            results_thresholds,
            axis=0,
            name='concat_make_desing_matrix_predictions'
        ))

        result = fn_transform_data(result)

        result = tf.cond(
            scale_matrix,
            lambda: (
                            result - means_scaling_design_matrices_prediction[idx_timestep][idx_param_pdf][
                        tf.newaxis, ...]
                    ) / stds_scaling_design_matrices_prediction[idx_timestep][idx_param_pdf][tf.newaxis, ...],
            lambda: result
        )

        return result

    def make_part_design_matrix_seasonals_timestep_param_pdf(
            self,
            x,
            datetimes_x,
            idx_timestep,
            idx_param_pdf,
    ):
        """ For each annual season, use individual DOW dummies with smooth transition
            using b-splines / sine

        :param days_week_x:
        :param datetimes_x:
        :return:
        """
        self.logger.debug('Seasonals: getting datetimes_y')
        # get datetimes_y
        datetimes_y = self.get_datetimes_y(datetimes_x)

        # make copy of datetimes_y with hours, mins, secs, etc. to 0
        self.logger.debug('Seasonals: getting datetimes_y_days')
        datetimes_y_days = self.floor_days_datetimes(datetimes_y)

        # make differences to datetimes_y and divide by timedelta of step in data
        #   in order to obtain step numbers for the days.
        self.logger.debug('Seasonals: getting timesteps_days')
        timesteps_days = ((datetimes_y - datetimes_y_days) / np.timedelta64(self.timedelta_data, 'ns')).astype(int)[:,
                         idx_timestep]

        self.logger.debug('Seasonals: getting days_week_y')
        days_week_y_all = self.get_days_week_y(datetimes_x)
        days_week_y = days_week_y_all[:, idx_timestep]

        # for each group of days of weeks in self
        dummys_steps_one_season = np.ndarray(
            (x.shape[0], len(self.seasons_days_weeks) * self.n_steps_day),
            dtype=self.DTYPE
        )
        idx = 0
        self.logger.debug('Seasonals: iterating through days week groups and day timesteps.')
        for group_days_week in self.seasons_days_weeks:
            for timestep_day in range(self.n_steps_day):
                dummys_steps_one_season[:, idx] = np.logical_and(
                    np.isin(days_week_y, group_days_week),
                    timesteps_days == timestep_day
                ).astype(self.DTYPE)
                idx += 1

        # get days of year for prediction step
        if self.n_seasons_annual != 1:
            steps_after_reference_y = self.get_n_timesteps_after_reference_y(
                datetimes_x,
                self.date_reference_seasonals_year
            )[:, idx_timestep]
            result = np.tile(dummys_steps_one_season, (1, self.n_seasons_annual))

            for idx_season_year in range(self.n_seasons_annual):
                dummys_steps_one_season_season_year = dummys_steps_one_season
                weights_season_year = self.weights_nth_period(
                    self.n_seasons_annual,
                    steps_after_reference_y / (PERIOD_DAYS_YEAR * self.n_steps_day),
                    idx_season_year
                )
                idx_start = idx_season_year * dummys_steps_one_season_season_year.shape[1] * idx_season_year
                idx_stop = idx_start + dummys_steps_one_season_season_year.shape[1]
                result[:, idx_start:idx_stop] *= weights_season_year[:, np.newaxis]

        else:
            result = dummys_steps_one_season

        # extract the corresponding steps according to indices_steps_day_include.
        indices_use = []
        idx_start = 0
        for indices_use_season_year in self.indices_steps_day_include[idx_timestep][idx_param_pdf]:
            for indices_use_group_days in indices_use_season_year:
                for index_use_step_day in indices_use_group_days:
                    indices_use.append(idx_start + index_use_step_day)

                # indices aregiven for each day and must be mapped to indices of result.
                idx_start += self.n_steps_day
        result = tf.gather(result, tf.constant(indices_use, tf.int32), axis=1)

        result_sliced = self.slice_design_matrix_n_datapoints_first_prediction(
            result,
        )

        return result_sliced

    @staticmethod
    def weights_nth_period(frequency, x, idx_season):
        days = x * PERIOD_DAYS_YEAR

        # TODO: make order configurable
        result = get_pbas(days, dK=PERIOD_DAYS_YEAR / frequency, order=3)
        return result[:, idx_season]

    def make_design_matrix_base_timestep_param_pdf(
            self,
            x,
            datetimes_x,
            idx_timestep,
            idx_param_pdf,
            do_scaling=True
    ):

        self.logger.debug(f'Making design matrix LAGS for timestep {idx_timestep}, param {idx_param_pdf}')
        matrix_lags = self.make_part_design_matrix_base_lags_timestep_param_pdf(
            x,
            datetimes_x,
            idx_timestep,
            idx_param_pdf,
        )

        # make part with seasonal dummies
        self.logger.debug(f'Making design matrix SEASONALS for timestep {idx_timestep}, param {idx_param_pdf}')
        matrix_seasonals = self.make_part_design_matrix_seasonals_timestep_param_pdf(
            x,
            datetimes_x,
            idx_timestep,
            idx_param_pdf,
        )

        # make part with additional (interaction) terms
        design_matrix = tf.concat((matrix_lags, matrix_seasonals), axis=1)

        if do_scaling:
            design_matrix_scaled = (
                                           design_matrix -
                                           self.means_scaling_design_matrices_base[idx_timestep][idx_param_pdf][
                                               tf.newaxis, ...]
                                   ) / self.stds_scaling_design_matrices_base[idx_timestep][idx_param_pdf][
                                       tf.newaxis, ...]
        else:
            design_matrix_scaled = design_matrix
        return design_matrix_scaled

    def slice_design_matrix_n_datapoints_first_prediction(self, design_matrix):
        idx_start = self.get_idx_datapoint_first_prediction()
        # if only 1 dim, it is time series vector.
        #   else, it is 3d design matrix with possibly preceding batch dimes.
        if len(design_matrix.shape) == 1:
            return design_matrix[idx_start:]
        else:
            return design_matrix[..., idx_start:, :]

    def make_design_matrices_base(
            self,
            x,
            datetimes_x,
            n_timesteps=None
    ):
        result = []
        if n_timesteps is None:
            n_timesteps = self.n_steps_forecast_horizon
        for idx_timestep in range(n_timesteps):
            matrices_timestep = []
            for idx_param_pdf in range(self.n_params_pdf):
                # use same matrix as first pdf param if matrix will be the same.

                lags_thresholds_same_as_first = True
                for name_column in self.columns_needed:
                    lags_column_pdf_current = self.lags_columns[name_column][idx_timestep][idx_param_pdf]
                    lags_column_pdf_0 = self.lags_columns[name_column][idx_timestep][0]

                    thresholds_column_pdf_current = self.thresholds[name_column][idx_timestep][idx_param_pdf]
                    thresholds_column_pdf_0 = self.thresholds[name_column][idx_timestep][0]

                    if not np.array_equal(thresholds_column_pdf_0, thresholds_column_pdf_current):
                        lags_thresholds_same_as_first = False
                        break

                    if not len(lags_column_pdf_current) == len(lags_column_pdf_0):
                        lags_thresholds_same_as_first = False
                        break

                    for lags_current_threshold, lags_0_threshold in zip(lags_column_pdf_current, lags_column_pdf_0):
                        if not np.array_equal(lags_current_threshold, lags_0_threshold):
                            lags_thresholds_same_as_first = False
                            break

                    steps_pdf_current = self.indices_steps_day_include[idx_timestep][idx_param_pdf]
                    steps_pdf_0 = self.indices_steps_day_include[idx_timestep][0]

                    for steps_annual_current, steps_annual_0 in zip(steps_pdf_current, steps_pdf_0):
                        for steps_days_current, steps_days_0 in zip(steps_annual_current, steps_annual_0):
                            if not np.array_equal(steps_days_0, steps_days_current):
                                lags_thresholds_same_as_first = False
                                break

                if idx_param_pdf != 0 and lags_thresholds_same_as_first:
                    self.logger.info(
                        f'Reusing matrix of first pdf param for timestep {idx_timestep}, pdf param {idx_param_pdf}')
                    matrices_timestep.append(matrices_timestep[0])
                else:
                    matrices_timestep.append(
                        self.make_design_matrix_base_timestep_param_pdf(
                            x,
                            datetimes_x,
                            idx_timestep,
                            idx_param_pdf
                        )
                    )

            result.append(matrices_timestep)

        return result

    def make_design_matrix_timestep_param_pdf(
            self,
            x: np.ndarray,
            predictions_timesteps_prev_scaled: [np.ndarray],
            datetimes_x: np.ndarray,
            idx_timestep: int,
            idx_param_pdf: int,
            make_sparse: bool = False,
            design_matrix_base=None
    ) -> np.ndarray:
        """ Make design matrix for one timestep, using x and predictions from prior timesteps.
        """
        self.logger.debug(f'Making design matrix for timestep {idx_timestep}, param {idx_param_pdf}')
        # index of day for which first prediction is made
        # make base design matrix with lags of known historical data and seasonals
        design_matrix_base = self.make_design_matrix_base_timestep_param_pdf(
            x,
            datetimes_x,
            idx_timestep,
            idx_param_pdf,
        ) if design_matrix_base is None else design_matrix_base

        design_matrix = design_matrix_base

        if self.pairs_interactions:
            design_matrix_interactions = self.make_design_matrix_interactions_timestep_param_pdf(
                x,
                # list [timestep], with tensor elements of [datapoint[
                predictions_timesteps_prev_scaled,
                datetimes_x,
                idx_timestep,
                idx_param_pdf,
            )

            design_matrix = tf.concat((design_matrix_interactions, design_matrix), axis=-1)

        if idx_timestep > 0:
            design_matrix_predictions = self.make_design_matrix_predicions_timestep_param_pdf(
                x,
                predictions_timesteps_prev_scaled,
                datetimes_x,
                idx_timestep,
                idx_param_pdf,
            )

            design_matrix = tf.concat((design_matrix_predictions, design_matrix), axis=-1)

        self.logger.debug('Finished making design matrix.')

        return csr_matrix(design_matrix) if make_sparse else np.asfortranarray(design_matrix)  # Fortran for lasso

    def get_indices_datapoints_predict_valid(self, bool_x_valid, slice_to_prediction=True):
        raise NotImplementedError

    def slice_tensor_valid(self, value, indices_valid):
        return tf.gather(value, indices_valid)

    def get_bool_history_valid(self, bool_x_valid, slice_to_prediction):
        min_lag = self.min_lag_wrt_first_step_prediction
        self.logger.error('get_indices_datapoints_valid must also consider rain cumulation!!!')
        x_any_column_invalid = tf.reduce_any(np.logical_not(bool_x_valid), axis=1)
        x_any_column_invalid_lagged = [
            # Roll column for lagging. +1 because x corresponds to lag: -1 already.
            np.roll(x_any_column_invalid, -(lag + 1), axis=0)
            for lag in range(min_lag, 0)
        ]

        stacked = tf.stack(x_any_column_invalid_lagged, axis=1)
        bool_is_valid = tf.logical_not(tf.reduce_any(stacked, axis=1))

        # indices are relative to prediction-sliced datapoints
        if slice_to_prediction:
            bool_is_valid_sliced = self.slice_tensor_sample_predictions(bool_is_valid)
        else:
            bool_is_valid_sliced = bool_is_valid

        return bool_is_valid_sliced

    def get_indices_datapoints_history_valid(self, bool_x_valid, slice_to_prediction=True):
        bool_is_valid = self.get_bool_history_valid(bool_x_valid, slice_to_prediction)
        result = tf.where(bool_is_valid)[:, 0]

        return result

    def set_params_scaling_y(self, y_in_sample, indices_y_valid):
        self.logger.warning('set_params_scaling_y: Must consider only valid data')

        y_valid = self.slice_tensor_valid(y_in_sample, indices_y_valid)
        self.mean_scaling_y, self.std_scaling_y = np.nanmean(y_valid), np.nanstd(y_valid)

    def set_params_scaling_x(self, x_in_sample, datetimes_x_in_sample, y_fit, indices_y_fit_valid):
        self.logger.warning('set_params_scaling_x: Must consider only valid data')
        self.logger.info('Setting scaling parameters')
        for idx_timestep in tqdm(range(self.n_steps_forecast_horizon)):
            for idx_param_pdf in range(self.n_params_pdf):
                self.logger.debug('Making design matrix base lags.')
                matrix_lags = self.make_part_design_matrix_base_lags_timestep_param_pdf(
                    x_in_sample,
                    datetimes_x_in_sample,
                    idx_timestep,
                    idx_param_pdf,
                )
                matrix_lags = self.slice_tensor_valid(matrix_lags, indices_y_fit_valid)

                # make part with seasonal dummies
                self.logger.debug('Making design matrix base seasonals.')
                matrix_seasonals = self.make_part_design_matrix_seasonals_timestep_param_pdf(
                    x_in_sample,
                    datetimes_x_in_sample,
                    idx_timestep,
                    idx_param_pdf,
                )
                matrix_seasonals = self.slice_tensor_valid(matrix_seasonals, indices_y_fit_valid)

                # base design matrix scaling: means and stds for columns from data
                #   and no scaling at all for the seasonals.
                self.logger.debug('Calculating design matrix base scaling.')
                if self.make_design_matrix_sparse:
                    # Do not set mean in order to not loose sparsity for thresholds.
                    means_base_lags = np.zeros((matrix_lags.shape[1],), dtype=self.DTYPE)
                    # only std over non-zero entries
                    stds_base_lags = np.std(matrix_lags, axis=0, dtype=self.DTYPE)
                else:
                    means_base_lags = np.mean(matrix_lags, axis=0, dtype=self.DTYPE)
                    stds_base_lags = np.std(matrix_lags, axis=0, dtype=self.DTYPE)

                stds_base_seasonals = np.ones((matrix_seasonals.shape[1],), dtype=self.DTYPE)
                means_base_seasonals = np.zeros((matrix_seasonals.shape[1],), dtype=self.DTYPE)

                self.stds_scaling_design_matrices_base[idx_timestep][idx_param_pdf] = tf.constant(np.concatenate(
                    (stds_base_lags, stds_base_seasonals)
                ), dtype=self.DTYPE)
                self.means_scaling_design_matrices_base[idx_timestep][idx_param_pdf] = tf.constant(np.concatenate(
                    (means_base_lags, means_base_seasonals)
                ), dtype=self.DTYPE)

                self.logger.debug('finished Calculating design matrix base scaling.')

                if self.pairs_interactions:
                    design_matrix_interactions = self.make_design_matrix_interactions_timestep_param_pdf(
                        x_in_sample,
                        tf.unstack(y_fit[:, :idx_timestep], axis=1) if idx_timestep > 0 else [],
                        datetimes_x_in_sample,
                        idx_timestep,
                        idx_param_pdf,
                        do_scaling=False,
                        predictions_are_scaled=False
                    )
                    design_matrix_interactions = self.slice_tensor_valid(design_matrix_interactions,
                                                                         indices_y_fit_valid)

                    if self.make_design_matrix_sparse:
                        # Do not set mean in order to not loose sparsity for thresholds.
                        means_interaction_lags = np.zeros((design_matrix_interactions.shape[1],))
                        # only std over non-zero entries
                        stds_interaction_lags = np.std(design_matrix_interactions, axis=0)
                    else:

                        # means_interaction_lags = np.zeros((design_matrix_interactions.shape[1],))

                        means_interaction_lags = np.mean(design_matrix_interactions, axis=0)
                        stds_interaction_lags = np.std(design_matrix_interactions, axis=0)

                    self.logger.debug('Calculating design matrix interaction scaling.')
                    self.stds_scaling_design_matrices_interaction[idx_timestep][idx_param_pdf] = tf.constant(
                        stds_interaction_lags,
                        dtype=self.DTYPE)
                    self.means_scaling_design_matrices_interaction[idx_timestep][idx_param_pdf] = tf.constant(
                        means_interaction_lags,
                        dtype=self.DTYPE
                    )
                    self.logger.debug('Finished Calculating design matrix interaction scaling.')
                else:
                    self.stds_scaling_design_matrices_interaction[idx_timestep][idx_param_pdf] = tf.zeros((0,),
                                                                                                          dtype=self.DTYPE)
                    self.means_scaling_design_matrices_interaction[idx_timestep][idx_param_pdf] = tf.zeros((0,),
                                                                                                           dtype=self.DTYPE)

                if idx_timestep > 0:
                    self.logger.debug('making design matrix predictions.')
                    design_matrix_predictions = self.make_design_matrix_predicions_timestep_param_pdf(
                        x_in_sample,
                        # must be [timestep, datapoint]
                        tf.unstack(y_fit[:, :idx_timestep], axis=1),
                        datetimes_x_in_sample,
                        idx_timestep,
                        idx_param_pdf,
                        scale_matrix=False,
                        predictions_are_scaled=False
                    )

                    design_matrix_predictions = self.slice_tensor_valid(design_matrix_predictions,
                                                                        indices_y_fit_valid)

                    if self.make_design_matrix_sparse:
                        # Do not set mean in order to not loose sparsity for thresholds.
                        means_predictions_lags = np.zeros((design_matrix_predictions.shape[1],))
                        # only std over non-zero entries
                        stds_predictions_lags = np.std(design_matrix_predictions, axis=0)
                    else:
                        means_predictions_lags = np.mean(design_matrix_predictions, axis=0)
                        stds_predictions_lags = np.std(design_matrix_predictions, axis=0)

                    self.logger.debug('Calculating design matrix predictions scaling.')
                    self.stds_scaling_design_matrices_prediction[idx_timestep][idx_param_pdf] = tf.constant(
                        stds_predictions_lags,
                        dtype=self.DTYPE)
                    self.means_scaling_design_matrices_prediction[idx_timestep][idx_param_pdf] = tf.constant(
                        means_predictions_lags,
                        dtype=self.DTYPE
                    )
                    self.logger.debug('Finished Calculating design matrix predictions scaling.')

                else:
                    self.stds_scaling_design_matrices_prediction[idx_timestep][idx_param_pdf] = tf.zeros((0,),
                                                                                                         dtype=self.DTYPE)
                    self.means_scaling_design_matrices_prediction[idx_timestep][idx_param_pdf] = tf.zeros((0,),
                                                                                                          dtype=self.DTYPE)

    def do_forecast_study(self, data: pd.DataFrame, n_days_rolling_history_window, warm_start, n_days_study=None):
        raise NotImplementedError('Need to adapt to data structure of inflow.')
        n_datapoints = len(data)
        n_days = int(n_datapoints / self.n_steps_day)
        # index where last rolling window starts
        idx_day_start_last_window = n_days - 2 - n_days_rolling_history_window  # tODO: correct?

        idx_day_start_first_window = idx_day_start_last_window - n_days_study if n_days_study is not None else 0

        truths, predictions, x = [], [], []
        print('starting forecast study')
        for idx_day_start_window in tqdm(range(idx_day_start_first_window, idx_day_start_last_window)):
            idx_start = self.n_steps_day * idx_day_start_window
            idx_day_stop = idx_day_start_window + n_days_rolling_history_window + 1
            idx_stop = idx_day_stop * self.n_steps_day
            data_window = data.iloc[idx_start:idx_stop]

            prediction, x_, truth = self.fit_predict_forecast_study(
                data_window,
                warm_start=warm_start and idx_day_start_first_window != idx_day_start_first_window
            )
            truths.append(truth)
            x.append(x_)
            predictions.append(prediction)

        return (
            np.concatenate(predictions, axis=0),
            np.concatenate(x, axis=1),
            np.concatenate(truths, axis=0),
        )

    def fit_predict_forecast_study(self, data: pd.DataFrame, warm_start=False):
        """ Fit and predict for forecasting study: Model is fitted without last datapoint
            and then last datapoint is predicted.

        :return: prediction, x, truth
        """
        data_wo_last_day = data.iloc[:-self.n_steps_day]

        self.fit(data_wo_last_day, warm_start=warm_start)

        # build dataset for predicting last datapojnt

        data_predict = data.iloc[-self.n_steps_day * (self.n_datapoints_history_needed + 1):]

        # Omit last prediction based on last day's data so that returned prediction
        #   is prediction for last day in data
        return self.predict_get_truth(data_predict, omit_unknown_future_truth=True)

    def do_setup(self, data: pd.DataFrame, *args, overwrite_params_scaling=True, bool_datapoints_valid=None, **kwargs):
        # TODO: only use columns_needed?

        self.set_up_other_variables(data, *args, **kwargs)

        if bool_datapoints_valid is None:
            bool_datapoints_valid = pd.DataFrame(
                np.full_like(data, True, dtype=np.bool),
                columns=data.columns,
                index=data.index
            )

        # Start at beginning of year
        self.date_reference_seasonals_year = np.datetime64(datetime(
            year=data.index[0].year,
            month=1,
            day=1,
            tzinfo=data.index.tzinfo
        ))

        stds = data[self.columns_needed].std()
        stds_zero = stds[stds == 0]
        if len(stds_zero) != 0:
            raise ValueError(f'Zeros standard deviations for columns: {list(stds_zero.keys())}')

        # is for each datapoint (also history), but not unknown feature.
        # hence, y_sample also has beginning steps not belonging to y_fit.
        x_in_sample, y_in_sample, datetimes_x_in_sample = self.make_x_y(data, omit_unknown_future_truth=True)

        self.timedelta_data = self.get_timedelta_data(datetimes_x_in_sample)

        if self.indices_steps_day_include is None:
            indices_include_ = [
                [
                    list(range(self.n_steps_day))
                    for group_days in self.seasons_days_weeks
                ]
                for idx_season_year in range(self.n_seasons_annual)
            ]

            self.indices_steps_day_include = [
                [indices_include_ for idx_param in range(self.n_params_pdf)]
                for idx_timestep in range(self.n_steps_forecast_horizon)
            ]

        bool_x_valid, bool_y_valid, _ = self.make_x_y(bool_datapoints_valid, omit_unknown_future_truth=True)
        # for fitting
        indices_y_uses_valid_history_fit = self.get_indices_datapoints_history_valid(bool_x_valid, True)

        y_fit = self.make_y_fit(y_in_sample)

        # set scaling params for y.
        # don't do this if overwriting is not desired (e.g. after loading from file) and scaling params
        #   are already set.
        if overwrite_params_scaling or self.std_scaling_y is None:
            self.set_params_scaling_y(y_fit, indices_y_uses_valid_history_fit)
            self.set_params_scaling_x(x_in_sample, datetimes_x_in_sample, y_fit, indices_y_uses_valid_history_fit)

        self.set_indices_columns_variables_design_matrices_base()

        return x_in_sample, y_fit, datetimes_x_in_sample, indices_y_uses_valid_history_fit

    def set_indices_columns_variables_design_matrices_base(self):
        # traverse logic of design matrix creation
        result = []
        for idx_timestep in range(self.n_steps_forecast_horizon):
            result_timestep = []
            for idx_param_pdf in range(self.n_params_pdf):
                result_param_pdf = {}
                idx_column_current = 0
                for name_column in self.columns_needed:
                    lags_column_thresholds = self.lags_columns[name_column][idx_timestep][idx_param_pdf]
                    indices_column = []
                    for lags_column_threshold in lags_column_thresholds:
                        for lag in lags_column_threshold:
                            if lag >= -idx_timestep:
                                if name_column == self.variable_target:
                                    continue
                                else:
                                    raise ValueError(f'lag column {name_column}: {lag} for idx_timestep {idx_timestep} '
                                                     f'would need unknown data.')
                            indices_column.append(idx_column_current)
                            idx_column_current += 1
                    result_param_pdf[name_column] = indices_column
                result_timestep.append(result_param_pdf)
            result.append(result_timestep)
        self.indices_columns_variables_design_matrices_base = result

    def fit(self, data: pd.DataFrame, *args, warm_start=False, overwrite_params_scaling=True, params_pdf_fit=None,
            bool_datapoints_valid=None, **kwargs):
        """

        :param data:
        :param warm_start: Hopefully enhance performance in forecast study. But should be
            used with care due to possibly inconsistent feature selection of lasso.
        :return:
        """
        self.logger.info('Setup with data')
        x, y_fit, datetimes_x, indices_y_fit_valid_data = self.do_setup(data,
                                                                        overwrite_params_scaling=overwrite_params_scaling,
                                                                        bool_datapoints_valid=bool_datapoints_valid)

        self.logger.info('Starting fit process')
        return self.fit_after_setup(x, datetimes_x, y_fit, indices_y_fit_valid_data, warm_start, params_pdf_fit, *args,
                                    **kwargs)

    def fit_after_setup(self, x, datetimes_x, y_fit, indices_y_fit_valid_data, warm_start, params_pdf_fit, *args,
                        **kwargs):
        raise NotImplementedError

    def predict_get_truth(self, data: pd.DataFrame,
                          omit_unknown_future_truth: bool, n_samples: int or None, bool_datapoints_valid=None, *args,
                          **kwargs):
        """ Perform prediction for data and obtain corresponding truth values

        :param data: Dataframe with time series data
        :return: Prediction, x, truth
        """
        x, y, datetimes_x = self.make_x_y(data, omit_unknown_future_truth=omit_unknown_future_truth)
        x_prediction, truth = self.slice_x_y_sample_predictions(x, y)
        datetimes_x_prediction = self.slice_tensor_sample_predictions(datetimes_x)
        prediction = self.predict(data, omit_unknown_future_truth, n_samples, *args, **kwargs)

        if bool_datapoints_valid is not None:
            bool_x_valid, y_for_valid, datetimes_x = self.make_x_y(bool_datapoints_valid,
                                                                   omit_unknown_future_truth=omit_unknown_future_truth)
            indices_valid = self.get_indices_datapoints_history_valid(bool_x_valid, True)
            self.logger.info(f'Omitted {truth.shape[0] - indices_valid.shape[0]} predictions with invalid datapoints')

            prediction = tf.gather(prediction, indices_valid, axis=1)
            x_prediction = tf.gather(x_prediction, indices_valid)
            truth = tf.gather(truth, indices_valid)
            datetimes_x_prediction = datetimes_x_prediction[indices_valid]

        return prediction, x_prediction, truth, datetimes_x_prediction

    def predict_x_scaled(self, x, datetimes_x, n_samples, *args, **kwargs):
        """Returns [sample, datapoint, timestep_predict]"""
        raise NotImplementedError

    def predict_rescale_x(self, x, datetimes_x, n_samples, *args, **kwargs):
        prediction_scaled = self.predict_x_scaled(x, datetimes_x, n_samples, *args, **kwargs)
        prediction = self.rescale_y(prediction_scaled)
        # prediction = prediction_scaled * self.std_scaling_y + self.mean_scaling_y

        return prediction

    def predict(self, data: pd.DataFrame, omit_unknown_future_truth: bool, n_samples, *args, **kwargs):
        """ Make prediction from data.

        :param data: Dataframe with time series data
        :param omit_unknown_future_truth: If true, last day (where no y is available) is not predicted
        :return: [sample, datapoint, timestep_predict]
        """
        x, _, datetimes_x = self.make_x_y(data, omit_unknown_future_truth=omit_unknown_future_truth)
        result = self.predict_rescale_x(x, datetimes_x, n_samples, *args, **kwargs)
        return result

    def make_indices_subsets(self, x, x_rain=None):
        window_last_rain_rise_fall = 10
        window_last_rain_dry = 20
        # ensure that for dry period, no rain is within forecast horizon,
        #   so that also the forecast is not in rain
        window_rain_future_dry = self.n_steps_forecast_horizon
        window_max_future_flow = 10
        window_min_future_flow = 10
        window_min_past_flow = 10
        window_max_past_flow = 10

        threshold_rain = 500

        data_target = pd.Series(x[:, self.indices_columns_x[self.variable_target]])
        variables_rain = [col for col in self.columns_needed if 'mm/min' in col]

        if x_rain is None:
            if len(variables_rain) != 1:
                raise ValueError(f'not unique rain variable: {variables_rain}')

            data_rain = pd.Series(x[:, self.indices_columns_x[variables_rain[0]]])
        else:
            data_rain = pd.Series(x_rain)

        # rolling max of rain
        max_rain_past_rise_fall = data_rain.rolling(
            window_last_rain_rise_fall,
        ).max()

        max_rain_past_dry = data_rain.rolling(
            window_last_rain_dry,
        ).max()

        # is_falling = np.roll(venturi_median, 1) > venturi_median
        is_falling = np.roll(data_target, 1) > data_target

        # workaround as pandas does not support forward rolling window.
        max_future_flow = data_target
        for shift in range(window_max_future_flow):
            max_future_flow = np.maximum(
                max_future_flow,
                np.roll(data_target, -shift)
            )

        min_future_flow = data_target
        for shift in range(window_min_future_flow):
            min_future_flow = np.minimum(
                min_future_flow,
                np.roll(data_target, -shift)
            )

        max_rain_future_dry = data_rain
        for shift in range(window_rain_future_dry):
            max_rain_future_dry = np.maximum(
                max_rain_future_dry,
                np.roll(data_rain, -shift)
            )

        min_past_flow = data_target.rolling(
            window_min_past_flow,
        ).min()

        max_past_flow = data_target.rolling(
            window_max_past_flow,
        ).max()

        bool_is_raise = np.logical_and(
            np.logical_and(
                max_rain_past_rise_fall > 0.0001,
                max_future_flow > threshold_rain
            ),
            min_past_flow < threshold_rain
        )

        bool_is_fall = np.logical_and(
            np.logical_and(max_past_flow > threshold_rain, is_falling),
            min_future_flow < threshold_rain
        )

        bool_constant = np.logical_and(
            data_target > threshold_rain,
            np.logical_not(np.logical_or(bool_is_raise, bool_is_fall))
        )

        # datapoint considered dry must be at least -min_lag after last fall datapoint,
        #   so that fall data is not in lags used for dry prediction.
        fall_not_in_data = np.logical_not(bool_is_fall)
        for shift in range(np.abs(self.min_lag_wrt_first_step_prediction)):
            fall_not_in_data = np.logical_and(
                fall_not_in_data,
                np.roll(np.logical_not(bool_is_fall), shift)
            )

        bool_not_other = np.logical_and(
            fall_not_in_data,
            np.logical_and(
                np.logical_not(bool_constant),
                np.logical_not(bool_is_raise),
            )
        )
        bool_dry = np.logical_and(
            bool_not_other,
            np.logical_and(
                np.logical_not(max_rain_past_dry > 0.0001),
                np.logical_not(max_rain_future_dry > 0.0001)
            )
        )

        sets_indices = {
            'all': np.full_like(bool_is_raise, True, dtype=np.bool),
            'raise': bool_is_raise,
            'constant': bool_constant,
            'decrease': bool_is_fall,
            'dry': bool_dry
        }

        return sets_indices

    def get_scores_subsets(self, x, truth, predictions_1, predictions_2, indices_valid, include_intraday=True,
                           include_timesteps=True, x_rain=None):
        self.logger.warning('get_scores_subsets(): verify that indices invalid work as desired!!!')

        bool_valid = np.zeros(truth.shape[0], dtype=bool)
        bool_valid[indices_valid] = True

        result = {}
        indices_subsets = self.make_indices_subsets(x, x_rain=x_rain)
        for name_subset, bool_subset in indices_subsets.items():
            bool_subset_valid = np.logical_and(bool_valid, bool_subset)
            truth_subset = np.asarray(truth)[bool_subset_valid]

            # predictions have first axis as sample
            predictions_subset_1 = np.asarray(predictions_1)[..., bool_subset_valid, :]
            predictions_subset_2 = np.asarray(predictions_2)[..., bool_subset_valid, :]
            result[name_subset] = self.get_scores(truth_subset, predictions_subset_1, predictions_subset_2,
                                                  include_intraday, include_timesteps)

        return result, indices_subsets

    def get_scores(self, truth, predictions_1, predictions_2, include_intraday, include_timesteps):
        """
            Remark: Predictions must be a list of samples.
        """
        result = {
            SCORES_SCALAR: self.get_scores_scalar(truth, predictions_1, predictions_2),
        }

        if include_intraday:
            result[SCORES_INTRADAY] = self.get_scores_datapoints(truth, predictions_1, predictions_2)
        if include_timesteps:
            result[SCORES_TIMESTEPS] = self.get_scores_timesteps(truth, predictions_1, predictions_2)
        return result

    def get_scores_scalar(self, truth, predictions_1, predictions_2):
        predictions_merged = tf.concat([predictions_1, predictions_2], axis=0)
        return {
            'rmse': scores.calc_rmse(truth, predictions_merged),
            'rmse_mean_forecast_horizons': scores.calc_rmse_mean_datapoints(truth, predictions_merged),
            'mae': scores.calc_mae(truth, predictions_merged),
            'rmse_vector': scores.calc_rmse_vector(truth, predictions_merged),
            'rmse_relative': scores.calc_rmse(truth, predictions_merged, relative=True),
            'rmse_mean_forecast_horizons_relative': scores.calc_rmse_mean_datapoints(truth, predictions_merged,
                                                                                     relative=True),
            'mae_relative': scores.calc_mae(truth, predictions_merged, relative=True),
            'rmse_vector_relative': scores.calc_rmse_vector(truth, predictions_merged, relative=True),
            'energy': scores.calc_energy(truth, predictions_1, predictions_2)
        }

    def get_scores_datapoints(self, truth, predictions_1, predictions_2):
        predictions_merged = tf.concat([predictions_1, predictions_2], axis=0)
        # take first sample (as point forecast has only one) from predictions
        return {
            'mae_intraday': scores.calc_mae_datapoints(truth, predictions_merged),
            'rmse_intraday': scores.calc_rmse_datapoints(truth, predictions_merged),
            'mae_intraday_relative': scores.calc_mae_datapoints(truth, predictions_merged, relative=True),
            'rmse_intraday_relative': scores.calc_rmse_datapoints(truth, predictions_merged, relative=True),
            'energy_intraday': scores.calc_energy_datapoints(truth, predictions_1, predictions_2),
        }

    def get_scores_timesteps(self, truth, predictions_1, predictions_2):
        predictions_merged = tf.concat([predictions_1, predictions_2], axis=0)
        # take first sample (as point forecast has only one) from predictions
        return {
            'mae_timesteps': scores.calc_mae_timesteps(truth, predictions_merged),
            'rmse_timesteps': scores.calc_rmse_timesteps(truth, predictions_merged),
            'mae_timesteps_relative': scores.calc_mae_timesteps(truth, predictions_merged, relative=True),
            'rmse_timesteps_relative': scores.calc_rmse_timesteps(truth, predictions_merged, relative=True),
            'energy_timesteps': scores.calc_energy_timesteps(truth, predictions_1, predictions_2),
            'energy_timesteps_relative': scores.calc_energy_timesteps(truth, predictions_1, predictions_2),
        }

    def evaluate_model(self, data_train,
                       data_test,
                       column_rain,
                       bool_datapoints_valid_train=None,
                       bool_datapoints_valid_test=None,
                       include_intraday=True,
                       include_timesteps=True,
                       n_samples_predict=100,
                       include_criteria=True
                       ):
        if bool_datapoints_valid_train is None:
            bool_datapoints_valid_train = pd.DataFrame(
                np.full_like(data_train, True, dtype=np.bool),
                columns=data_train.columns,
                index=data_train.index
            )
        if bool_datapoints_valid_test is None:
            bool_datapoints_valid_test = pd.DataFrame(
                np.full_like(data_test, True, dtype=np.bool),
                columns=data_test.columns,
                index=data_test.index
            )

        bool_x_valid_train, bool_y_valid_test, _ = self.make_x_y(bool_datapoints_valid_train,
                                                                 omit_unknown_future_truth=True)
        indices_prediction_uses_valid_history_train = self.get_indices_datapoints_history_valid(bool_x_valid_train,
                                                                                                True)

        bool_x_valid_test, bool_y_valid_test, _ = self.make_x_y(bool_datapoints_valid_test,
                                                                omit_unknown_future_truth=True)
        indices_prediction_uses_valid_history_test = self.get_indices_datapoints_history_valid(bool_x_valid_test, True)

        n_samples_score = int(n_samples_predict / 2) if n_samples_predict is not None else None

        self.logger.info('Evaluation: Sampling training')
        predictions_train, x_train, truth_train, datetimes_x_train = self.predict_get_truth(
            data_train,
            True,
            n_samples_predict,
        )

        # if rain is no regressor, it is not present in x.
        #   hence, construct rain for x from data.
        x_rain_train = np.asarray(
            data_train[column_rain],
            dtype=self.DTYPE
        )[:-self.n_steps_forecast_horizon]
        x_rain_train_predictions = self.slice_tensor_sample_predictions(x_rain_train)

        self.logger.info('Evaluation: Scoring Training')
        scores_train, indices_subsets_train = self.get_scores_subsets(
            x_train,
            truth_train,
            predictions_train[:n_samples_score] if n_samples_predict is not None else predictions_train,
            predictions_train[n_samples_score:] if n_samples_predict is not None else predictions_train,
            indices_prediction_uses_valid_history_train,
            include_intraday=include_intraday,
            include_timesteps=include_timesteps,
            x_rain=x_rain_train_predictions
        )

        self.logger.info('Evaluation: Sampling testing')
        predictions_test, x_test, truth_test, datetimes_x_test = self.predict_get_truth(
            data_test,
            True,
            n_samples_predict,
        )

        # # if rain is no regressor, it is not present in x.
        # #   hence, construct rain for x from data.
        x_rain_test = np.asarray(
            data_test[column_rain],
            dtype=self.DTYPE
        )[:-self.n_steps_forecast_horizon]
        x_rain_test_predictions = self.slice_tensor_sample_predictions(x_rain_test)

        self.logger.info('Evaluation: Scoring testing')
        scores_test, indices_subsets_test = self.get_scores_subsets(
            x_test,
            truth_test,
            predictions_test[:n_samples_score] if n_samples_predict is not None else predictions_test,
            predictions_test[n_samples_score:] if n_samples_predict is not None else predictions_test,
            indices_prediction_uses_valid_history_test,
            x_rain=x_rain_test_predictions,
            include_intraday=include_intraday,
            include_timesteps=include_timesteps,
        )

        criteria = None
        if include_criteria:
            try:
                criteria = self.get_criteria(
                    data_train,
                    bool_datapoints_valid_train
                )
            except NotImplementedError as e:
                self.logger.warning(f'No criterion implemented for model {type(self).__name__}')

        return (
            scores_train,
            scores_test,
            criteria,
            predictions_train,
            predictions_test,
            truth_train,
            truth_test,
            indices_prediction_uses_valid_history_train,
            indices_prediction_uses_valid_history_test,
            indices_subsets_train,
            indices_subsets_test,
            datetimes_x_train,
            datetimes_x_test,
            x_train,
            x_test
        )

    def get_criteria(self, data, bool_datapoints_valid):
        self.logger.warning(f'get_criteria(): NEED TO SUPPORT NANS INVALID!')

        return {
            CRITERIA_BIC: self.get_bic(data, bool_datapoints_valid),
            CRITERIA_AIC: self.get_aic(data, bool_datapoints_valid),
        }

    def get_bic(self, data, bool_datapoints_valid):
        x, y, datetimes_x = self.make_x_y(data, omit_unknown_future_truth=True)
        log_likelihood = self.get_log_likelihood(data, bool_datapoints_valid)
        n_params = self.n_params_fitted_active
        n_observations = y.shape[0]

        return n_params * np.log(n_observations) - 2 * log_likelihood

    def get_aic(self, data, bool_datapoints_valid):
        log_likelihood = self.get_log_likelihood(data, bool_datapoints_valid)
        n_params = self.n_params_fitted_active

        return 2 * n_params - 2 * log_likelihood

    def get_log_likelihood(self, x, y):
        raise NotImplementedError

    def check_significance(self, truth, predictions_baseline, predictions_evaluate_if_better):
        raise NotImplementedError

    @classmethod
    def make_dense_lags(
            cls,
            # for pairs, convention is that frist var is fixed and interacting var is lagged.
            min_lags_variables,
            max_lags_variables,
            n_thresholds_variables,
            n_steps_forecast_horizon,
            # variables where lags that lie in the future (w.r.t. last datapoint) are allowed.
            # used for oracles as well as () the target variable.
            variable_target,
            variables_oracles,
            # if predictions of target shall be used in design matrices
            # for idx_timestep > 0.
            # If true, lags with 0 > lag are included for target variable.
            #   else, only lag < -idx_timestep.
            include_predictions_target,
            # If True, lag -n for timestep m will translate to -(n+m).
            #   hence, minimum lags for respective timestep are interpreted as relative to the
            #   first prediction step instead of actual timestep.
            min_lags_relative_to_first_step_predict,
            params_pdf_no_lags=None,
            lags_target_additional=None,
            furain_all_lags=False
    ):
        if params_pdf_no_lags is None:
            params_pdf_no_lags = []

        lags_variables = {}
        for name_variable, min_lag_variable in min_lags_variables.items():
            if n_thresholds_variables[name_variable] == 0:
                continue
            if isinstance(name_variable, (list, tuple)):
                # interactions are built separately in make_lags_thresholds_interactions()
                continue

            max_lag_variable = max_lags_variables[name_variable]
            lags_variable = []

            for idx_timestep_predict in range(n_steps_forecast_horizon):
                lags_variable_timestep = []
                for idx_param_pdf in range(cls.n_params_pdf):
                    lags_variable_param_pdf = []
                    if idx_param_pdf not in params_pdf_no_lags:
                        for idx_threshold in range(n_thresholds_variables[name_variable]):
                            if name_variable in variables_oracles:
                                lags_variable_threshold = [-(idx_timestep_predict + 1)]
                            else:
                                lags_variable_threshold = []
                                if min_lags_relative_to_first_step_predict:
                                    min_lag_variable_use = min_lag_variable - idx_timestep_predict
                                else:
                                    min_lag_variable_use = min_lag_variable

                                lags_use = list(range(min_lag_variable_use, max_lag_variable + 1))
                                if name_variable == variable_target and lags_target_additional is not None:
                                    lags_use.extend(lags_target_additional)
                                for lag in lags_use:
                                    # only include valid lags.
                                    if (
                                            lag <= -(idx_timestep_predict + 1)
                                            or (
                                            name_variable == variable_target
                                            and include_predictions_target
                                            and lag < 0
                                    )
                                    ):
                                        lags_variable_threshold.append(lag)
                            lags_variable_param_pdf.append(lags_variable_threshold)

                    lags_variable_timestep.append(lags_variable_param_pdf)
                lags_variable.append(lags_variable_timestep)

            lags_variables[name_variable] = lags_variable

        return lags_variables

    @staticmethod
    def check_variable_is_rain(name_variable):
        return 'ORACLE' in name_variable or 'mm/min' in name_variable or 'RAIN' in name_variable or 'radar' in name_variable

    @classmethod
    def get_bounds_thresholds_variable(cls, data_variable, has_rain, is_interaction=False):
        if has_rain:
            data_not_zero = data_variable[data_variable > 0]

            if is_interaction:
                bound_lower = np.nanpercentile(data_not_zero, 0)

                # for rain, spread thresholds over values where rain is not zero.
                bound_upper = np.nanpercentile(
                    data_not_zero,
                    95
                )
            else:
                bound_lower = np.nanpercentile(data_not_zero, 0)

                # for rain, spread thresholds over values where rain is not zero.
                bound_upper = np.nanpercentile(
                    data_not_zero,
                    95
                )
        else:
            bound_lower = np.nanpercentile(data_variable, 0)

            bound_upper = np.nanpercentile(data_variable, 99)

        return bound_lower, bound_upper

    @classmethod
    def make_thresholds_variables(cls, data, n_thresholds_variables, fn_transform_thresholds, n_steps_forecast_horizon,
                                  params_pdf_no_lags=None):
        if params_pdf_no_lags is None:
            params_pdf_no_lags = []
        thresholds_variables = {}

        for name_variable, n_thresholds_variable in n_thresholds_variables.items():
            if n_thresholds_variable == 0:
                continue
            if isinstance(name_variable, (list, tuple)):
                # interactions are built separately in make_lags_thresholds_interactions()
                continue

            # get min and max quantiles of data
            # First (min) is omitted because this is covered by -inf inserted later.
            has_rain = cls.check_variable_is_rain(name_variable)
            data_variable = data[name_variable]
            bounds = cls.get_bounds_thresholds_variable(
                data_variable, has_rain
            )

            thresholds_all = []
            for idx_timestep in range(n_steps_forecast_horizon):
                thresholds_timestep = []
                for idx_param_pdf in range(cls.n_params_pdf):
                    # if the variable is interaction, individual thresholds are built for each lag.
                    thresholds = cls.make_thresholds_from_bounds(data_variable, *bounds, n_thresholds_variable,
                                                                 fn_transform_thresholds, has_rain)
                    thresholds_timestep.append(
                        thresholds
                        if idx_param_pdf not in params_pdf_no_lags else []
                    )
                thresholds_all.append(thresholds_timestep)

            thresholds_variables[name_variable] = thresholds_all
        return thresholds_variables

    @classmethod
    def make_thresholds_from_bounds(cls, data, bound_lower, bound_upper, n_thresholds_variable, fn_transform_thresholds,
                                    has_rain):
        steps_relative = np.linspace(0, 1, n_thresholds_variable, endpoint=False)[1:]
        steps_relative = eval(fn_transform_thresholds, {
            'np': np,
            'x': steps_relative
        })

        thresholds = bound_lower + (bound_upper - bound_lower) * steps_relative
        thresholds = np.insert(thresholds, thresholds.shape[0], np.inf)

        return thresholds.tolist()

    @classmethod
    def make_lags_thresholds_interactions(
            cls,
            data,
            variables_oracles,
            variable_target,
            include_predictions_target,
            min_lags_variables,
            max_lags_variables,
            n_thresholds_variables,
            fn_transform_thresholds,
            n_steps_forecast_horizon,
            min_lags_relative_to_first_step_predict,
    ):

        result_lags, result_thresholds = {}, {}

        for name_variable, n_thresholds_variable in n_thresholds_variables.items():
            if n_thresholds_variable == 0:
                continue
            if not isinstance(name_variable, (list, tuple)):
                continue

            min_lag_variable = min_lags_variables[name_variable]
            max_lag_variable = max_lags_variables[name_variable]
            thresholds_var, lags_var = [], []
            for idx_timestep in range(n_steps_forecast_horizon):
                thresholds_timestep, lags_timestep = [], []
                for idx_param_pdf in range(cls.n_params_pdf):
                    # make individual threshold sets for all lags, as different data ranges occur for different
                    #   combinations of lagged vars', multiplied, show different data ranges.

                    lag_first_var_last_known = -1 if name_variable[
                                                         0] == variable_target and include_predictions_target else -(
                            idx_timestep + 1)

                    # first, get the basic lags for the variable combination
                    if name_variable[1] in variables_oracles:

                        lags_base = [(lag_first_var_last_known, -(idx_timestep + 1))]
                    else:
                        lags_base = []
                        if min_lags_relative_to_first_step_predict:
                            min_lag_variable_use = min_lag_variable - idx_timestep
                        else:
                            min_lag_variable_use = min_lag_variable

                        lags_use = list(range(min_lag_variable_use, max_lag_variable + 1))
                        for lag in lags_use:
                            # only include valid lags.
                            if (lag <= -(idx_timestep + 1)):
                                # for interaction pairs, convention is that frist var is fixed and interacting var is lagged.
                                lags_base.append(
                                    (lag_first_var_last_known, lag)
                                )

                    # then, for each lag, make a separate threshold with only this lag assigned.
                    lags_thresholds_param_pdf = []
                    thresholds_param_pdf = []
                    for lag_base in lags_base:
                        a_lagged = cls.lag_data(data[name_variable[0]], lag_base[0], idx_timestep)
                        b_lagged = cls.lag_data(data[name_variable[1]], lag_base[1], idx_timestep)

                        interaction = a_lagged * b_lagged

                        has_rain = cls.check_variable_is_rain(name_variable[0]) or cls.check_variable_is_rain(
                            name_variable[1])
                        bounds = cls.get_bounds_thresholds_variable(
                            interaction,
                            has_rain,
                            True
                        )
                        thresholds_lag = cls.make_thresholds_from_bounds(interaction, *bounds, n_thresholds_variable,
                                                                         fn_transform_thresholds, has_rain)
                        for idx_threshold in range(len(thresholds_lag)):
                            thresholds_param_pdf.append(thresholds_lag[idx_threshold])
                            lags_thresholds_param_pdf.append([lag_base])

                    lags_timestep.append(lags_thresholds_param_pdf)
                    thresholds_timestep.append(thresholds_param_pdf)
                thresholds_var.append(thresholds_timestep)
                lags_var.append(lags_timestep)
            result_lags[name_variable] = lags_var
            result_thresholds[name_variable] = thresholds_var

        return result_lags, result_thresholds
