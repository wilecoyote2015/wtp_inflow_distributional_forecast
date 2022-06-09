from code.inflow_forecast.model.gamlss.model_gamlss_base import ModelGamlssBase

from tensorflow_probability import distributions as tfd
import tensorflow as tf

class ModelGamlssJsu(ModelGamlssBase):
    n_params_pdf = 4
    idx_param_pdf_loc = 2
    params_pdf_constant = [1]
    order_params_fit = [2,3,1,0]
    # TODO: check link functions
    link_functions_default = [
        # skew, tailweight, loc, scale
        lambda x: (tf.math.sigmoid(x) - 0.5)*3,
        lambda x: tf.math.sigmoid(x) * 3 + 0.05,
        lambda x: x,
        lambda x: tf.math.softplus(x*5.)/5.,  # factor for softness control
    ]

    @staticmethod
    def fn_make_distribution(*args, **kwargs):
        return tfd.JohnsonSU(*args, **kwargs)

class ModelGamlssGaussian(ModelGamlssBase):
    n_params_pdf = 2
    idx_param_pdf_loc = 0
    params_pdf_constant = []
    link_functions_default = [
        lambda x: x,
        lambda x: tf.math.softplus(x*5.)/5.,  # factor for softness control
    ]

    @staticmethod
    def fn_make_distribution(*args, **kwargs):
        return tfd.Normal(*args, **kwargs)