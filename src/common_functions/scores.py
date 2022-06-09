import numpy as np
from tqdm import tqdm
import tensorflow as tf


# prediction and truth are arrays with [d,s]

def mean_prediction(prediction):
    """ Prediction has [sample, datapoint, timestep].
        Calc sample mean for each datapoint and timestep.

    """
    return np.mean(prediction, axis=0)


def calc_errors(truth, prediction, relative):
    errors = truth - mean_prediction(prediction)
    if relative:
        errors = errors / truth

    return errors


def calc_rmse(truth: np.ndarray, prediction: np.ndarray, relative=False):
    errors = calc_errors(truth, prediction, relative)
    result = np.sqrt(np.mean(errors ** 2))

    return result


def calc_mae(truth: np.ndarray, prediction: np.ndarray, relative=False):
    return np.mean(calc_mae_datapoints(truth, prediction, relative=relative))


def calc_rmse_vector(truth: np.ndarray, prediction: np.ndarray, relative=False):
    errors = calc_errors(truth, prediction, relative)
    # vector sum over each say
    l2_days = np.sum(errors ** 2, axis=1)
    sum_root_square_errors = np.sum(np.sqrt(l2_days))
    result = sum_root_square_errors / errors.size

    return result


def calc_rmse_mean_datapoints(truth: np.ndarray, prediction: np.ndarray, relative=False):
    rmse_intraday = calc_rmse_datapoints(truth, prediction, relative)
    result = np.mean(rmse_intraday, axis=-1)

    return result


def calc_mae_vector(truth: np.ndarray, prediction: np.ndarray, relative=False):
    return calc_mae(truth, prediction, relative=relative)


def calc_rmse_datapoints(truth: np.ndarray, prediction: np.ndarray, relative=False):
    errors = calc_errors(truth, prediction, relative)
    result = np.sqrt(np.mean(errors ** 2, axis=-1))

    return result


def calc_rmse_timesteps(truth: np.ndarray, prediction: np.ndarray, relative=False):
    errors = calc_errors(truth, prediction, relative)
    result = np.sqrt(np.mean(errors ** 2, axis=0))

    return result


def calc_mae_datapoints(truth: np.ndarray, prediction: np.ndarray, relative=False):
    errors = calc_errors(truth, prediction, relative)
    result = np.mean(np.abs(errors), axis=-1)

    return result


def calc_mae_timesteps(truth: np.ndarray, prediction: np.ndarray, relative=False):
    errors = calc_errors(truth, prediction, relative)
    result = np.mean(np.abs(errors), axis=0)

    return result


def calc_energy(
        truth: np.ndarray,
        # [sample,datapoint,timestep_predict]
        predictions_1: np.ndarray,
        # [datapoint, timestep_predict]
        predictions_2: np.ndarray,
        # second independent draw, shape as above
):
    return np.mean(calc_energy_datapoints(truth, predictions_1, predictions_2))


def calc_l2_beta(diff_samples, beta):
    sq_diff_samples = tf.pow(diff_samples, 2)

    # along time axis
    sum_squares = tf.reduce_sum(sq_diff_samples, axis=-1)

    # sqrt does not work for differentiation. Calc final exponent directly
    #   and consider that only betal leading to even exponent are possible for gradient calculation.
    exp_final = 0.5 * beta

    result = tf.pow(sum_squares, exp_final)

    return result


def calc_energy_datapoints(
        truth: np.ndarray,
        # [sample,datapoint,timestep_predict]
        predictions_1: np.ndarray,
        # [datapoint, timestep_predict]
        predictions_2: np.ndarray,
        beta=1
):
    samples_all = np.concatenate([predictions_1, predictions_2], axis=0)

    # samples are [sample, section, timestep]
    n_samples = samples_all.shape[0]

    diff_samples_y = samples_all - truth[tf.newaxis, ...]
    l2_diff_samples = calc_l2_beta(diff_samples_y, beta)

    # mean l2 over samples
    ed_datapoints = tf.reduce_mean(l2_diff_samples, axis=0)

    ### Brute Force R implementation
    ### All samples with all samples

    sum_l2_diff_samples = tf.zeros_like(samples_all[0, :, 0])
    for idx_sample_1 in range(n_samples):
        for idx_sample_2 in range(n_samples):
            l2_diff_samples_1_2 = calc_l2_beta(
                samples_all[idx_sample_1] - samples_all[idx_sample_2],
                beta
            )
            sum_l2_diff_samples += l2_diff_samples_1_2

    ei_datapoints = sum_l2_diff_samples / n_samples ** 2

    energy_datapoints = ed_datapoints - 0.5 * ei_datapoints

    return np.asarray(energy_datapoints)


def calc_energy_datapoints_k_band(
        truth: np.ndarray,
        # [sample,datapoint,timestep_predict]
        predictions_1: np.ndarray,
        # [datapoint, timestep_predict]
        predictions_2: np.ndarray,
        # second independent draw, shape as above
        K=None,
        beta=1
):
    samples_all = np.concatenate([predictions_1, predictions_2], axis=0)
    n_samples = samples_all.shape[0]

    if K is None:
        # K = M
        K = n_samples

    ### score for one sample
    # result vector of a sample and datapoint is vector of all timesteps
    #   sampled for that datapoint

    # l2 differences for all result vectors and all samples
    differences_squared = np.power(
        samples_all - truth[np.newaxis, :, :],
        2
    )

    # l2 along the prediction horizon axis,
    #   so that result is l2 norms for all samples and datapoints
    l2_differences = np.sqrt(np.sum(
        differences_squared,
        axis=2
    ))

    # mean over all samples. result is [datapoint]
    first_term = np.mean(
        l2_differences ** beta,
        axis=0
    )

    # todo: more elegant implementation
    sum_second_term = np.zeros((truth.shape[0]))
    for j in tqdm(range(n_samples)):
        for k in range(j, K):
            k_use = k if k < n_samples else k - n_samples
            sum_second_term += np.sqrt(
                np.sum(
                    np.power(
                        samples_all[j] - samples_all[k_use],
                        2
                    )
                )
            ) ** beta
    second_term = sum_second_term / n_samples

    result_datapoints = first_term - second_term / 2.

    return result_datapoints


def calc_energy_timesteps(
        truth: np.ndarray,
        # [sample,datapoint,timestep_predict]
        predictions_1: np.ndarray,
        # [datapoint, timestep_predict]
        predictions_2: np.ndarray,
        beta=1
):
    samples_all = np.concatenate([predictions_1, predictions_2], axis=0)

    # samples are [sample, section, timestep]
    n_samples = samples_all.shape[0]

    diff_samples_y = samples_all - truth[tf.newaxis, ...]
    l2_diff_samples = tf.abs(diff_samples_y) ** beta

    # mean l2 over samples
    ed_datapoints = tf.reduce_mean(l2_diff_samples, axis=0)

    ### Brute Force R implementation
    ### All samples with all samples

    sum_l2_diff_samples = tf.zeros_like(samples_all[0])
    for idx_sample_1 in range(n_samples):
        for idx_sample_2 in range(n_samples):
            l2_diff_samples_1_2 = tf.abs(
                samples_all[idx_sample_1] - samples_all[idx_sample_2],
            ) ** beta
            sum_l2_diff_samples += l2_diff_samples_1_2

    ei_datapoints = sum_l2_diff_samples / n_samples ** 2

    energy_datapoints = ed_datapoints - 0.5 * ei_datapoints

    result = tf.reduce_mean(energy_datapoints, axis=0)

    return np.asarray(result)
