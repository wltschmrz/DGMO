import numpy as np

def get_mean_sdr_from_dict(sdris_dict):
    mean_sdr = np.nanmean(list(sdris_dict.values()))
    return mean_sdr

def calculate_sdr(ref: np.ndarray, est: np.ndarray, eps=1e-10) -> float:
    r"""Calculate SDR between reference and estimation.
    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """
    reference = ref
    noise = est - reference
    numerator = np.clip(a=np.mean(reference ** 2), a_min=eps, a_max=None)
    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)
    sdr = 10. * np.log10(numerator / denominator)
    return sdr

def calculate_sisdr(ref, est):
    r"""Calculate SDR between reference and estimation.
    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """
    eps = np.finfo(ref.dtype).eps
    reference = ref.copy()
    estimate = est.copy()
    reference = reference.reshape(reference.size, 1)
    estimate = estimate.reshape(estimate.size, 1)
    Rss = np.dot(reference.T, reference)
    # get the scaling factor for clean sources
    a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)
    e_true = a * reference
    e_res = estimate - e_true
    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()
    sisdr = 10 * np.log10((eps+ Sss)/(eps + Snn))
    return sisdr 