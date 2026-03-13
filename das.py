import torch
from torch.utils.checkpoint import checkpoint


def das(iqraw, tA, tB, fs, fd, A=None, B=None, apoA=1, apoB=1, interp="cubic"):
    """
    Delay-and-sum IQ data according to a given time delay profile.
    @param iqraw   [na, nb, nsamps]  Raw IQ data (baseband)
    @param tA      [na, *pixdims]    Time delays to apply to dimension 0 of iq
    @param tB      [nb, *pixdims]    Time delays to apply to dimension 1 of iq
    @param fs      scalar            Sampling frequency to convert from time to samples
    @param fd      scalar            Demodulation frequency (0 for RF modulated data)
    @param A       [*na_out, na]     Linear combination of dimension 0 of iqraw
    @param B       [*nb_out, nb]     Linear combination of dimension 1 of iqraw
    @param apoA    [na, *pixdims]    Broadcastable apodization on dimension 0 of iq
    @param apoB    [nb, *pixdims]    Broadcastable apodization on dimension 1 of iq
    @param interp  string            Interpolation method to use
    @return iqfoc  [*na_out, *nb_out, *pixel_dims]   Beamformed IQ data

    The tensors A and B specify how to combine the "elements" in dimensions 0 and 1 of
    iqraw via a tensor contraction. If A or B are None, they default to a vector of ones,
    i.e. a simple sum over all elements. If A or B are identity matrices, the result will
    be the delayed-but-not-summed output. A and B can be arbitrary tensors of arbitrary
    size, as long as the inner most dimension matches na or nb, respectively. Another
    alternative use case is for subaperture beamforming.

    Note that via acoustic reciprocity, it does not matter whether a or b correspond to
    the transmit or receive "elements".
    """
    # The default linear combination is to sum all elements.
    if A is None:
        A = torch.ones((iqraw.shape[0],), dtype=iqraw.dtype, device=iqraw.device)
    if B is None:
        B = torch.ones((iqraw.shape[1],), dtype=iqraw.dtype, device=iqraw.device)

    # Choose the interpolating function
    fints = {
        "nearest": interp_nearest,
        "linear": interp_linear,
        "cubic": interp_cubic,
        "lanczos3": lambda x, t: interp_lanczos(x, t, nlobe=3),
        "lanczos5": lambda x, t: interp_lanczos(x, t, nlobe=5),
    }
    fint = fints[interp]

    # Baseband interpolator
    def bbint(iq, t):
        iqfoc = fint(iq, fs * t)
        return iqfoc * torch.exp(2j * torch.pi * fd * t)

    # Delay-and-sum beamforming (loop outer, vmap inner)
    # This method does not vmap the outer loop and thus saves memory.
    def das_b(iq_i, tA_i):
        # vmap over the nb dimension (dim 0 of iq_i)
        results = torch.vmap(bbint)(iq_i, tA_i + tB) * apoB
        return torch.tensordot(B, results, dims=([-1], [0]))

    # Use checkpoint for memory efficiency (equivalent to jax.checkpoint)
    results = []
    for i in range(iqraw.shape[0]):
        result = checkpoint(das_b, iqraw[i], tA[i], use_reentrant=False)
        results.append(result)
    mapped_results = torch.stack(results) * apoA

    return torch.tensordot(A, mapped_results, dims=([-1], [0]))


def safe_access(x, s):
    """Safe access to array x at indices s.
    @param x: Array to access
    @param s: Indices to access at
    @return: Array of values at indices s
    """
    s = s.to(torch.int64)
    valid = (s >= 0) & (s < x.numel())
    safe_s = torch.clamp(s, 0, x.numel() - 1)
    return torch.where(valid, x[safe_s], torch.zeros_like(x[safe_s]))


def interp_nearest(x, si):
    """1D nearest neighbor interpolation with PyTorch.
    @param x: 1D array of values to interpolate
    @param si: Indices to interpolate at
    @return: Interpolated signal
    """
    return x[torch.clamp(torch.round(si), 0, x.shape[0] - 1).to(torch.int64)]


def interp_linear(x, si):
    """1D linear interpolation with PyTorch.
    @param x: 1D array of values to interpolate
    @param si: Indices to interpolate at
    @return: Interpolated signal
    """
    f = si - torch.floor(si)  # fractional part
    s = torch.floor(si)       # integer part
    x0 = safe_access(x, s + 0)
    x1 = safe_access(x, s + 1)
    return (1 - f) * x0 + f * x1


def interp_cubic(x, si):
    """1D cubic Hermite interpolation with PyTorch.
    @param x: 1D array of values to interpolate
    @param si: Indices to interpolate at
    @return: Interpolated signal
    """
    f = si - torch.floor(si)  # fractional part
    s = torch.floor(si)       # integer part
    # Values
    x0 = safe_access(x, s - 1)
    x1 = safe_access(x, s + 0)
    x2 = safe_access(x, s + 1)
    x3 = safe_access(x, s + 2)
    # Coefficients
    a0 = 0 + f * (-1 + f * (+2 * f - 1))
    a1 = 2 + f * (+0 + f * (-5 * f + 3))
    a2 = 0 + f * (+1 + f * (+4 * f - 3))
    a3 = 0 + f * (+0 + f * (-1 * f + 1))
    return (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3) / 2


def _lanczos_helper(x, nlobe=3):
    """Lanczos kernel"""
    a = (nlobe + 1) / 2
    return torch.where(torch.abs(x) < a, torch.sinc(x) * torch.sinc(x / a),
                       torch.zeros_like(x))


def interp_lanczos(x, si, nlobe=3):
    """Lanczos interpolation with PyTorch.
    @param x: 1D array of values to interpolate
    @param si: Indices to interpolate at
    @param nlobe: Number of lobes of the sinc function (e.g., 3 or 5)
    @return: Interpolated signal
    """
    f = si - torch.floor(si)  # fractional part
    s = torch.floor(si)       # integer part
    x0 = safe_access(x, s - 1)
    x1 = safe_access(x, s + 0)
    x2 = safe_access(x, s + 1)
    x3 = safe_access(x, s + 2)
    a0 = _lanczos_helper(f + 1, nlobe)
    a1 = _lanczos_helper(f + 0, nlobe)
    a2 = _lanczos_helper(f - 1, nlobe)
    a3 = _lanczos_helper(f - 2, nlobe)
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3
