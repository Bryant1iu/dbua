import numpy as np
import torch
from das import das


def lag_one_coherence(iq, t_tx, t_rx, fs, fd):
    """
    Lag-one coherence of the receive aperture (DOI: 10.1109/TUFFC.2018.2855653).
    The LOC measures the quality of a signal relative to its noise, and can be
    used to select acoustic output.
    """
    iq = iq.permute(1, 0, 2)  # Place rx aperture in 0-th index
    rxdata = das(iq, t_rx, t_tx, fs, fd, torch.eye(iq.shape[0], dtype=iq.dtype, device=iq.device))
    # Compute the correlation coefficient
    xy = torch.real(torch.nansum(rxdata[:-1] * torch.conj(rxdata[1:]), dim=0))
    xx = torch.nansum(torch.abs(rxdata[:-1]) ** 2, dim=0)
    yy = torch.nansum(torch.abs(rxdata[1:]) ** 2, dim=0)
    ncc = xy / torch.sqrt(xx * yy)
    return ncc


def coherence_factor(iq, t_tx, t_rx, fs, fd):
    """
    The coherence factor of the receive aperture (DOI: 10.1121/1.410562).
    The CF is a focusing criterion used to measure the amount of aberration in
    an image.
    """
    iq = iq.permute(1, 0, 2)  # Place rx aperture in 0-th index
    rxdata = das(iq, t_rx, t_tx, fs, fd, torch.eye(iq.shape[0], dtype=iq.dtype, device=iq.device))
    num = torch.abs(torch.nansum(rxdata, dim=0))
    den = torch.nansum(torch.abs(rxdata), dim=0)
    return num / den


def speckle_brightness(iq, t_tx, t_rx, fs, fd):
    """
    The speckle brightness criterion (DOI: 10.1121/1.397889)
    Speckle brightness can be used to measure the focusing quality.
    """
    return torch.nanmean(torch.abs(das(iq, t_tx, t_rx, fs, fd)))


def total_variation(c):
    """
    Total variation of sound speed map in x and z.
    The sound speed map c should be specified as a 2D matrix of size [nx, nz]
    """
    tvx = torch.nanmean(torch.square(torch.diff(c, dim=0)))
    tvz = torch.nanmean(torch.square(torch.diff(c, dim=1)))
    return tvx + tvz


def phase_error(iq, t_tx, t_rx, fs, fd, thresh=0.9):
    """
    The phase error between translating transmit and receive apertures.
    This error is closesly related to the "Translated Transmit Apertures" algorithm
    (DOI: 10.1109/58.585209), where translated transmit and receive apertures
    with common midpoint should have perfect speckle correlation by the van
    Cittert Zernike theorem (DOI: 10.1121/1.418235). High correlation will
    result in high-quality phase shift estimates (DOI: 10.1121/10.0000809).
    CUTE also takes a similar approach (DOI: 10.1016/j.ultras.2020.106168),
    but in the angular basis instead of the element basis.
    """
    # Compute the IQ data for given transmit and receive subapertures.
    nrx, ntx, nsamps = iq.shape
    mask = np.zeros((nrx, ntx))
    halfsa = 8  # Half of a subaperture
    dx = 1  # Subaperture increment
    for diag in range(-halfsa, halfsa + 1):
        mask = mask + np.diag(np.ones((ntx - abs(diag),)), diag)
    mask = mask[halfsa: mask.shape[0] - halfsa: dx]
    At = torch.tensor(mask[::-1].copy(), dtype=iq.dtype, device=iq.device)
    Ar = torch.tensor(mask.copy(), dtype=iq.dtype, device=iq.device)
    iqfoc = das(iq, t_tx, t_rx, fs, fd, At, Ar)

    # Now compute the correlation between neighboring pulse-echo signals with
    # common midpoints.
    xy = iqfoc[:-1, :-1] * torch.conj(iqfoc[+1:, +1:])
    xx = iqfoc[:-1, :-1] * torch.conj(iqfoc[:-1, :-1])
    yy = iqfoc[+1:, +1:] * torch.conj(iqfoc[+1:, +1:])
    # Remove correlations with only one signal
    valid1 = (iqfoc[:-1, :-1] != 0) & (iqfoc[1:, 1:] != 0)
    xy = torch.where(valid1, xy, torch.zeros_like(xy))
    xx = torch.where(valid1, xx, torch.zeros_like(xx))
    yy = torch.where(valid1, yy, torch.zeros_like(yy))
    # Determine where the correlation coefficient is high enough to use
    xy = torch.sum(xy, dim=-1)  # Sum over kernel
    xx = torch.sum(xx, dim=-1)  # Sum over kernel
    yy = torch.sum(yy, dim=-1)  # Sum over kernel
    ccsq = torch.square(torch.abs(xy)) / (torch.abs(xx) * torch.abs(yy))
    valid2 = ccsq > thresh * thresh
    xy = torch.where(valid2, xy, torch.zeros_like(xy))
    # Convert
    xy = xy.flip(0)  # Anti-diagonal --> diagonal
    xy = xy.reshape(*xy.shape[:2], -1)
    xy = xy.permute(2, 0, 1)  # Place subap dimensions inside
    xy = torch.triu(xy) + torch.conj(torch.tril(xy)).permute(0, 2, 1)
    dphi = torch.angle(xy)  # Compute the phase shift.
    return dphi
