import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from das import das


@partial(jit, static_argnums=(3, 4))
def lag_one_coherence(iq, t_tx, t_rx, fs, fd):
    """
    Lag-one coherence of the receive aperture (DOI: 10.1109/TUFFC.2018.2855653).
    The LOC measures the quality of a signal relative to its noise, and can be
    used to select acoustic output.
    """
    iq = jnp.transpose(iq, (1, 0, 2))  # Place rx aperture in 0-th index
    rxdata = das(iq, t_rx, t_tx, fs, fd, jnp.eye(iq.shape[0]))  # Get rx channel data
    # Compute the correlation coefficient
    xy = jnp.real(jnp.nansum(rxdata[:-1] * jnp.conj(rxdata[1:]), axis=0))
    xx = jnp.nansum(jnp.abs(rxdata[:-1]) ** 2, axis=0)
    yy = jnp.nansum(jnp.abs(rxdata[1:]) ** 2, axis=0)
    ncc = xy / jnp.sqrt(xx * yy)
    return ncc


@partial(jit, static_argnums=(3, 4))
def coherence_factor(iq, t_tx, t_rx, fs, fd):
    """
    The coherence factor of the receive aperture (DOI: 10.1121/1.410562).
    The CF is a focusing criterion used to measure the amount of aberration in
    an image.
    """
    iq = jnp.transpose(iq, (1, 0, 2))  # Place rx aperture in 0-th index
    rxdata = das(iq, t_rx, t_tx, fs, fd, jnp.eye(iq.shape[0]))  # Get rx channel data
    num = jnp.abs(jnp.nansum(rxdata, axis=0))
    den = jnp.nansum(jnp.abs(rxdata), axis=0)
    return num / den


@partial(jit, static_argnums=(3, 4))
def speckle_brightness(iq, t_tx, t_rx, fs, fd):
    """
    The speckle brightness criterion (DOI: 10.1121/1.397889)
    Speckle brightness can be used to measure the focusing quality.
    """
    return jnp.nanmean(jnp.abs(das(iq, t_tx, t_rx, fs, fd)))


@jit
def total_variation(c):
    """
    Total variation of sound speed map in x and z.
    The sound speed map c should be specified as a 2D matrix of size [nx, nz]
    """
    tvx = jnp.nanmean(jnp.square(jnp.diff(c, axis=0)))
    tvz = jnp.nanmean(jnp.square(jnp.diff(c, axis=1)))
    return tvx + tvz


def _build_subaperture_masks(nrx, ntx, halfsa=8, stride=1):
    """Build transmit and receive subaperture masks for common-midpoint beamforming.

    Creates masks that select overlapping subapertures from the full aperture.
    Each subaperture has 2*halfsa+1 elements. The tx mask is the reverse of
    the rx mask so that paired subapertures share a common midpoint.

    @param nrx:    Number of receive elements
    @param ntx:    Number of transmit elements
    @param halfsa: Half-width of each subaperture (full width = 2*halfsa+1)
    @param stride: Step size between adjacent subapertures
    @return: (tx_mask, rx_mask) for use in das() as combination matrices
    """
    mask = np.zeros((nrx, ntx))
    for diag in range(-halfsa, halfsa + 1):
        mask = mask + jnp.diag(jnp.ones((ntx - abs(diag),)), diag)
    mask = mask[halfsa: mask.shape[0] - halfsa: stride]
    tx_mask = mask[::-1]
    rx_mask = mask
    return tx_mask, rx_mask


def _compute_midpoint_correlation(iqfoc, thresh=0.9):
    """Compute correlation between neighboring common-midpoint subaperture signals.

    For a subaperture IQ matrix indexed by (tx_subap, rx_subap, ...),
    correlates neighboring diagonal entries that share a common midpoint,
    e.g. <(i,j), (i+1,j+1)>. Only keeps correlations above the given
    coherence threshold to reject noisy estimates.

    @param iqfoc:  Focused subaperture IQ data from das()
    @param thresh: Minimum correlation coefficient to accept (0 to 1)
    @return: Complex cross-correlation matrix with low-coherence entries zeroed
    """
    # Cross- and auto-correlations between diagonal neighbors
    prev_subap = iqfoc[:-1, :-1]
    next_subap = iqfoc[+1:, +1:]
    xy = prev_subap * jnp.conj(next_subap)
    xx = prev_subap * jnp.conj(prev_subap)
    yy = next_subap * jnp.conj(next_subap)

    # Zero out correlations where either signal is missing
    both_nonzero = (prev_subap != 0) & (next_subap != 0)
    xy = jnp.where(both_nonzero, xy, 0)
    xx = jnp.where(both_nonzero, xx, 0)
    yy = jnp.where(both_nonzero, yy, 0)

    # Sum over kernel dimension, then threshold by coherence
    xy = jnp.sum(xy, axis=-1)
    xx = jnp.sum(xx, axis=-1)
    yy = jnp.sum(yy, axis=-1)
    coherence_squared = jnp.square(jnp.abs(xy)) / (jnp.abs(xx) * jnp.abs(yy))
    above_threshold = coherence_squared > thresh * thresh
    xy = jnp.where(above_threshold, xy, 0)
    return xy


def _extract_phase_angles(xy_correlation):
    """Convert cross-correlation matrix to phase shift estimates.

    Rearranges the anti-diagonal correlation structure into a symmetric
    matrix and extracts phase angles. The anti-diagonal is flipped to
    align with the diagonal convention, then the upper and conjugate-lower
    triangles are combined for a Hermitian result.

    @param xy_correlation: Cross-correlation from _compute_midpoint_correlation
    @return: Phase shift matrix [n_patches, n_subap, n_subap]
    """
    xy = xy_correlation[::-1]  # Anti-diagonal --> diagonal ordering
    xy = jnp.reshape(xy, (*xy.shape[:2], -1))
    xy = jnp.transpose(xy, (2, 0, 1))  # Place subap dimensions inside
    # Make Hermitian: combine upper triangle with conjugate of lower triangle
    xy = jnp.triu(xy) + jnp.transpose(jnp.conj(jnp.tril(xy)), (0, 2, 1))
    return jnp.angle(xy)


@partial(jit, static_argnums=(3, 4, 5))
def phase_error(iq, t_tx, t_rx, fs, fd, thresh=0.9):
    """
    The phase error between translating transmit and receive apertures.
    This error is closely related to the "Translated Transmit Apertures" algorithm
    (DOI: 10.1109/58.585209), where translated transmit and receive apertures
    with common midpoint should have perfect speckle correlation by the van
    Cittert Zernike theorem (DOI: 10.1121/1.418235). High correlation will
    result in high-quality phase shift estimates (DOI: 10.1121/10.0000809).
    CUTE also takes a similar approach (DOI: 10.1016/j.ultras.2020.106168),
    but in the angular basis instead of the element basis.
    """
    nrx, ntx, nsamps = iq.shape
    tx_mask, rx_mask = _build_subaperture_masks(nrx, ntx)
    iqfoc = das(iq, t_tx, t_rx, fs, fd, tx_mask, rx_mask)
    xy_correlation = _compute_midpoint_correlation(iqfoc, thresh)
    dphi = _extract_phase_angles(xy_correlation)
    return dphi
