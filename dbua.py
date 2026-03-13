from pathlib import Path
import time

import numpy as np
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from jaxopt import OptaxSolver
import optax
from hdf5storage import loadmat
from tqdm import tqdm

from das import das
from paths import time_of_flight
from losses import (
    lag_one_coherence,
    coherence_factor,
    phase_error,
    total_variation,
    speckle_brightness,
)


N_ITERS = 301
LEARNING_RATE = 10
ASSUMED_C = 1540  # [m/s]

# B-mode limits in m
BMODE_X_MIN = -12e-3
BMODE_X_MAX = 12e-3
BMODE_Z_MIN = 0e-3
BMODE_Z_MAX = 40e-3

# Sound speed grid in m
SOUND_SPEED_X_MIN = -12e-3
SOUND_SPEED_X_MAX = 12e-3
SOUND_SPEED_Z_MIN = 0e-3
SOUND_SPEED_Z_MAX = 40e-3
SOUND_SPEED_NXC = 19
SOUND_SPEED_NZC = 31

# Phase estimate kernel size in samples
NXK, NZK = 5, 5

# Phase estimate patch grid size in samples
NXP, NZP = 17, 17
PHASE_ERROR_X_MIN = -20e-3
PHASE_ERROR_X_MAX = 20e-3
PHASE_ERROR_Z_MIN = 4e-3
PHASE_ERROR_Z_MAX = 44e-3

# Global sound speed sweep range for initialization [m/s]
SPEED_SWEEP_MIN = 1340
SPEED_SWEEP_MAX = 1740
SPEED_SWEEP_NPTS = 201

# B-mode display dynamic range [dB]
BMODE_DR = [-45, +5]

# Default sound speed display range when no ground truth is available [m/s]
DEFAULT_SPEED_DR = [1400, 1600]

# Sound speed display range offset from ground truth [m/s]
SPEED_DR_OFFSET = 50

# Loss options
# -"pe" for phase error
# -"sb" for speckle brightness
# -"cf" for coherence factor
# -"lc" for lag one coherence

LOSS = "pe"

# Data options:
# (Constant Phantoms)
# - 1420
# - 1465
# - 1480
# - 1510
# - 1540
# - 1555
# - 1570
# (Heterogeneous Phantoms)
# - inclusion
# - inclusion_layer
# - four_layer
# - two_layer
# - checker2
# - checker8

SAMPLE = "checker2"

CTRUE = {
    "1420": 1420,
    "1465": 1465,
    "1480": 1480,
    "1510": 1510,
    "1540": 1540,
    "1555": 1555,
    "1570": 1570,
    "inclusion": 0,
    "inclusion_layer": 0,
    "four_layer": 0,
    "two_layer": 0,
    "checker2": 0,
    "checker8": 0
}


# Refocused plane wave datasets from base dataset directory
DATA_DIR = Path("./data")


def _uniform_speed_map(speed):
    """Create a uniform sound speed map filled with the given speed value."""
    return speed * jnp.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC))


def imagesc(xc, y, img, dr, **kwargs):
    """MATLAB style imagesc"""
    dx = xc[1] - xc[0]
    dy = y[1] - y[0]
    ext = [xc[0] - dx / 2, xc[-1] + dx / 2, y[-1] + dy / 2, y[0] - dy / 2]
    im = plt.imshow(img, vmin=dr[0], vmax=dr[1], extent=ext, **kwargs)
    plt.colorbar()
    return im


def load_dataset(sample):
    mdict = loadmat(f"{DATA_DIR}/{sample}.mat")
    iqdata = mdict["iqdata"]
    fs = mdict["fs"][0, 0]  # Sampling frequency
    fd = mdict["fd"][0, 0]  # Demodulation frequency
    dsf = mdict["dsf"][0, 0]  # Downsampling factor
    t = mdict["t"]  # time vector
    t0 = mdict["t0"]  # time zero of transmit
    elpos = mdict["elpos"]  # element position
    return iqdata, t0, fs, fd, elpos, dsf, t


def plot_errors_vs_sound_speeds(c0, dsb, dlc, dcf, dpe, sample):
    plt.clf()
    plt.plot(c0, dsb, label="Speckle Brightness")
    plt.plot(c0, dlc, label="Lag One Coherence")
    plt.plot(c0, dcf, label="Coherence Factor")
    # divided by 10 for visualization
    plt.plot(c0, dpe / 10, label="Phase Error")
    plt.grid()
    plt.xlabel("Global sound speed (m/s)")
    plt.ylabel("Loss function")
    plt.title(sample)
    plt.legend()
    plt.savefig(f"images/losses_{sample}.png")
    plt.savefig("scratch.png")
    plt.clf()


def _setup_coordinate_grids(wavelength):
    """Set up all coordinate grids for imaging, sound speed, and patch evaluation.

    @param wavelength: Acoustic wavelength at the demodulation frequency [m]
    @return: Dictionary containing all coordinate arrays and grid dimensions
    """
    # B-mode image pixel grid (spacing = wavelength / 3)
    xi_vec = jnp.arange(BMODE_X_MIN, BMODE_X_MAX, wavelength / 3)
    zi_vec = jnp.arange(BMODE_Z_MIN, BMODE_Z_MAX, wavelength / 3)
    nxi, nzi = xi_vec.size, zi_vec.size
    xi, zi = np.meshgrid(xi_vec, zi_vec, indexing="ij")

    # Sound speed grid
    xc = jnp.linspace(SOUND_SPEED_X_MIN, SOUND_SPEED_X_MAX, SOUND_SPEED_NXC)
    zc = jnp.linspace(SOUND_SPEED_Z_MIN, SOUND_SPEED_Z_MAX, SOUND_SPEED_NZC)

    # Kernels for loss calculations (2λ x 2λ patches)
    xk, zk = np.meshgrid(
        (jnp.arange(NXK) - (NXK - 1) / 2) * wavelength / 2,
        (jnp.arange(NZK) - (NZK - 1) / 2) * wavelength / 2,
        indexing="ij")

    # Kernel patch centers distributed throughout the field of view
    xpc, zpc = np.meshgrid(
        np.linspace(PHASE_ERROR_X_MIN, PHASE_ERROR_X_MAX, NXP),
        np.linspace(PHASE_ERROR_Z_MIN, PHASE_ERROR_Z_MAX, NZP),
        indexing="ij")

    return {
        "xi": xi, "zi": zi, "nxi": nxi, "nzi": nzi,
        "xc": xc, "zc": zc,
        "xk": xk, "zk": zk,
        "xpc": xpc, "zpc": zpc,
    }


def _build_tof_and_image_functions(iqdata, t0, fs, fd, elpos, grids):
    """Build closures for time-of-flight computation and image formation.

    Creates functions that compute TOF for image pixels and loss-evaluation
    patches, plus a beamforming function, all bound to the loaded dataset.

    @param iqdata: Raw IQ data [nrx, ntx, nsamps]
    @param t0:     Transmit time zero
    @param fs:     Sampling frequency [Hz]
    @param fd:     Demodulation frequency [Hz]
    @param elpos:  Element positions [3, n_elements]
    @param grids:  Coordinate grids from _setup_coordinate_grids()
    @return: (tof_image_fn, tof_patch_fn, make_image_fn, loss_wrapper_fn)
    """
    xe, _, ze = jnp.array(elpos)
    xi, zi = grids["xi"], grids["zi"]
    xc, zc = grids["xc"], grids["zc"]
    xk, zk = grids["xk"], grids["zk"]
    xpc, zpc = grids["xpc"], grids["zpc"]

    # Reshape for broadcasting: [elements, pixels, patches]
    xe = jnp.reshape(xe, (-1, 1, 1))
    ze = jnp.reshape(ze, (-1, 1, 1))
    # Patch positions = patch_center + kernel_offset
    xp = jnp.reshape(xpc, (1, -1, 1)) + jnp.reshape(xk, (1, 1, -1))
    zp = jnp.reshape(zpc, (1, -1, 1)) + jnp.reshape(zk, (1, 1, -1))
    # Force full broadcasting so xp and zp have identical shapes
    xp = xp + 0 * zp
    zp = zp + 0 * xp

    def tof_image(sound_speed_map):
        """Compute TOF from each element to each image pixel."""
        return time_of_flight(
            xe, ze, xi, zi, xc, zc, sound_speed_map, fnum=0.5, npts=64)

    def tof_patch(sound_speed_map):
        """Compute TOF from each element to each loss-evaluation patch."""
        return time_of_flight(
            xe, ze, xp, zp, xc, zc, sound_speed_map, fnum=0.5, npts=64)

    def make_image(sound_speed_map):
        """Beamform a B-mode image envelope for the given sound speed map."""
        tof = tof_image(sound_speed_map)
        return jnp.abs(das(iqdata, tof - t0, tof, fs, fd))

    def loss_wrapper(loss_func, sound_speed_map):
        """Evaluate a loss function using patch-based TOF delays."""
        tof = tof_patch(sound_speed_map)
        return loss_func(iqdata, tof - t0, tof, fs, fd)

    return tof_image, tof_patch, make_image, loss_wrapper


def _define_loss_functions(loss_name, tof_patch, loss_wrapper, iqdata, t0, fs, fd, dxc, dzc):
    """Construct the named loss function and individual metric functions.

    @param loss_name:    One of "sb", "lc", "cf", "pe"
    @param tof_patch:    Patch TOF closure from _build_tof_and_image_functions
    @param loss_wrapper: Loss wrapper closure
    @param iqdata:       Raw IQ data
    @param t0:           Transmit time zero
    @param fs:           Sampling frequency
    @param fd:           Demodulation frequency
    @param dxc:          Sound speed grid spacing in x
    @param dzc:          Sound speed grid spacing in z
    @return: (loss_fn, sb_loss, lc_loss, cf_loss, pe_loss) — the combined loss and individual metrics
    """
    sb_loss = jit(lambda c: 1 - loss_wrapper(speckle_brightness, c))
    lc_loss = jit(lambda c: 1 - jnp.mean(loss_wrapper(lag_one_coherence, c)))
    cf_loss = jit(lambda c: 1 - jnp.mean(loss_wrapper(coherence_factor, c)))

    @jit
    def pe_loss(sound_speed_map):
        """Phase error loss: penalizes phase shifts between common-midpoint subapertures."""
        tof = tof_patch(sound_speed_map)
        dphi = phase_error(iqdata, tof - t0, tof, fs, fd)
        valid = dphi != 0
        dphi = jnp.where(valid, dphi, jnp.nan)
        return jnp.nanmean(jnp.log1p(jnp.square(100 * dphi)))

    tv_loss = jit(lambda c: total_variation(c) * dxc * dzc)

    loss_functions = {
        "sb": lambda c: sb_loss(c) + tv_loss(c) * 1e2,
        "lc": lambda c: lc_loss(c) + tv_loss(c) * 1e2,
        "cf": lambda c: cf_loss(c) + tv_loss(c) * 1e2,
        "pe": lambda c: pe_loss(c) + tv_loss(c) * 1e2,
    }
    if loss_name not in loss_functions:
        raise NotImplementedError(
            f'Unknown loss "{loss_name}". Options: {", ".join(loss_functions.keys())}'
        )
    loss_fn = loss_functions[loss_name]

    return loss_fn, sb_loss, lc_loss, cf_loss, pe_loss


def _find_optimal_initial_speed(sb_loss, lc_loss, cf_loss, pe_loss, sample):
    """Sweep global sound speeds to find the best initialization.

    Evaluates all four loss metrics over a range of uniform sound speed maps,
    plots the results, and returns the speed that minimizes phase error.

    @param sb_loss: Speckle brightness loss function
    @param lc_loss: Lag-one coherence loss function
    @param cf_loss: Coherence factor loss function
    @param pe_loss: Phase error loss function
    @param sample:  Sample name (for plot title)
    @return: Optimal initial sound speed [m/s]
    """
    speed_range = np.linspace(SPEED_SWEEP_MIN, SPEED_SWEEP_MAX, SPEED_SWEEP_NPTS)
    dsb = np.array([sb_loss(_uniform_speed_map(cc)) for cc in speed_range])
    dlc = np.array([lc_loss(_uniform_speed_map(cc)) for cc in speed_range])
    dcf = np.array([cf_loss(_uniform_speed_map(cc)) for cc in speed_range])
    dpe = np.array([pe_loss(_uniform_speed_map(cc)) for cc in speed_range])

    plot_errors_vs_sound_speeds(speed_range, dsb, dlc, dcf, dpe, sample)
    return speed_range[np.argmin(dpe)]


def _benchmark_beamforming(make_image, sound_speed_map, n_frames=30):
    """Run a short benchmark and print beamforming throughput.

    @param make_image:      Image formation function
    @param sound_speed_map: Current sound speed map
    @param n_frames:        Number of frames to time
    """
    tic = time.perf_counter_ns()
    for _ in range(n_frames):
        bmode = make_image(sound_speed_map)
    bmode.block_until_ready()
    toc = time.perf_counter_ns()
    elapsed_sec = (toc - tic) * 1e-9
    print("jaxbf runs at %.1f fps." % (n_frames / elapsed_sec))


def _prepare_bmode_for_display(bmode_envelope, bmax, nxi, nzi):
    """Convert a beamformed envelope to a log-compressed B-mode image.

    @param bmode_envelope: Absolute beamformed IQ data
    @param bmax:           Peak value for normalization
    @param nxi:            Number of lateral pixels
    @param nzi:            Number of axial pixels
    @return: Log-compressed image in dB, shape (nzi, nxi) for display
    """
    normalized = bmode_envelope / bmax
    # Avoid log(0) by adding a tiny offset where the image is exactly zero
    normalized = normalized + 1e-10 * (normalized == 0)
    bmode_db = 20 * np.log10(normalized)
    return np.reshape(bmode_db, (nxi, nzi)).T


def _format_speed_title(iteration, speed_image, sample):
    """Format the sound speed subplot title based on ground truth availability."""
    if CTRUE[sample] > 0:
        return "Iteration %d: MAE %.2f" % (
            iteration, np.mean(np.abs(speed_image - CTRUE[sample])))
    return "Iteration %d: Mean value %.2f" % (iteration, np.mean(speed_image))


def _create_figure(fig, sound_speed_map, make_image, sb_loss, cf_loss, pe_loss,
                   ximm, zimm, xcmm, zcmm, nxi, nzi, sample):
    """Create the initial two-panel figure (B-mode + sound speed map).

    @return: (bmode_handle, speed_handle, bmode_title, speed_title, bmax) — plot handles
    """
    bmode_envelope = make_image(sound_speed_map)
    bmax = np.max(bmode_envelope)
    bmode_db = _prepare_bmode_for_display(bmode_envelope, bmax, nxi, nzi)
    speed_image = np.reshape(sound_speed_map, (SOUND_SPEED_NXC, SOUND_SPEED_NZC)).T

    # Benchmark beamforming throughput on first render
    _benchmark_beamforming(make_image, sound_speed_map)

    # Sound speed display range depends on whether ground truth is available
    if CTRUE[sample] > 0:
        speed_dr = np.array([-SPEED_DR_OFFSET, +SPEED_DR_OFFSET]) + CTRUE[sample]
        speed_cmap = "seismic"
    else:
        speed_dr = DEFAULT_SPEED_DR
        speed_cmap = "jet"

    fig.clf()

    plt.subplot(121)
    bmode_handle = imagesc(ximm, zimm, bmode_db, BMODE_DR,
                           cmap="bone", interpolation="bicubic")
    bmode_title = plt.title(
        "SB: %.2f, CF: %.3f, PE: %.3f" % (
            sb_loss(sound_speed_map), cf_loss(sound_speed_map), pe_loss(sound_speed_map)))
    plt.xlim(ximm[0], ximm[-1])
    plt.ylim(zimm[-1], zimm[0])

    plt.subplot(122)
    speed_handle = imagesc(xcmm, zcmm, speed_image, speed_dr,
                           cmap=speed_cmap, interpolation="bicubic")
    speed_title = plt.title(_format_speed_title(0, speed_image, sample))
    plt.xlim(ximm[0], ximm[-1])
    plt.ylim(zimm[-1], zimm[0])

    fig.tight_layout()
    return bmode_handle, speed_handle, bmode_title, speed_title, bmax


def _update_figure(handles, sound_speed_map, iteration, make_image,
                   sb_loss, cf_loss, pe_loss, nxi, nzi, sample):
    """Update the existing figure with new data from the current optimization step.

    @param handles: (bmode_handle, speed_handle, bmode_title, speed_title, bmax)
    """
    bmode_handle, speed_handle, bmode_title, speed_title, bmax = handles
    bmode_envelope = make_image(sound_speed_map)
    bmode_db = _prepare_bmode_for_display(bmode_envelope, bmax, nxi, nzi)
    speed_image = np.reshape(sound_speed_map, (SOUND_SPEED_NXC, SOUND_SPEED_NZC)).T

    bmode_handle.set_data(bmode_db)
    speed_handle.set_data(speed_image)
    bmode_title.set_text(
        "SB: %.2f, CF: %.3f, PE: %.3f" % (
            sb_loss(sound_speed_map), cf_loss(sound_speed_map), pe_loss(sound_speed_map)))
    speed_title.set_text(_format_speed_title(iteration, speed_image, sample))
    plt.savefig(f"scratch/{sample}.png")


def _run_optimization_loop(sound_speed_map, loss_fn, make_image,
                           sb_loss, cf_loss, pe_loss,
                           fig, grids, sample, loss_name):
    """Execute the iterative sound speed optimization with visualization and video output.

    @param sound_speed_map: Initial sound speed map
    @param loss_fn:         Combined loss function to minimize
    @param make_image:      Beamforming function
    @param sb_loss:         Speckle brightness loss (for display)
    @param cf_loss:         Coherence factor loss (for display)
    @param pe_loss:         Phase error loss (for display)
    @param fig:             Matplotlib figure handle
    @param grids:           Coordinate grids from _setup_coordinate_grids()
    @param sample:          Sample name
    @param loss_name:       Loss name (for video filename)
    @return: Optimized sound speed map
    """
    xi, zi = grids["xi"], grids["zi"]
    xc, zc = grids["xc"], grids["zc"]
    nxi, nzi = grids["nxi"], grids["nzi"]

    # Axes in millimeters for display
    ximm = xi[:, 0] * 1e3
    zimm = zi[0, :] * 1e3
    xcmm = xc * 1e3
    zcmm = zc * 1e3

    # Create optimizer
    optimizer = OptaxSolver(opt=optax.amsgrad(LEARNING_RATE), fun=loss_fn)
    state = optimizer.init_state(sound_speed_map)

    # Set up video writer
    video_writer = FFMpegWriter(fps=30)
    video_writer.setup(fig, "videos/%s_opt%s.mp4" % (sample, loss_name), dpi=144)

    # Create initial figure
    handles = _create_figure(
        fig, sound_speed_map, make_image, sb_loss, cf_loss, pe_loss,
        ximm, zimm, xcmm, zcmm, nxi, nzi, sample)

    # Optimization loop
    for i in tqdm(range(N_ITERS)):
        sound_speed_map, state = optimizer.update(sound_speed_map, state)
        _update_figure(handles, sound_speed_map, i + 1, make_image,
                       sb_loss, cf_loss, pe_loss, nxi, nzi, sample)
        video_writer.grab_frame()
    video_writer.finish()

    return sound_speed_map


def main(sample, loss_name):
    """Run the full sound speed optimization pipeline.

    Loads ultrasound data, sweeps global sound speeds for initialization,
    then iteratively optimizes a spatially-varying sound speed map by
    minimizing the selected loss function.

    @param sample:    Dataset name (must be a key in CTRUE)
    @param loss_name: Loss function to optimize ("pe", "sb", "cf", or "lc")
    @return: Optimized sound speed map [SOUND_SPEED_NXC, SOUND_SPEED_NZC]
    """
    assert (
        sample in CTRUE
    ), f'The data sample string was "{sample}".\
                            \nOptions are {", ".join(CTRUE.keys()).lstrip(" ,")}.'

    # Load dataset
    iqdata, t0, fs, fd, elpos, _, _ = load_dataset(sample)
    wavelength = ASSUMED_C / fd

    # Set up coordinate grids
    grids = _setup_coordinate_grids(wavelength)
    xc, zc = grids["xc"], grids["zc"]
    dxc, dzc = xc[1] - xc[0], zc[1] - zc[0]

    # Build TOF and image functions bound to this dataset
    tof_image, tof_patch, make_image, loss_wrapper = \
        _build_tof_and_image_functions(iqdata, t0, fs, fd, elpos, grids)

    # Define loss functions
    loss_fn, sb_loss, lc_loss, cf_loss, pe_loss = _define_loss_functions(
        loss_name, tof_patch, loss_wrapper, iqdata, t0, fs, fd, dxc, dzc)

    # Find optimal global sound speed for initialization
    optimal_speed = _find_optimal_initial_speed(
        sb_loss, lc_loss, cf_loss, pe_loss, sample)
    sound_speed_map = _uniform_speed_map(optimal_speed)

    # Run optimization with visualization
    fig, _ = plt.subplots(1, 2, figsize=[9, 4])
    sound_speed_map = _run_optimization_loop(
        sound_speed_map, loss_fn, make_image,
        sb_loss, cf_loss, pe_loss,
        fig, grids, sample, loss_name)

    return sound_speed_map


if __name__ == "__main__":
    main(SAMPLE, LOSS)

    # # Run all examples
    # for sample in CTRUE.keys():
    #     print(sample)
    #     main(sample, LOSS)
