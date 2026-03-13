from jax import vmap, jit
import jax.numpy as jnp
import numpy as np
from functools import partial


@partial(jit, static_argnums=(7, 8, 9))
def time_of_flight(x0, z0, x1, z1, xc, zc, c, fnum, npts, Dmin=3e-3):
    """
    Get the time-of-flight from (x0,z0) to (x1,z1) according to the
    speed of sound map c, defined on grid points (xc,zc).
    x0:     [...]       Path origin in x (arbitrary dimensions, broadcasting allowed)
    z0:     [...]       Path origin in z (arbitrary dimensions, broadcasting allowed)
    x1:     [...]       Path finish in x (arbitrary dimensions, broadcasting allowed)
    z1:     [...]       Path finish in z (arbitrary dimensions, broadcasting allowed)
    xc:     [nxc,]      Vector of x-grid points in sound speed definition (c.shape[0],)
    zc:     [nzc,]      Vector of x-grid points in sound speed definition
    c:      [nxc, nzc]  Sound speed map in (xc, zc) coordinates
    fnum:   scalar      f-number to apply
    npts:   scalar      Number of points in time-of-flight line segment
    Dmin:   scalar      Minimum size of the aperture, regardless of f-number
    """
    # Find the path along the path curve, modeled as a straight ray
    # parameterized by t. We will put t in the innermost dimension.
    t_all = jnp.linspace(1, 0, npts, endpoint=False)[::-1]

    # Calculate slowness map
    s = 1 / c

    def interpolate_slowness(t):
        """Bilinearly interpolate the slowness map at a point along the ray path.

        Given parameter t in [0, 1], computes the spatial position along
        the straight-line ray from (x0, z0) to (x1, z1), then samples the
        slowness map using bilinear interpolation on the (xc, zc) grid.

        @param t: Scalar in [0, 1], parameterizing the ray path
        @return:  Slowness value at the interpolated position
        """
        xt = t * (x1 - x0) + x0  # Spatial position along ray in x
        zt = t * (z1 - z0) + z0  # Spatial position along ray in z

        # Convert spatial positions to fractional grid indices in (xc, zc)
        dxc, dzc = xc[1] - xc[0], zc[1] - zc[0]
        xit = jnp.clip((xt - xc[0]) / dxc, 0, s.shape[0] - 1)
        zit = jnp.clip((zt - zc[0]) / dzc, 0, s.shape[1] - 1)

        # Floor indices for the four surrounding grid points
        xi0 = jnp.floor(xit)
        zi0 = jnp.floor(zit)
        xi1 = xi0 + 1
        zi1 = zi0 + 1

        # Sample slowness at the four surrounding grid points
        s00 = s[xi0.astype("int32"), zi0.astype("int32")]
        s10 = s[xi1.astype("int32"), zi0.astype("int32")]
        s01 = s[xi0.astype("int32"), zi1.astype("int32")]
        s11 = s[xi1.astype("int32"), zi1.astype("int32")]

        # Bilinear weights based on distance to each corner
        w00 = (xi1 - xit) * (zi1 - zit)
        w10 = (xit - xi0) * (zi1 - zit)
        w01 = (xi1 - xit) * (zit - zi0)
        w11 = (xit - xi0) * (zit - zi0)
        return s00 * w00 + s10 * w10 + s01 * w01 + s11 * w11

    # Compute the time-of-flight by integrating slowness along the ray path
    dx = jnp.abs(x1 - x0)
    dz = jnp.abs(z1 - z0)
    ray_length = jnp.sqrt(dx**2 + dz**2)
    slowness = vmap(interpolate_slowness)(t_all)
    tof = jnp.nanmean(slowness, axis=0) * ray_length

    # F-number apodization mask: reject rays with too-steep angles.
    # The f-number constraint limits the receive aperture angle:
    #   |lateral_distance| / (2 * axial_distance) <= fnum
    # Rearranged: |2 * fnum * dx| <= dz
    fnum_valid = jnp.abs(2 * fnum * dx) <= dz

    # Ensure a minimum aperture width (Dmin) regardless of f-number,
    # so that near-field points still get enough element coverage.
    # Active when axial depth is shallow (dz < Dmin * fnum) AND
    # lateral offset is within half the minimum aperture (dx < Dmin / 2).
    min_aperture_valid = jnp.logical_and(dz < Dmin * fnum, dx < Dmin / 2)

    valid = jnp.logical_or(fnum_valid, min_aperture_valid)
    # Invalid regions get a large negative TOF so they interpolate as zero
    tof = jnp.where(valid, tof, -10 * jnp.ones_like(tof))
    return tof


