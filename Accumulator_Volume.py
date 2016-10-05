__author__ = 'Michael Rossol'

__all__ = ['extract_DIC_numb', 'DIC_P', 'img_numb', 'get_h', 'PV_werror', 'sphere_fit', 'hemisphere_PV', 'cap_PV',
           'DIC_PV', 'paraboloid_PV', 'p_integrate_PV', 'get_lines', 'get_all_lines', 'get_shifts', 'contour_overlay']

import numpy as np
import scipy as sc
import scipy.optimize
import scipy.interpolate
import DIC
import sympy as sp
import scipy.io
from scipy import misc
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as mplt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def extract_DIC_numb(file):
    """
    extract file number from .mat file
    Parameters
    ----------
    file : 'string'
        .mat file path

    Returns
    -------
    numb : 'int'
        file number
    """
    numb = file.split('/')[-1]
    return int(numb[:4])


def DIC_P(data_path, DIC_data, t_shift=None, KPa=False):
    """
    Get pressure for each .mat file
    Parameters
    ----------
    data_path : 'string'
        file path for .dat file
    DIC_data : 'instance'
        DIC Class instance
    t_shift : 'float'
        time shift to align pressure and DIC data, default is None

    Returns
    -------
        Array of [img #, time, pressure]
    """
    data = np.loadtxt(data_path, skiprows=2)

    t_p = data[:, :2]
    if t_shift is not None:
        t_p = t_p + [t_shift, 0]

    img_t = data[1:, 2:]/[1,1000]
    img_t = np.asarray([np.mean(img_t[img_t[:, 0] == img], axis=0) for img in np.unique(img_t[:, 0])])
    img_t_interp = scipy.interpolate.interp1d(img_t[:, 0], img_t[:, 1])

    imgs = [extract_DIC_numb(file) for file in DIC_data.mat]
    times = img_t_interp(imgs)
    pressures = [t_p[DIC.nearest(t_p[:, 0], t), 1] for t in times]

    if KPa:
        pressures = np.asarray(pressures) * 6.89476

    return np.dstack((imgs, times, pressures))[0]


def img_numb(data, pressure):
    """
    ?
    Parameters
    ----------
    data
    pressure

    Returns
    -------

    """
    return data[0, 0] + DIC.nearest(data[:, 2], pressure)


def get_h(DIC_data, method='fit', z_shift=None):
    """
    extract accumulator height
    Parameters
    ----------
    DIC_data : 'instance'
        DIC Class instance
    method : 'string'
        method to extract height, default = 'fit' (hemisphere)
    z_shift : 'float'
        z offset

    Returns
    -------
        Array of height values in mm
    """

    data = DIC_data.get_data(0)
    sigma = data['sigma'].flatten()
    Z = data['Z'].flatten()
    Z[sigma == -1] = np.nan
    if z_shift is None:
        Zo = 0
    elif z_shift.lower().startswith('max'):
        Zo = np.nanmax(Z)
    elif z_shift.lower().startswith('min'):
        Zo = np.nanmin(Z)

    bad_pos = np.where(data['sigma'].flatten() == -1)
    X = np.delete(data['X'].flatten(), bad_pos)
    Y = np.delete(data['Y'].flatten(), bad_pos)
    x_lims = [X.min(), X.max()]
    y_lims = [Y.min(), Y.max()]
    a0 = np.mean((x_lims[1] - x_lims[0], y_lims[1] - y_lims[0]))/2

    height = []
    for frame in range(len(DIC_data.mat)):
        data = DIC_data.get_data(frame)
        bad_pos = np.where(data['sigma'].flatten() == -1)
        X = np.delete(data['X'].flatten(), bad_pos)
        Y = np.delete(data['Y'].flatten(), bad_pos)
        Z = np.delete(data['Z'].flatten(), bad_pos)
        U = np.delete(data['U'].flatten(), bad_pos)
        V = np.delete(data['V'].flatten(), bad_pos)
        W = np.delete(data['W'].flatten(), bad_pos)
        Xi = X + U
        Yi = Y + V
        Zi = Z + W - Zo

        if method.lower().startswith('m'):
            h = Zi.max
        else:
            p0 = [0, 0, 0, 1]

            def fitfunc(p, coords):
                x0, y0, z0, R = p
                x, y, z = coords
                return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)

            errfunc = lambda p, x: fitfunc(p, x) - p[3]

            coords = np.vstack((Xi, Yi, Zi))

            (x0, y0, z0, R), flag = scipy.optimize.leastsq(errfunc, p0, args=(coords,))

            h = R + z0

            a = np.sqrt(h*(2*R - h))

            if np.isnan(a) or a > a0*1.25 or z0>0:
                h = np.nan

        height.append(h)

    return np.asarray(height)


def PV_werror(DIC_data, pressure, step=None, z_shift=None):
    """
    Extract accumulator volume as function of pressure with error in volume calculation
    Parameters
    ----------
    DIC_data : 'instance'
        DIC Class instance
    pressure : 'array'
        pressure values
    step : 'float'
        step size for gridding DIC data, default = DIC step size
    z_shift : 'float'
        z offset

    Returns
    -------
        Array of [[pressure, volume], [np.nan, volume_error]]
    """

    data = DIC_data.get_data(0)
    sigma = data['sigma'].flatten()
    Z = data['Z'].flatten()
    Z[sigma == -1] = np.nan
    if z_shift is None:
        Zo = 0
    elif z_shift.lower().startswith('max'):
        Zo = np.nanmax(Z)
    elif z_shift.lower().startswith('min'):
        Zo = np.nanmin(Z)

    bad_pos = np.where(data['sigma'].flatten() == -1)
    X = np.delete(data['X'].flatten(), bad_pos)
    Y = np.delete(data['Y'].flatten(), bad_pos)
    x_lims = [X.min(), X.max()]
    y_lims = [Y.min(), Y.max()]
    a0 = np.mean((x_lims[1] - x_lims[0], y_lims[1] - y_lims[0]))/2

    volume = []
    error = []
    for frame in range(len(DIC_data.mat)):
        data = DIC_data.get_data(frame)
        bad_pos = np.where(data['sigma'].flatten() == -1)
        X = np.delete(data['X'].flatten(), bad_pos)
        Y = np.delete(data['Y'].flatten(), bad_pos)
        Z = np.delete(data['Z'].flatten(), bad_pos)
        U = np.delete(data['U'].flatten(), bad_pos)
        V = np.delete(data['V'].flatten(), bad_pos)
        W = np.delete(data['W'].flatten(), bad_pos)
        Xi = X + U
        Yi = Y + V
        Zi = Z + W - Zo

        p0 = [0, 0, 0, 1]

        def fitfunc(p, coords):
            x0, y0, z0, R = p
            x, y, z = coords
            return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)

        errfunc = lambda p, x: fitfunc(p, x) - p[3]

        coords = np.vstack((Xi, Yi, Zi))

        (x0, y0, z0, R), flag = scipy.optimize.leastsq(errfunc, p0, args=(coords,))

        h = R + z0

        a_h = np.sqrt(h*(2*R - h))

        if np.isnan(a_h) or a_h > a0*1.25 or z0>0:
            V = np.nan
            dV = np.nan
        else:
            XYi = np.dstack((Xi, Yi))[0]

            if step is None:
                step = np.round(DIC_data.get_hstep()*DIC_data.get_mag()[1], 2)

            xrange = np.arange(Xi.min(), Xi.max() + step, step)
            yrange = np.arange(Yi.min(), Yi.max() + step, step)

            grid_x, grid_y = np.meshgrid(xrange, yrange)
            grid_z = scipy.interpolate.griddata(XYi, Zi, (grid_x, grid_y), method='linear')

            xyz = np.dstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()))[0]
            xyz = xyz[~np.isnan(xyz[:, 2])]

            def h_z(xy):
                x, y = xy.T
                return np.sqrt(R**2 -(x-x0)**2 - (y-y0)**2) + z0

            dz = xyz[:, 2] - h_z(xyz[:, :2])
            dV = np.sqrt(np.sum(dz**2))*step**2

            V = (1/6)*np.pi*h*(3*a_h**2 + h**2)

        volume.append(V)
        error.append(dV)

    PV = np.dstack((pressure, np.asarray(volume)))[0]

    PVe = np.dstack((np.zeros(len(pressure))*np.nan, np.asarray(error)))[0]
    return np.dstack((PV, PVe))


def sphere_fit(DIC_data, pressure, z_shift=None):
    """
    extract spherical fit parameters
    Parameters
    ----------
    DIC_data : 'instance'
        DIC Class instance
    pressure : 'array'
        Array of pressure values
    z_shift :
        z offset

    Returns
    -------
        Array of [P_i, h, R, a_h, V, dV]
    """

    data = DIC_data.get_data(0)
    sigma = data['sigma'].flatten()
    Z = data['Z'].flatten()
    Z[sigma == -1] = np.nan
    if z_shift is None:
        Zo = 0
    elif z_shift.lower().startswith('max'):
        Zo = np.nanmax(Z)
    elif z_shift.lower().startswith('min'):
        Zo = np.nanmin(Z)

    bad_pos = np.where(data['sigma'].flatten() == -1)
    X = np.delete(data['X'].flatten(), bad_pos)
    Y = np.delete(data['Y'].flatten(), bad_pos)
    x_lims = [X.min(), X.max()]
    y_lims = [Y.min(), Y.max()]
    a0 = np.mean((x_lims[1] - x_lims[0], y_lims[1] - y_lims[0]))/2

    output=[]
    for frame in range(len(DIC_data.mat)):
        P_i = pressure[frame]

        data = DIC_data.get_data(frame)
        bad_pos = np.where(data['sigma'].flatten() == -1)
        X = np.delete(data['X'].flatten(), bad_pos)
        Y = np.delete(data['Y'].flatten(), bad_pos)
        Z = np.delete(data['Z'].flatten(), bad_pos)
        U = np.delete(data['U'].flatten(), bad_pos)
        V = np.delete(data['V'].flatten(), bad_pos)
        W = np.delete(data['W'].flatten(), bad_pos)
        Xi = X + U
        Yi = Y + V
        Zi = Z + W - Zo

        p0 = [0, 0, 0, 1]

        def fitfunc(p, coords):
            x0, y0, z0, R = p
            x, y, z = coords
            return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)

        errfunc = lambda p, x: fitfunc(p, x) - p[3]

        coords = np.vstack((Xi, Yi, Zi))

        (x0, y0, z0, R), flag = scipy.optimize.leastsq(errfunc, p0, args=(coords,))

        h = R + z0

        a_h = np.sqrt(h*(2*R - h))

        if np.isnan(a_h) or a_h > a0*1.25 or z0>0:
            output.append([P_i, np.nan,  np.nan,  np.nan,  np.nan,  np.nan])
        else:
            XYi = np.dstack((Xi, Yi))[0]

            step = np.round(DIC_data.get_hstep()*DIC_data.get_mag()[1], 2)

            xrange = np.arange(Xi.min(), Xi.max() + step, step)
            yrange = np.arange(Yi.min(), Yi.max() + step, step)

            grid_x, grid_y = np.meshgrid(xrange, yrange)
            grid_z = scipy.interpolate.griddata(XYi, Zi, (grid_x, grid_y), method='linear')

            xyz = np.dstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()))[0]
            xyz = xyz[~np.isnan(xyz[:, 2])]

            def h_z(xy):
                x, y = xy.T
                return np.sqrt(R**2 -(x-x0)**2 - (y-y0)**2) + z0

            dz = xyz[:, 2] - h_z(xyz[:, :2])
            dV = np.sqrt(np.sum(dz**2))*step**2

            V = (1/6)*np.pi*h*(3*a_h**2 + h**2)

            output.append([P_i, h, R, a_h, V, dV])

    return np.asarray(output)


def hemisphere_PV(DIC_data, pressure, a=None, z_shift=None):
    """
    Extract accumulator volume as function of pressure using hemispherical cap
    Parameters
    ----------
    DIC_data : 'instance'
        DIC Class instance
    pressure : 'array'
        pressure values
    a : 'float'
        cap radius, default = use spherical fit
    z_shift : 'float'
        z offset

    Returns
    -------
        Array of [pressure, volume]
    """
    data = DIC_data.get_data(0)
    sigma = data['sigma'].flatten()
    Z = data['Z'].flatten()
    Z[sigma == -1] = np.nan
    if z_shift is None:
        Zo = 0
    elif z_shift.lower().startswith('max'):
        Zo = np.nanmax(Z)
    elif z_shift.lower().startswith('min'):
        Zo = np.nanmin(Z)

    bad_pos = np.where(data['sigma'].flatten() == -1)
    X = np.delete(data['X'].flatten(), bad_pos)
    Y = np.delete(data['Y'].flatten(), bad_pos)
    x_lims = [X.min(), X.max()]
    y_lims = [Y.min(), Y.max()]
    a0 = np.mean((x_lims[1] - x_lims[0], y_lims[1] - y_lims[0]))/2

    volume = []
    for frame in range(len(DIC_data.mat)):
        data = DIC_data.get_data(frame)
        bad_pos = np.where(data['sigma'].flatten() == -1)
        X = np.delete(data['X'].flatten(), bad_pos)
        Y = np.delete(data['Y'].flatten(), bad_pos)
        Z = np.delete(data['Z'].flatten(), bad_pos)
        U = np.delete(data['U'].flatten(), bad_pos)
        V = np.delete(data['V'].flatten(), bad_pos)
        W = np.delete(data['W'].flatten(), bad_pos)
        Xi = X + U
        Yi = Y + V
        Zi = Z + W - Zo

        p0 = [0, 0, 0, 1]

        def fitfunc(p, coords):
            x0, y0, z0, R = p
            x, y, z = coords
            return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)

        errfunc = lambda p, x: fitfunc(p, x) - p[3]

        coords = np.vstack((Xi, Yi, Zi))

        (x0, y0, z0, R), flag = scipy.optimize.leastsq(errfunc, p0, args=(coords,))

        h = R + z0

        a_h = np.sqrt(h*(2*R - h))

        if np.isnan(a_h) or a_h > a0*1.25 or z0>0:
            a_h = a0
            h = Zi.max()

        if a is None:
            V = (1/6)*np.pi*h*(3*a_h**2 + h**2)
        else:
            def h_fit(x):
                return np.sqrt(R**2 - x**2) + z0

            z_a = h_fit(a)
            h_a = h - z_a
            V = (1/6)*np.pi*h_a*(3*a**2 + h_a**2) + np.pi*a**2*z_a

        volume.append(V)

    return np.dstack((pressure, np.asarray(volume)))[0]


def cap_PV(DIC_data, pressure, a):
    """
    Extract accumulator volume as function of pressure using hemispherical cap equation
    Parameters
    ----------
    DIC_data : 'instance'
        DIC Class instance
    pressure : 'array'
        pressure values
    a : 'float'
        cap radius

    Returns
    -------
        Array of [pressure, volume]
    """
    data = DIC_data.get_data(0)
    sigma = data['sigma'].flatten()
    Z = data['Z'].flatten()

    data = DIC_data.get_data(pressure.argmax())
    max_W = data['W'].flatten()
    max_W[sigma == -1] = np.nan
    max_pos = np.nanargmax(max_W)

    Z[sigma == -1] = np.nan
    Zo = Z[max_pos] - np.nanmin(Z)

    volume = []
    for frame in range(len(DIC_data.mat)):
        data = DIC_data.get_data(frame)
        W = data['W'].flatten()
        h = W[max_pos] + Zo
        volume.append((1/6)*np.pi*h*(3*a**2 + h**2))

    return np.dstack((pressure, np.asarray(volume)))[0]


def DIC_PV(DIC_data, pressure, step=0.01, method='linear', z_shift=None):
    """
    Extract accumulator volume as function of pressure by integreting DIC data
    Parameters
    ----------
    DIC_data : 'instance'
        DIC Class instance
    pressure : 'array'
        pressure values
    step : 'float'
        mesh grid spacing, default = 0.01 mm
    method : 'string'
        method of integration to re-grid DIC data
    z_shift : 'float'
        z offset, default = None

    Returns
    -------
        Array of [pressure, volume]
    """
    data = DIC_data.get_data(0)
    sigma = data['sigma'].flatten()
    Z = data['Z'].flatten()
    Z[sigma == -1] = np.nan
    if z_shift is None:
        Zo = 0
    elif z_shift.lower().startswith('max'):
        Zo = np.nanmax(Z)
    elif z_shift.lower().startswith('min'):
        Zo = np.nanmin(Z)

    volume = []
    for frame in range(len(DIC_data.mat)):
        data = DIC_data.get_data(frame)
        bad_pos = np.where(data['sigma'].flatten() == -1)
        X = np.delete(data['X'].flatten(), bad_pos)
        Y = np.delete(data['Y'].flatten(), bad_pos)
        Z = np.delete(data['Z'].flatten(), bad_pos)
        U = np.delete(data['U'].flatten(), bad_pos)
        V = np.delete(data['V'].flatten(), bad_pos)
        W = np.delete(data['W'].flatten(), bad_pos)
        Xi = X + U
        Yi = Y + V
        Zi = Z + W - Zo

        XYi = np.dstack((Xi, Yi))[0]

        xrange = np.arange(Xi.min(), Xi.max() + step, step)
        yrange = np.arange(Yi.min(), Yi.max() + step, step)

        grid_x, grid_y = np.meshgrid(xrange, yrange)
        grid_z = scipy.interpolate.griddata(XYi, Zi, (grid_x, grid_y), method=method)

        volume.append(np.nansum(grid_z)*step**2)

    return np.dstack((pressure, np.asarray(volume)))[0]


def paraboloid_PV(DIC_data, pressure, z_shift=None):
    """
    Extract accumulator volume as function of pressure using parabolic fit and parabolic cap equation
    Parameters
    ----------
    DIC_data : 'instance'
        DIC Class instance
    pressure : 'array'
        pressure values
    z_shift : 'float'
        z offset, default = None

    Returns
    -------
        Array of [pressure, volume]
    """
    data = DIC_data.get_data(0)
    sigma = data['sigma'].flatten()
    Z = data['Z'].flatten()
    Z[sigma == -1] = np.nan
    if z_shift is None:
        Zo = 0
    elif z_shift.lower().startswith('max'):
        Zo = np.nanmax(Z)
    elif z_shift.lower().startswith('min'):
        Zo = np.nanmin(Z)

    bad_pos = np.where(data['sigma'].flatten() == -1)
    X = np.delete(data['X'].flatten(), bad_pos)
    Y = np.delete(data['Y'].flatten(), bad_pos)
    x_lims = [X.min(), X.max()]
    y_lims = [Y.min(), Y.max()]
    a = np.mean((x_lims[1] - x_lims[0], y_lims[1] - y_lims[0]))/2

    volume = []
    for frame in range(len(DIC_data.mat)):
        data = DIC_data.get_data(frame)
        bad_pos = np.where(data['sigma'].flatten() == -1)
        X = np.delete(data['X'].flatten(), bad_pos)
        Y = np.delete(data['Y'].flatten(), bad_pos)
        Z = np.delete(data['Z'].flatten(), bad_pos)
        U = np.delete(data['U'].flatten(), bad_pos)
        V = np.delete(data['V'].flatten(), bad_pos)
        W = np.delete(data['W'].flatten(), bad_pos)
        Xi = X + U
        Yi = Y + V
        Zi = Z + W - Zo

        A = np.vstack([Xi**2, Xi, Yi**2, Yi, np.ones(len(Zi))]).T
        params = np.linalg.lstsq(A, Zi)[0]

        def ifit(x):
            return -1*(params[0]*x[0]**2 + params[1]*x[0] + params[2]*x[1]**2 + params[3]*x[1] + params[4])

        xo, yo = sc.optimize.fmin(ifit, np.array([0, 0]), disp=False)

        if xo > x_lims[1] or xo < x_lims[0] or yo > y_lims[1] or yo < y_lims[0]:
            xo = 0
            yo = 0

        else:
            def xzfit(x):
                return params[0]*x**2 + params[1]*x + params[2]*yo**2 + params[3]*yo + params[4]
            x_lims = [sc.optimize.fsolve(xzfit, x_lims[0]), sc.optimize.fsolve(xzfit, x_lims[1])]

            def yzfit(y):
                return params[0]*xo**2 + params[1]*xo + params[2]*y**2 + params[3]*y + params[4]
            y_lims = [sc.optimize.fsolve(yzfit, y_lims[0]), sc.optimize.fsolve(yzfit, y_lims[1])]

            ao = np.mean((x_lims[1] - x_lims[0], y_lims[1] - y_lims[0]))/2

            if ao < a*1.25:
                a = ao

        h = -1 * ifit([xo, yo])

        volume.append((1/2)*np.pi*a**2*h)

    return np.dstack((pressure, np.asarray(volume)))[0]


def p_integrate_PV(DIC_data, pressure, z_shift=None):
    """
    Extract accumulator volume as function of pressure using intergration of parabolic fit equation
    Parameters
    ----------
    DIC_data : 'instance'
        DIC Class instance
    pressure : 'array'
        pressure values
    a : 'float'
        cap radius

    Returns
    -------
        Array of [pressure, volume]
    """
    data = DIC_data.get_data(0)
    sigma = data['sigma'].flatten()
    Z = data['Z'].flatten()
    Z[sigma == -1] = np.nan
    if z_shift is None:
        Zo = 0
    elif z_shift.lower().startswith('max'):
        Zo = np.nanmax(Z)
    elif z_shift.lower().startswith('min'):
        Zo = np.nanmin(Z)

    volume = []
    for frame in range(len(DIC_data.mat)):
        data = DIC_data.get_data(frame)
        bad_pos = np.where(data['sigma'].flatten() == -1)
        X = np.delete(data['X'].flatten(), bad_pos)
        Y = np.delete(data['Y'].flatten(), bad_pos)
        Z = np.delete(data['Z'].flatten(), bad_pos)
        W = np.delete(data['W'].flatten(), bad_pos)
        Zi = Z + W - Zo

        A = np.vstack([X**2, X, Y**2, Y, np.ones(len(Zi))]).T
        params = np.linalg.lstsq(A, Zi)[0]

        def ifit(x):
            return -1*(params[0]*x[0]**2 + params[1]*x[0] + params[2]*x[1]**2 + params[3]*x[1] + params[4])

        xo, yo = sc.optimize.fmin(ifit, np.array([0, 0]), disp=False)
        x_range = [X.min(), X.max()]
        y_range = [Y.min(), Y.max()]
        if xo > X.max() or xo < X.min() or yo > Y.max() or yo < Y.min():
            xo = 0
            yo = 0

            x_lims = x_range
            y_lims = y_range

            h = -1* ifit([xo, yo])
            a = np.mean((x_lims[1] - x_lims[0], y_lims[1] - y_lims[0]))/2
            V = (1/2)*np.pi*a**2*h
        else:
            def xzfit(x):
                return params[0]*x**2 + params[1]*x + params[2]*yo**2 + params[3]*yo + params[4]
            x_lims = [sc.optimize.fsolve(xzfit, x_range[0]), sc.optimize.fsolve(xzfit, x_range[1])]

            def yzfit(y):
                return params[0]*xo**2 + params[1]*xo + params[2]*y**2 + params[3]*y + params[4]
            y_lims = [sc.optimize.fsolve(yzfit, y_range[0]), sc.optimize.fsolve(yzfit, y_range[1])]

            if np.mean((x_lims[1] - x_lims[0], y_lims[1] - y_lims[0]))/2 > \
                            (np.mean((x_range[1] - x_range[0], y_range[1] - y_range[0]))/2)*1.25:
                y_lims = y_range

            xf, yf = sp.symbols('xf yf')
            sols = sp.solve(params[0]*xf**2 + params[1]*xf + params[2]*yf**2 + params[3]*yf + params[4], xf)

            V = sc.integrate.dblquad(lambda x, y: params[0]*x**2 + params[1]*x + params[2]*y**2 + params[3]*y
                                                  + params[4], y_lims[0], y_lims[1],
                                   lambda x: sols[0].subs(yf, x), lambda x: sols[1].subs(yf, x))[0]

        volume.append(V)

    return np.dstack((pressure, np.asarray(volume)))[0]


def get_lines(DIC_data, frame, a, xo=0, yo=0, zo=0):
    """
    Extract line scans through accumulator center.
    Parameters
    ----------
    DIC_data : 'instance'
        DIC Class instance
    frame : 'int'
        frame number
    a : 'float'
        cap radius
    xo : 'float'
        x center
    yo : 'float'
        y center
    zo : 'float'
        z center

    Returns
    -------
        Arrays of [x, z] for DIC_data, hemisphere_fit, hemisphere_cap
    """
    data = DIC_data.get_data(frame)
    sigma = data['sigma']
    bad_pos = np.where(sigma.flatten() == -1)
    Xa = data['X'] - xo + data['U']
    X = np.delete(Xa.flatten(), bad_pos)
    Ya = data['Y'] - yo + data['V']
    Y = np.delete(Ya.flatten(), bad_pos)
    Za = data['Z'] - zo
    Z = np.delete(Za.flatten(), bad_pos)
    Wa = data['W']
    W = np.delete(Wa.flatten(), bad_pos)
    Zia = Za + Wa
    Zi = Z + W

    xlen = sigma.shape[0]
    pos = DIC.nearest(Xa[int(np.round(xlen/2))], 0)
    DIC_line = np.dstack((Xa, Zia))[pos]
    DIC_line = np.delete(DIC_line, np.where(sigma[pos] == -1), axis=0)

    p0 = [0, 0, 0, 1]

    def fitfunc(p, coords):
        x0, y0, z0, R = p
        x, y, z = coords
        return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)

    errfunc = lambda p, x: fitfunc(p, x) - p[3]

    coords = np.vstack((X, Y, Zi))

    (x0, y0, z0, R), flag = scipy.optimize.leastsq(errfunc, p0, args=(coords,))

    def h_fit(x):
        return np.sqrt(R**2 - (x-x0)**2) + z0

    hemi_line = np.dstack((np.arange(-a, a+.1, 0.1), h_fit(np.arange(-a, a+.1, 0.1))))[0]

    h = DIC_line[:, 1].max()
    r = (a**2 + h**2)/(2*h)

    def cap(x):
        return np.sqrt(r**2 - x**2) + (h - r)

    cap_line = np.dstack((np.arange(-a, a+.1, 0.1), cap(np.arange(-a, a+.1, 0.1))))[0]

    return DIC_line, hemi_line, cap_line


def get_all_lines(DIC_data, frame, a, xo=0, yo=0, zo=0):
    """
    Extract all line scans through accumulator center.
    Parameters
    ----------
    DIC_data : 'instance'
        DIC Class instance
    frame : 'int'
        frame number
    a : 'float'
        cap radius
    xo : 'float'
        x center
    yo : 'float'
        y center
    zo : 'float'
        z center

    Returns
    -------
        Arrays of [x, z] for DIC_data, hemisphere_fit, hemisphere_cap, parabolic_fit
    """
    data = DIC_data.get_data(frame)
    sigma = data['sigma']
    bad_pos = np.where(sigma.flatten() == -1)
    Xa = data['X'] - xo + data['U']
    X = np.delete(Xa.flatten(), bad_pos)
    Ya = data['Y'] - yo + data['V']
    Y = np.delete(Ya.flatten(), bad_pos)
    Za = data['Z'] - zo
    Z = np.delete(Za.flatten(), bad_pos)
    Wa = data['W']
    W = np.delete(Wa.flatten(), bad_pos)
    Zia = Za + Wa
    Zi = Z + W

    xlen = sigma.shape[0]
    pos = DIC.nearest(Xa[int(np.round(xlen/2))], 0)
    DIC_line = np.dstack((Xa, Zia))[pos]
    DIC_line = np.delete(DIC_line, np.where(sigma[pos] == -1), axis=0)

    p0 = [0, 0, 0, 1]

    def fitfunc(p, coords):
        x0, y0, z0, R = p
        x, y, z = coords
        return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)

    errfunc = lambda p, x: fitfunc(p, x) - p[3]

    coords = np.vstack((X, Y, Zi))

    (x0, y0, z0, R), flag = scipy.optimize.leastsq(errfunc, p0, args=(coords,))

    def h_fit(x):
        return np.sqrt(R**2 - (x-x0)**2) + z0

    hemi_line = np.dstack((np.arange(-a, a+.1, 0.1), h_fit(np.arange(-a, a+.1, 0.1))))[0]

    h = DIC_line[:, 1].max()
    r = (a**2 + h**2)/(2*h)

    def cap(x):
        return np.sqrt(r**2 - x**2) + (h - r)

    cap_line = np.dstack((np.arange(-a, a+.1, 0.1), cap(np.arange(-a, a+.1, 0.1))))[0]

    A = np.vstack([X**2, X, Y**2, Y, np.ones(len(Zi))]).T
    params = np.linalg.lstsq(A, Zi)[0]

    def p_fit(x, y):
        return (params[0]*x**2 + params[1]*x + params[2]*y**2 + params[3]*y + params[4])

    para_line = np.dstack((np.arange(-a, a+.1, 0.1), p_fit(np.arange(-a, a+.1, 0.1), 0)))[0]

    return DIC_line, hemi_line, cap_line, para_line


def get_shifts(DIC_data, frame = -1, z_shift=None):
    """
    Extract accumulator center
    Parameters
    ----------
    DIC_data : 'instance'
        DIC Class instance
    frame : 'int'
        frame number, default = -1 (last frame)
    z_shift : 'float'
        z offset, default = None

    Returns
    -------
        xo, yo, zo of z-center
    """

    data = DIC_data.get_data(frame)
    sigma = data['sigma']
    Z = data['Z']
    Z[sigma == -1] = np.nan
    if z_shift is None:
        Zo = 0
    elif z_shift.lower().startswith('max'):
        Zo = np.nanmax(Z)
    elif z_shift.lower().startswith('min'):
        Zo = np.nanmin(Z)

    bad_pos = np.where(data['sigma'].flatten() == -1)
    X = np.delete(data['X'].flatten(), bad_pos)
    Y = np.delete(data['Y'].flatten(), bad_pos)
    Z = np.delete(data['Z'].flatten(), bad_pos)
    U = np.delete(data['U'].flatten(), bad_pos)
    V = np.delete(data['V'].flatten(), bad_pos)
    W = np.delete(data['W'].flatten(), bad_pos)
    Xi = X + U
    Yi = Y + V
    Zi = Z + W - Zo

    p0 = [0, 0, 0, 1]

    def fitfunc(p, coords):
        x0, y0, z0, R = p
        x, y, z = coords
        return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)

    errfunc = lambda p, x: fitfunc(p, x) - p[3]

    coords = np.vstack((Xi, Yi, Zi))

    (x0, y0, z0, R), flag = scipy.optimize.leastsq(errfunc, p0, args=(coords,))

    return x0, y0, Zo


def contour_overlay(DIC_data, frame, z_shift=None, xlim=None, ylim=None, zlim=None,
                major_spacing=None, minor_spacing=None, contour_width=1, contour_color='k', opacity=1.,
                colorbar_on=True, colorbar_location='right', colorbar_label=None, colorbar_lines=True,
                colorbar_ticks=None, colormap=None,
                font='Arial', fontsize_other=18, fontsize_colorbar=21,
                figsize=6, resolution=300, showfig=True, filename=None):
    """
    Plot (x, y, z) arrays in 'data' as contours overlaid on top of DIC image.
    Parameters
    ----------
    frame : 'int'
        DIC frame number.
    var : 'string'
        Varialbe to be plotted.
    xlim : 'array-like', len(xlim) == 2
        Upper and lower limits for the x-axis.
    ylim : 'array-like', len(ylim) == 2
        Upper and lower limits for the y-axis.
    zlim : 'array-like', len(ylim) == 2
        Upper and lower limits for the z-data.
    major_spacing : 'float'
        Spacing between major contours.
    minor_spacing :  'float'
        Spacing between minor contours.
    contour_width : 'int'
        Width of contour lines
    contour_color : 'string'
        Color of contour lines
    opacity : 'float'
        Opacity of contour plot (0 = transparent)
    colorbar_on : 'boole', default=True
        Show colorbar.
    colorbar_location : 'string'
        Location of colorbar with respect to contour_plot
    colorbar_label : 'string'
        Label for colorbar.
    font : 'String'
        Font to be used.
    fontsize_axes : 'Int'
        Font size to be used for axes labels.
    fontsize_other : 'Int'
        Font size to be used for all other text (legend, axis numbers, etc.).
    fontsize_colorbar : 'Int'
        Font size to be used for colorbar label.
    figsize : 'Tuple', default = '(8,6)'
        Width and height of figure
    resolution : 'Int', default = '300'
        DPI resolution of figure.
    showfig : 'Bool', default = 'True'
        Whether to show figure.
    filename : 'String', default = None.
        Name of file/path to save the figure to.
    """
    image = misc.imread(DIC_data.img[frame])
    ymax, xmax = image.shape[:2]

    mat = DIC_data.get_data(frame)
    sigma = mat['sigma']
    Z = mat['Z']
    Z[sigma == -1] = np.nan
    if z_shift is None:
        Zo = 0
    elif z_shift.lower().startswith('max'):
        Zo = np.nanmax(Z)
    elif z_shift.lower().startswith('min'):
        Zo = np.nanmin(Z)

    Zi = Z + mat['W'] - Zo

    u = mat['u']
    x = mat['x']
    v = mat['v']
    y = mat['y']
    u[sigma == -1.] = 0
    v[sigma == -1.] = 0
    x = x + u
    y = y + v

    y = -1*(y - ymax)
    z_m = ma.masked_invalid(Zi)

    a_ratio = image.shape
    a_ratio = a_ratio[1] / a_ratio[0]

    if isinstance(figsize, (int, float)):
        cbar_size = figsize/20
        figsize = (figsize*a_ratio, figsize)
    else:
        figsize = max(figsize)
        cbar_size = figsize/20
        figsize = (figsize*a_ratio, figsize)

    if zlim is None:
        cf_levels = np.linspace(np.nanmin(z), np.nanmax(z), 100)
        cl_levels = np.linspace(np.nanmin(z), np.nanmax(z), 10)
        l_levels = None
    else:
        if major_spacing is None:
            major_spacing = (zlim[1] - zlim[0]) / 10
        if minor_spacing is None:
            minor_spacing = major_spacing/10

        cl_levels = np.arange(zlim[0], zlim[1] + major_spacing, major_spacing)
        cf_levels = np.arange(zlim[0], zlim[1] + minor_spacing, minor_spacing)

        if colorbar_ticks is None:
            l_levels = cl_levels[::2]
        else:
            l_levels = (zlim[1] - zlim[0]) / colorbar_ticks
            l_levels = np.arange(zlim[0], zlim[1] + l_levels, l_levels)

    orientation = 'vertical'
    if colorbar_location in ['top', 'bottom']:
        orientation = 'horizontal'

    fig = mplt.figure(figsize=figsize, dpi=resolution)
    axis = fig.add_subplot(111)

    mplt.imshow(image, cmap=mplt.cm.gray, extent=[0, xmax, 0, ymax])

    cf = mplt.contourf(x, y, z_m, alpha=opacity, levels=cf_levels, extend='both')

    if contour_color is not None:
        cl = mplt.contour(cf, levels=cl_levels, colors=(contour_color,), linewidths=(contour_width,))

    if colormap is not None:
        cf.set_cmap(colormap)

    mpl.rcParams['font.sans-serif'] = font
    mpl.rcParams['pdf.fonttype'] = 42
    mplt.axes().set_aspect('equal')

    if xlim is not None:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    mplt.axis('off')

    if colorbar_on:
        divider = make_axes_locatable(axis)
        caxis = divider.append_axes(colorbar_location, size=cbar_size, pad=0.1)

        cbar = mplt.colorbar(cf, ticks=l_levels, cax=caxis, orientation=orientation, ticklocation=colorbar_location)
        cbar.ax.tick_params(labelsize=fontsize_other)

        if colorbar_label is not None:
            cbar.set_label(colorbar_label, size=fontsize_colorbar)

        if colorbar_lines:
            cbar.add_lines(cl)

    fig.tight_layout()

    if filename is not None:
        mplt.savefig(filename, dpi=resolution, transparent=True, bbox_inches='tight')

    if showfig:
        mplt.show()
    else:
        return fig, axis