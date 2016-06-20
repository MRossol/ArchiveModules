__author__ = 'MNR'

__all__ = ["fit_plane", "nearest", "find_linear_fit", "shift_SS", "average_SS",
           "get_mat_numbs", "get_t_end", "get_t_start", "DIC_3D", "DIC_2D"]

import os
import datetime
import scipy.io
import scipy.interpolate
from scipy import misc
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as mplt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def fit_plane(data):
    """
    Find plane fit (z = a x + b y + c) for (x, y, z) data.
    Parameters
    ----------
    data : 'array_like', shape(data) = (n,3)
        (disp,load) data.

    Returns
    ----------
    a, b, c
    """

    assert data.shape[1] == 3, "Data is not an nx3 array."
    x, y, z = (data[:, 0], data[:, 1], data[:, 2])
    A = np.vstack([x, y, np.ones(len(z))]).T
    return np.linalg.lstsq(A, z)[0]


def nearest(array, value):
    """
    Find nearest position in array to value.
    Parameters
    ----------
    array : 'array_like'
    value: 'float', 'list'

    Returns
    ----------
    Searches n x 1 and n x m x 1  arrays for floats.
    Searches n x m and n x m x p arrays for lists of size m or p.
    """

    if isinstance(array, (list, tuple)):
        array = np.asarray(array)

    if isinstance(value, (float, int)):
        pos = (np.abs(array - value)).argmin()
        if len(array.shape) == 2:
            return np.unravel_index(pos, array.shape)
        else:
            return pos
    else:
        pos = (np.sum((array - value)**2, axis=1)**(1/2)).argmin()
        if len(array.shape) == 3:
            return np.unravel_index(pos, array.shape)
        else:
            return pos


def find_linear_fit(data, max_pos):
    """
    Find optimum linear fit by maximizing R^2.
    Parameters
    ----------
    data : 'array_like', shape(data) = (n,2)
        (disp,load) data.
    max_pos : 'Float'
        Position in data of maximum.

    Returns
    ----------
    (R^2, m, b)
    """

    fit = []

    for xo in range(max_pos-2):
        linear_data = data[xo:max_pos + 1]
        x = linear_data[:, 0]
        y = linear_data[:, 1]
        A = np.vstack([x, np.ones(len(y))]).T
        model, resid = np.linalg.lstsq(A, y)[:2]

        r2 = 1 - resid / (y.size * y.var())
        if len(r2) > 0:
            fit.append(np.hstack((r2, model)).tolist())

    fit = np.asarray(fit)
    opt_fit = fit[np.argmin((1 - fit[:, 0])**2)]

    return opt_fit


def shift_SS(data, e1=0.1, e2=0.3):
    """
    Shifts stress strain data to pass through origin, using linear fit of elastic region.
    ----------
    data : 'array_like', shape(data) = (n,2)
        (disp,load) data.
    e1 : 'float'
    e2 : 'float'

    Returns
    ----------
    shifted stress strain curve passing through the origin
    """

    m, b = find_linear_fit(data, nearest(data[:, 0], e2))[1:]
    origin = -1*(b/m)
    shift_data = data - [origin, 0]

    e1Pos = nearest(shift_data[:, 0], e1)
    e2Pos = nearest(shift_data[:, 0], e2)
    fitData = shift_data[e1Pos:e2Pos]
    m, b = fit_line(fitData)
    origin = -1*(b/m)
    return np.vstack((np.zeros((1,2)), shift_data - [origin, 0]))


def average_SS(data, step=0.01, axial=True, clean=False):
    """
    Find average of multiple Stress Strain curves. Options to average (clean) identical strain entries,
    and to average axial (True) and transverse (False) Stress Strain data.
    Parameters
    ----------
    data : 'list' of nx2 arrays
    step : 'float'
    axial : 'boole'
    clean : 'boole'

    Returns
    ----------
    average stress-strain curve of given input curves
    """

    strainRange = ([sample[0, 0] for sample in data], [sample[-1, 0] for sample in data])

    if axial:
        strainRange = np.arange(np.max(strainRange[0]),
                            np.min(strainRange[1]), step)
    else:
        strainRange = np.arange(-1 * np.min(strainRange[0]),
                            -1 * np.max(strainRange[1]), step) * -1.

    smooth = []
    for sample in data:
        if clean:
            strains = np.unique(sample[:, 0])
            clean = []
            for strain in strains:
                clean.append(np.mean(sample[sample[:, 0] == strain], axis=0))
            sample = np.asarray(clean)

        interp = scipy.interpolate.interp1d(sample[:, 0], sample[:, 1])
        smooth.append(np.dstack((strainRange, interp(strainRange)))[0])

    return np.mean(np.array(smooth), axis=0)


def fit_line(data):
    """
    Find linear fit (y = n x + b) for (x, y) data.
    Parameters
    ----------
    data : 'array_like', shape(data) = (n,2)
        (disp,load) data.

    Returns
    ----------
    m, b
    """

    assert data.shape[1] == 2, "Data is not an nx2 array."
    x=data[:, 0]
    y=data[:, 1]
    A = np.vstack([x, np.ones(len(y))]).T
    return np.linalg.lstsq(A, y)[0]


def chord_modulus(data, e1=0.1, e2=0.3):
    """
    Find Youngs modulus from data between e1 and e2
    Parameters
    ----------
    data : 'array_like', shape(data) = (n,2)
        (disp,load) data.
    e1 : 'float'
    e2 : 'float'

    Returns
    ----------
    E = m of linear_fit between e1 and e2
    """
    assert data.shape[1] == 2, "Data is not an nx2 array."
    e1Pos = nearest(data[:, 0], e1)
    e2Pos = nearest(data[:, 0], e2)
    fitData = data[e1Pos:e2Pos]
    E, so = fit_line(fitData)
    return E


def get_t_start(path):
    """
    Parameters
    ----------
    path : 'str' File path

    Returns
    ----------
    start time for DIC images
    """
    with open(path) as f:
        ts = next(f)
    ts = ts.strip()[11:].split('.')
    ts = datetime.datetime.strptime(ts[0].strip(), "%m/%d/%Y %H:%M:%S").timestamp() + float('.'+ ts[1])
    return ts


def get_t_end(path):
    """
    Parameters
    ----------
    path : 'str' File path

    Returns
    ----------
    end time for instron data
    """
    with open(path) as f:
        next(f)
        te = next(f)
    te = te.strip().split('"')[1]
    te = datetime.datetime.strptime(te, "%A, %B %d, %Y %I:%M:%S %p").timestamp()
    return te


def get_mat_numbs(files, version='3D'):
    """
    get list of image #s corresponding to .mat files
    Parameters
    ----------
    files : 'list'
        list of .mat files
    version : 'string'
        2D or 3D DIC

    Returns
    -------
    list of [img_numbs]
    """
    numbs = []
    for file in files:
        if version is '3D':
            numbs.append(int(file.split("_")[-2]))
        else:
            numbs.append(int(file.split("_")[-1][:-4]))

    return np.asarray(numbs)


class DIC_3D(object):
    def __init__(self, path):
        """
        Create new class instance and import DIC .mat files and image files.
        Parameters
        ----------
        path : 'sting'
            Path to DIC files directory.


        Returns
        ---------
        self.path : path to DIC data
        self.mat : list of .mat files
        self.img : list of .tiff files
        """
        self.path = path

        mat_files = []
        img_files = []
        for filename in os.listdir(path):
            if filename.endswith('.mat') and not filename.startswith('.'):
                mat_files.append(os.path.join(path, filename))
            elif filename.endswith('_0.tif') or filename.endswith('_0.tiff') and not filename.startswith('.'):
                img_files.append(os.path.join(path, filename))

        if len(mat_files) == 0:
            for filename in os.listdir(os.path.join(path, 'v6 files')):
                if filename.endswith('.mat') and not filename.startswith('.'):
                    mat_files.append(os.path.join(path, filename))

        if len(mat_files) != len(img_files):
            mat_num = [file.replace('.mat', '') for file in mat_files]
            img_type = img_files[0].split('.')[-1]
            img_num = [file.replace('.' + img_type, '') for file in img_files]
            img_files = [img_files[img_num.index(num)] for num in mat_num]

        self.mat = sorted(mat_files)
        self.img = sorted(img_files)

    def get_data(self, frame):
        """
        import DIC data from .mat files
        Parameters
        ----------
        frame : 'int'
            DIC frame

        Returns
        -------
        dictionary of DIC data from given .mat file
        """
        return scipy.io.loadmat(self.mat[frame])

    def get_mag(self, frame = 0):
        """
        Calculate DIC magnification (pixel/mm, mm/pixel).
        Parameters
        ----------
        frame : 'int', default=0
            DIC frame number.

        Returns
        ----------
        pixel/mm, mm/pixel magnifications
        """
        data = self.get_data(frame)
        sigma = np.where(data['sigma'].flatten() == -1.)
        xy_data = np.dstack((data['x'].flatten(), data['y'].flatten(), data['X'].flatten(), data['Y'].flatten()))[0]
        xy_data = np.delete(xy_data, sigma, axis=0)
        pixel_dist = np.linalg.norm(xy_data[0, :2] - xy_data[-1, :2])
        metric_dist = np.linalg.norm(xy_data[0, 2:] - xy_data[-1, 2:])

        return pixel_dist / metric_dist, metric_dist / pixel_dist

    def get_hstep(self):
        """
        Returns
        -------
        step size in pixels
        """
        data = self.get_data(0)
        x = data['x']
        y = data['y']

        return np.mean([np.mean(x[0, 1:] - x[0, :-1]), np.mean(y[1:, 0] - y[:-1, 0])])

    def get_error(self, variables = 'Metric'):
        """
        Parameters
        ----------
        variables : 'string'
            Analyze Matrix or Pixel displacements

        Returns
        -------
        error in [U, V, W] ('Metric) or [u, v] ('Pixel')
        """

        data = self.get_data(1)
        bad_pos = np.where(data['sigma'].flatten() == -1)

        if variables.lower().startswith('m'):
            U = np.delete(data['U'].flatten(), bad_pos)
            V = np.delete(data['V'].flatten(), bad_pos)
            W = np.delete(data['W'].flatten(), bad_pos)

            error = [(np.mean(var), np.std(var)) for var in [U, V, W]]
        else:
            u = np.delete(data['u'].flatten(), bad_pos)
            v = np.delete(data['v'].flatten(), bad_pos)

            error = [(np.mean(var), np.std(var)) for var in [u, v]]

        return np.asarray(error)

    def get_strains(self, method='Plane', units='Metric'):
        """
        Calculate mean strains (exx, eyy, exy)
        Parameters
        ----------
        method : 'string', default='Plane'
            method for strain calculation: Plane fits displacements to planes; Average takes mean of exx, eyy, exy.
        units : 'string', default='Metric'
            Units used to calculate strains, pixels or metric.

        Returns
        ----------
        Array of [exx_i, eyy_i, exy_i]
        """
        strains = []
        for file in self.mat:
            data = scipy.io.loadmat(file)

            sigma = np.where(data['sigma'].flatten() == -1.)
            if method.lower().startswith('p'):
                if units.lower().startswith('m'):
                    disp_data = np.dstack((data['X'].flatten(), data['Y'].flatten(), data['U'].flatten(),
                                           data['V'].flatten()))[0]
                elif units.lower().startswith('p'):
                    disp_data = np.dstack((data['x'].flatten(), data['y'].flatten(), data['u'].flatten(),
                                           data['v'].flatten()))[0]

                disp_data = np.delete(disp_data, sigma, axis=0)

                u_data = disp_data[:, :-1]
                v_data = disp_data[:, [0, 1, 3]]

                dudx, dudy = fit_plane(u_data)[:-1]
                dvdx, dvdy = fit_plane(v_data)[:-1]

                strains.append([dudx, dvdy, (dvdx + dudy)/2])
            else:
                strain_data = np.dstack((data['exx'].flatten(), data['eyy'].flatten(), data['exy'].flatten()))[0]
                strain_data = np.delete(strain_data, sigma, axis=0)

                strains.append(np.mean(strain_data, axis=0))

        return np.asarray(strains)

    def get_mat_time(self):
        """
        get global time for each DIC data point
        Returns
        -------
        Array of [t_mat]
        """
        for filename in os.listdir(self.path):
            if filename.endswith('.dat'):
                ts = get_t_start(os.path.join(self.path, filename))
                img_time = np.loadtxt(os.path.join(self.path, filename), skiprows=2)[:, 1] + ts

        return img_time[get_mat_numbs(self.mat)]

    def get_img_stress(self):
        """
        Determines stress for each DIC img
        Returns
        -------
        Array of stress
        """
        for filename in os.listdir(self.path):
            if filename.endswith('.csv'):
                instron_data = pd.read_csv(os.path.join(self.path, filename), skiprows=8).values[:,[0,4]]
                te = get_t_end(os.path.join(self.path, filename))

        instron_data[:, 0] = te + (instron_data[:, 0] - instron_data[-1, 0])
        mat_time = self.get_mat_time()

        return np.interp(mat_time, instron_data[:, 0], instron_data[:, 1])

    def get_contourData(self, frame, var, coordinates="Metric"):
        """
        Extracts contour data
        Parameters
        ----------
        frame : 'int'
            DIC frame number.
        var : 'string'
            Variable to be plotted.
        coordinates : 'string', default='Metric'
            X, Y units.

        Returns
        ----------
        arrays of x_i, y_i, var for given frame.
        x_i = x + u or X + U
        y_i = y + v or Y + V
        """
        mat = self.get_data(frame)

        if var.lower().startswith('zi'):
            data = mat['Z'] + mat['W']
        else:
            data = mat[var]

        sigma = mat['sigma']

        if var.lower().startswith('u') or var.lower().startswith('v'):
            data = data - np.mean(np.delete(data.flatten(),np.where(sigma.flatten() == -1.)))

        data[sigma == -1.] = np.nan

        if coordinates.lower().startswith('p'):
            u = mat['u']
            x = mat['x']
            v = mat['v']
            y = mat['y']
        else:
            u = mat['U']
            x = mat['X']
            x[sigma == -1.] = 0
            v = mat['V']
            y = mat['Y']
            y[sigma == -1.] = 0

        u[sigma == -1.] = 0
        v[sigma == -1.] = 0
        x = x + u
        y = y + v

        return x, y, data

    def contour_overlay(self, frame, var, xlim=None, ylim=None, zlim=None,
                    major_spacing=None, minor_spacing=None, contour_width=1, contour_color='k', opacity=1.,
                    colorbar_on=True, colorbar_location='right', colorbar_label=None, colorbar_lines=True,
                    colorbar_ticks=None, colormap=None,
                    font='Arial', fontsize_other=18, fontsize_colorbar=21,
                    figsize=6, resolution=300, showfig=True, filename=None):
        """
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

        Returns
        ----------
        Plot (x, y, z) arrays in 'data' as contours overlaid on top of DIC image.
        """
        image = misc.imread(self.img[frame])
        ymax, xmax = image.shape[:2]

        x, y, z = self.get_contourData(frame, var, coordinates='Pixels')
        y = -1*(y - ymax)
        z_m = ma.masked_invalid(z)

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

        cf = mplt.contourf(x, y, z_m, alpha=opacity, levels=cf_levels, extend='both', antialiased=True)

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


class DIC_2D(object):
    def __init__(self, path):
        """
        Create new class instance and import DIC .mat files and image files.
        Parameters
        ----------
        path : 'sting'
            Path to DIC files directory.


        Returns
        ---------
        self.path : path to DIC data
        self.mat : list of .mat files
        self.img : list of .tiff files
        """
        self.path = path

        mat_files = []
        img_files = []
        for filename in os.listdir(path):
            if filename.endswith('.mat') and not filename.startswith('.'):
                mat_files.append(os.path.join(path, filename))
            elif filename.endswith('.tif') or filename.endswith('.tiff') and not filename.startswith('.'):
                img_files.append(os.path.join(path, filename))

        if len(mat_files) == 0:
            for filename in os.listdir(os.path.join(path, 'v6 files')):
                if filename.endswith('.mat') and not filename.startswith('.'):
                    mat_files.append(os.path.join(path, filename))

        self.mat = sorted(mat_files)
        self.img = sorted(img_files)

    def get_data(self, frame):
        """

        Parameters
        ----------
        frame : 'int'
            DIC frame

        Returns
        -------
        dictionary of DIC data from given .mat file
        """
        return scipy.io.loadmat(self.mat[frame])

    def get_hstep(self):
        """
        Returns
        -------
        step size in pixels
        """
        data = scipy.io.loadmat(self.mat[0])
        x = data['x']
        y = data['y']

        return np.mean([np.mean(x[0, 1:] - x[0, :-1]), np.mean(y[1:, 0] - y[:-1, 0])])

    def get_error(self, variables = 'Pixel'):
        """
        Parameters
        ----------
        variables : 'string'
            Analyze Matrix or Pixel displacements

        Returns
        -------
        error in [u_c, v_c] ('Metric) or [u, v] ('Pixel')
        """

        data = self.get_data(1)
        bad_pos = np.where(data['sigma'].flatten() == -1)

        if variables.lower().startswith('p'):
            u = np.delete(data['u'].flatten(), bad_pos)
            v = np.delete(data['v'].flatten(), bad_pos)

            error = [(np.mean(var), np.std(var)) for var in [u, v]]
        else:
            U = np.delete(data['u_c'].flatten(), bad_pos)
            V = np.delete(data['v_c'].flatten(), bad_pos)

            error = [(np.mean(var), np.std(var)) for var in [U, V]]

        return np.asarray(error)

    def get_strains(self, method='Plane', units='Pixels'):
        """
        Calculate mean strains (exx, eyy, exy)
        Parameters
        ----------
        method : 'string', default='Plane'
            method for strain calculation: Plane fits displacements to planes; Average takes mean of exx, eyy, exy.
        units : 'string', default='Metric'
            Units used to calculate strains, pixels or metric.

        Returns
        ----------
        Array of [exx_i, eyy_i, exy_i]
        """
        strains = []
        for file in self.mat:
            data = scipy.io.loadmat(file)

            sigma = np.where(data['sigma'].flatten() == -1.)
            if method.lower().startswith('p'):
                if units.lower().startswith('p'):
                    disp_data = np.dstack((data['x'].flatten(), data['y'].flatten(), data['u'].flatten(), data['v'].flatten()))[0]
                else:
                    disp_data = np.dstack((data['x_c'].flatten(), data['y_c'].flatten(), data['u_c'].flatten(), data['v_c'].flatten()))[0]


                disp_data = np.delete(disp_data, sigma, axis=0)

                u_data = disp_data[:,:-1]
                v_data = disp_data[:,[0,1,3]]

                dudx, dudy = fit_plane(u_data)[:-1]
                dvdx, dvdy = fit_plane(v_data)[:-1]

                strains.append([dudx, dvdy, (dvdx + dudy)/2])
            else:
                strain_data = np.dstack((data['exx'].flatten(), data['eyy'].flatten(), data['exy'].flatten()))[0]
                strain_data = np.delete(strain_data, sigma, axis=0)

                strains.append(np.mean(strain_data, axis=0))

        return np.asarray(strains)

    def get_img_stress(self, area):
        """
        Determines stress for each DIC img
        Image load determined from image number and .dat file
        Parameters
        ----------
        area : 'float;
            samples area
        Returns
        -------
        Array of stress
        """
        for filename in os.listdir(self.path):
            if filename.endswith('.dat'):
                data = np.loadtxt(os.path.join(self.path, filename), skiprows=3)[:, [3, 2]] #[Time(s), Load (N)]
                data[:, 0] = data[:, 0] + get_t_start(os.path.join(self.path, filename))

        img_times = [os.path.getmtime(file) for file in self.img]
        load = [data[nearest(data[:, 0], img_t), 1] for img_t in img_times]

        return np.asarray(load)/area

    def get_stress(self, area):
        """
        Determines stress for each DIC img
        Image load determined from image time and .dat file
        Parameters
        ----------
        area : 'float;
            samples area
        Returns
        -------
        Array of stress
        """
        for filename in os.listdir(self.path):
            if filename.endswith('.dat'):
                data = np.loadtxt(os.path.join(self.path, filename), skiprows=1) #[Data Point, Disp (mm), Load (N), Time (s)]

        mat_numbs = get_mat_numbs(self.mat, version='2D')
        load = data[mat_numbs, 2]

        return load/area

    def get_contourData(self, frame, var, coordinates="Metric"):
        """
        Extracts contour data
        Parameters
        ----------
        frame : 'int'
            DIC frame number.
        var : 'string'
            Variable to be plotted.
        coordinates : 'string', default='Metric'
            X, Y units.

        Returns
        ----------
        arrays of x_i, y_i, var for given frame.
        x_i = x + u or x_c + u_c
        y_i = y + v or y_c + v_c
        """
        mat = scipy.io.loadmat(self.mat[frame])

        data = mat[var]
        sigma = mat['sigma']
        if var.startswith('u') or var.startswith('v') or var.startswith('u_c') or var.startswith('v_c'):
            data = data - np.mean(np.delete(data.flatten(),np.where(sigma.flatten() == -1.)))
        data[sigma == -1.] = np.nan

        if coordinates.lower().startswith('p'):
            u = mat['u']
            x = mat['x']
            v = mat['v']
            y = mat['y']
        else:
            u = mat['u_c']
            x = mat['x_c']
            x[sigma == -1.] = 0
            v = mat['v_c']
            y = mat['y_c']
            y[sigma == -1.] = 0

        u[sigma == -1.] = 0
        v[sigma == -1.] = 0
        x = x + u
        y = y + v

        return x, y, data

    def contour_overlay(self, frame, var, xlim=None, ylim=None, zlim=None,
                    major_spacing=None, minor_spacing=None, contour_width=1, contour_color='k', opacity=1.,
                    colorbar_on=True, colorbar_location='right', colorbar_label=None, colorbar_lines=True,
                    colorbar_ticks=None, colormap=None,
                    font='Arial', fontsize_other=18, fontsize_colorbar=21,
                    figsize=6, resolution=300, showfig=True, filename=None):
        """
        Parameters
        ----------

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

        Returns
        ----------
        Plot (x, y, z) arrays in 'data' as contours overlaid on top of DIC image.
        """
        image = misc.imread(self.img[frame])
        ymax, xmax = image.shape[:2]

        x, y, z = self.get_contourData(frame, var, coordinates = 'Pixels')
        y = -1*(y - ymax)
        z_m = ma.masked_invalid(z)

        a_ratio = image.shape
        a_ratio = a_ratio[1]/a_ratio[0]

        if isinstance(figsize, (int, float)):
            cbar_size = figsize/20
            figsize = (figsize*a_ratio, figsize)
        else:
            figsize = max(figsize)
            cbar_size = figsize/20
            figsize = (figsize*a_ratio, figsize)

        if zlim is None:
            cf_levels = 100
            cl_levels = 10
            l_levels = None
        else:
            if major_spacing is None:
                major_spacing = (zlim[1] - zlim[0])/10
            if minor_spacing is None:
                minor_spacing = major_spacing/10

            cl_levels = np.arange(zlim[0], zlim[1] + major_spacing, major_spacing)
            cf_levels = np.arange(zlim[0], zlim[1] + minor_spacing, minor_spacing)

            if colorbar_ticks is None:
                l_levels = cl_levels[::2]
            else:
                l_levels = (zlim[1] - zlim[0])/colorbar_ticks
                l_levels = np.arange(zlim[0], zlim[1] + l_levels, l_levels)

        orientation = 'vertical'
        if colorbar_location in ['top', 'bottom']:
            orientation = 'horizontal'

        fig = mplt.figure(figsize=figsize, dpi=resolution)
        axis = fig.add_subplot(111)

        mplt.imshow(image, cmap=mplt.cm.gray, extent=[0, xmax, 0, ymax])

        cf = mplt.contourf(x, y, z_m, alpha=opacity, levels=cf_levels, extend='both', antialiased=True)

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
            return (fig, axis)