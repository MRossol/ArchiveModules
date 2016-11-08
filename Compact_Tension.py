import numpy as np

__author__ = 'MNR'

__all__ = ["find_max_min_pos", "find_linear_fit", "nearest", "triangle_area",
           "riffle", "compact_tension", "virgin_ct", "healed_ct"]


def find_max_min_pos(data, x0, window=100):
    """
    Finds positions of local maximums and following minimums.
    Parameters
    ----------
    data : 'array_like', shape(data) = (n,2)
        (disp,load) data.
    x0 : 'Float'
        Guess for maximum location.
    window : 'Int', default = 100
        Size of window in which to search for max and min.
    """

    x_range = [x0 - window/2, x0 + window/2]
    if x_range[0] < np.min(data[:, 0]):
        x_range[0] = np.min(data[:, 0])
    if x_range[1] > np.max(data[:, 0]):
        x_range[1] = np.max(data[:, 0])

    pos_range = [nearest(data[:, 0], x) for x in x_range]

    # data_window = data[pos_range[0]:pos_range[1]]
    max_pos = pos_range[0] + data[pos_range[0]:pos_range[1], -1].argmax()
    min_pos = pos_range[0] + data[pos_range[0]:pos_range[1], -1].argmin()

    if max_pos > min_pos:
        max_pos = pos_range[0] + data[pos_range[0]:min_pos, -1].argmax()

    min_pos = max_pos + data[max_pos:pos_range[1], -1].argmin()

    return(max_pos, min_pos)


def find_linear_fit(data, max_pos, origin=True, horizontal=False):
    """
    Find linear fit.
    Parameters
    ----------
    data : 'array_like', shape(data) = (n,2)
        (disp,load) data.
    max_pos : 'Float'
        Position in data of maximum.
    origin : 'Boole'
        True - origin from first linear regime. False - find linear regime
        after maximum
    horizontal : 'Boole'
        Fit horizontal regime after load drop in healed samples.
    """

    fit = []

    if origin:
        for xo in range(max_pos - 5):
            linear_data = data[xo:max_pos + 1]
            x = linear_data[:, 0]
            y = linear_data[:, 1]
            A = np.vstack([x, np.ones(len(y))]).T
            model, resid = np.linalg.lstsq(A, y)[:2]

            r2 = 1 - resid / (y.size * y.var())
            if len(r2) > 0:
                fit.append(np.hstack((r2, model)).tolist())
    else:
        for xf in range(max_pos + 5, len(data)):
            linear_data = data[max_pos:xf]
            x = linear_data[:, 0]
            y = linear_data[:, 1]
            A = np.vstack([x, np.ones(len(y))]).T
            model, resid = np.linalg.lstsq(A, y)[:2]

            r2 = 1 - resid / (y.size * y.var())
            if len(r2) > 0:
                fit.append(np.hstack((r2, model)).tolist())

    fit = np.asarray(fit)
    if not horizontal:
        opt_fit = fit[np.argmin((1 - fit[:, 0])**2)]
    else:
        opt_fit = fit[np.argmin((0 - fit[:, 0])**2)]

    return opt_fit


def nearest(array, value):
    """
    Find position nearest to value in array.
    Parameters
    ----------
    array : 'array_like', shape(array) = (n,1)
        data array.
    value : 'Float'
        Value of sample to find.
    """
    return (np.abs(array - value)).argmin()


def triangle_area(maximum, minimum):
    """
    Calculate area under curve from maximum and minimum.
    Parameters
    ----------
    maximum : 'array_like', shape(maximum) = (1,3)
        (time, disp, load) at maximum.
    minimum : 'array_like', shape(minimum) = (1,3)
        (time, disp, load) at minimum.
    """
    return 1/2 * (minimum[1] * maximum[2]) - 1/2 * (minimum[1] * minimum[2])


def riffle(list1, list2):
    """
    Alternate entries from list1 and list2.
    Parameters
    ----------
    list1 : 'array_like'
        List 1 of data.
    list2 : 'array_like'
        List 2 of data.
    """
    return [item for sublist in zip(list1, list2) for item in sublist]


class compact_tension(object):
    def __init__(self, raw_data, load_time_data, linear_fit, shifted_data,
                 load_disp_data, maxima, minima, areas):
        """
        Create compact_tension class instance.
        Parameters
        ----------
        raw_data : 'array_like', shape(raw_data) = (n, 3)
            Raw (time, disp, load) data.
        load_time_data : 'array_like', shape(load_time_data) = (n, 2)
            (time, load) data.
        linear_fit : 'array_like', shape(linear_fit) = (n, 2) for virgin_ct,
        shape(linear_fit) = 3 for healed_ct w/
         shape(linear_fit[i]) = (n, 2)
            Fits to linear regions. virgin_ct = first linear region, healed_ct
            = [first linear region,
            post maximum region, end region post minimum].
        shifted_data : 'array_like', shape(shifted_data) = (n, 3)
            Shifted (time, disp, load) data to put fit linear region through
            origin.
        load_disp_data : 'array_like', shape(load_disp_data) = (n,2)
            Shifted (disp, load) data.
        maxima : 'array-like'
            List of shifted (time, disp, load) for all local maxima.
        minima : 'array-like'
            List of shifted (time, disp, load) for all local minima.
        areas : 'array-like'
            List of areas under each local maxima.
        """

        self.raw_data = raw_data
        self.load_time_data = load_time_data
        self.linear_fit = linear_fit
        self.shifted_data = shifted_data
        self.load_disp_data = load_disp_data
        self.maxima = maxima
        self.minima = minima
        self.areas = areas


class virgin_ct(compact_tension):
    def __init__(self, data, guesses, window=100):
        """
        Create compact_tension instance for virgin_ct sample.
        Parameters
        ----------
        data : 'array_like', shape(data) = (n, 3)
            Raw (time, disp, load) data.
        guesses : 'array_like'
            Guesses of displacement locations for local maxima.
        window : 'Int'
            Size of data window in which to locate local maxima and following
            minima.
        """

        assert data.shape[1] == 3, 'Data must equal (time, disp, load)'

        load_time_data = data[:, [0, 2]]

        if isinstance(guesses, int):
            guesses = [guesses, ]

        extrema = [find_max_min_pos(data[:, 1:], x0, window) for x0 in guesses]

        m, b = find_linear_fit(data[:, 1:], extrema[0][0])[1:]
        origin = - 1 * b / m
        linear_fit = np.dstack((data[:extrema[0][0] + 1, 1],
                               m * data[:extrema[0][0] + 1, 1] + b))[0]

        shifted_data = data - [0, origin, 0]
        load_disp_data = shifted_data[:, 1:]

        max_min = np.take(shifted_data, [pos for pos in extrema], axis=0)
        maxima = max_min[:, 0]
        minima = max_min[:, 1]

        areas = [triangle_area(maximum, minimum)
                 for (maximum, minimum) in list(zip(maxima, minima))]

        # call parent constructor with simplified arguments
        compact_tension.__init__(self, data, load_time_data, linear_fit,
                                 shifted_data, load_disp_data, maxima, minima,
                                 areas)


class healed_ct(compact_tension):
    def __init__(self, data):
        """
        Create compact_tension instance for healed_ct sample.
        Parameters
        ----------
        data : 'array_like', shape(data) = (n, 3)
            Raw (time, disp, load) data.
        """

        assert data.shape[1] == 3, 'Data must equal (time, disp, load)'

        load_time_data = data[:, [0, 2]]

        max_pos = data[:, -1].argmax()

        m, b = find_linear_fit(data[:, 1:], max_pos)[1:]
        origin = - 1 * b / m
        linear_fit = np.dstack((data[:max_pos + 1, 1],
                               m * data[:max_pos + 1, 1] + b))[0]

        shifted_data = data - [0, origin, 0]
        load_disp_data = shifted_data[:, 1:]

        maxima = [shifted_data[max_pos], ]

        end_pos = shifted_data[max_pos:, 1].argmax()
        min_m, min_b = find_linear_fit(shifted_data[:max_pos + end_pos, 1:],
                                       max_pos, origin=False)[1:]
        end_m, end_b = find_linear_fit(shifted_data[max_pos:, 1:], end_pos,
                                       horizontal=True)[1:]

        min_fit = np.dstack((shifted_data[max_pos:max_pos + end_pos + 1, 1],
                            min_m * shifted_data[max_pos:max_pos +
                            end_pos + 1, 1] + min_b))[0]
        end_fit = np.dstack((shifted_data[max_pos:max_pos + end_pos + 1, 1],
                             end_m * shifted_data[max_pos:max_pos +
                             end_pos + 1, 1] + end_b))[0]

        x_min = ((end_b - min_b)/(min_m - end_m))
        minima = [np.asarray([shifted_data[nearest(shifted_data[:, 1],
                  x_min), 0], x_min, min_m * x_min + min_b]), ]

        areas = [triangle_area(maximum, minimum)
                 for (maximum, minimum) in list(zip(maxima, minima))]

        # call parent constructor with simplified arguments
        compact_tension.__init__(self, data, load_time_data, (linear_fit,
                                 min_fit, end_fit), shifted_data,
                                 load_disp_data, maxima, minima, areas)
