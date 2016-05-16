__author__ = 'MNR'

__all__ = ['get_dB', 'get_Vt', 'detect_peaks', 'AE', 'AE_cont', 'AE_wavelets']

import os
import time
import datetime
import numpy as np
import pandas as pd
from scipy import ndimage
import itertools


def get_dB(volts, gain=40):
    """
    convert voltage w/ pre-gain to decibles
    Parameters
    ----------
    volts: 'float'
        input voltage
    gain: 'float'
        pre-gain

    Returns
    -------
    decibles (dB)
    """
    return (20 * np.log10(volts/10**(-6)) - gain)


def get_Vt(dB, gain=40):
    """
    convert threshold dB w/ pre-gain to voltage
    Parameters
    ----------
    dB: 'float'
        threshold decibles
    gain : 'float'
        pre-gain

    Returns
    -------
    threshold voltage
    """
    return 10**(-6) * 10**((dB+gain)/20)


def get_t_end(path):
    """
    Parameters
    ----------
    path : 'str' File path

    Returns
    ---------
    end time for instron data
    """
    with open(path) as f:
        next(f)
        te = next(f)
    te = te.strip().split('"')[1]
    te = datetime.datetime.strptime(te, "%A, %B %d, %Y %I:%M:%S %p").timestamp()
    return te


__author__ = "Marcos Duarte, https://github.com/demotu/BMC"
__version__ = "1.0.4"
__license__ = "MIT"


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


class AE(object):
    def __init__(self, wavelets, event_times, threshold, gain):
        """
        Initiate new class instance
        Parameters
        ----------
        wavelets : 'list'
            list of nx2 arrays of [time, v] for each wavelet
        event_times : 'array like'
            1D array of start time for each wavelet
        threshold : 'float'
            threshold in dB used to extract wavelets
        gain : 'float'
            pre-gain

        Returns
        -------
        self.wavelets : 'list'
            list of nx2 arrays of [time, v] for each wavelet
        self.event_times : 'array like'
            1D array of start time for each wavelet
        self.threshold : 'float'
            threshold in dB used to extract wavelets
        self.gain : 'float'
            pre-gain
        self.counts : 'array like'
            number of peaks (counts) within each wavelet
        self.amplitudes : 'array like'
            amplitude in dB of each wavelet
        self.rise_ts : 'array like'
            time to peak height (rise time) of wavelets
        self.durations : 'array like'
            duration of wavelet
        self.energies : 'array like'
            MARSE energy of wavelets
        self.data : 'dict like'
            Pandas Data Frame containing wavelets features
        """
        self.wavelets = wavelets
        self.event_times = event_times
        self.gain = gain
        self.threshold = threshold

        counts = []
        amplitudes = []
        rise_ts = []
        durations = []
        energies = []

        for wavelet in wavelets:
            peak_pos = detect_peaks(wavelet[:, 1], mph=get_Vt(threshold, gain=gain))

            counts.append(len(peak_pos))
            amplitudes.append(get_dB(np.max(wavelet[:, 1]), gain=gain))
            rise_ts.append(wavelet[np.argmax(wavelet[:, 1]), 0] - wavelet[peak_pos[0], 0])
            durations.append(wavelet[peak_pos[-1], 0] - wavelet[peak_pos[0], 0])
            MARSE = wavelet[peak_pos[0]:, 1]
            energies.append(np.sum(MARSE[MARSE>0]))

        self.counts = np.asarray(counts)
        self.amplitudes = np.asarray(amplitudes)
        self.rise_ts = np.asarray(rise_ts)
        self.durations = np.asarray(durations)
        self.energies = np.asarray(energies)

        self.data = pd.DataFrame({'counts': self.counts,
                                   'amplitudes': self.amplitudes,
                                   'rise_ts': self.rise_ts,
                                   'durations': self.durations,
                                   'energies': self.energies})

    def get_SS(self, file, percentage=True):
        """
        Import stress and strain data from file
        Parameters
        ----------
        file : 'string'
            files path for instron .csv
        percentage : 'boole'
            convert strain to percentage

        Returns
        -------
        self.strains : 'array like'
            contact extensometer strains
        self.stress : 'array like'
            instron stress
        self.data : 'dict like'
            update pandas data frame with stress and strains
        """
        instron_data = pd.read_csv(file, skiprows=8).values[:, [0, 3, 4]]
        te = get_t_end(file)

        instron_data[:, 0] = te + (instron_data[:, 0] - instron_data[-1, 0])

        strains = np.interp(self.event_times, instron_data[:, 0], instron_data[:, 1])

        if percentage:
            self.strains = strains*100
        else:
            self.strains = strains

        self.stresses = np.interp(self.event_times, instron_data[:, 0], instron_data[:, 2])

        self.data['strains'] = self.strains
        self.data['stresses'] = self.stresses

    def get_plot_data(self, keyword1, keyword2):
        """
        create data for plotting
        Parameters
        ----------
        keyword1 : 'string'
            Pandas data frame key
        keyword2 : 'string'
            Pandas data frame key

        Returns
        -------
        data_out : 'array like'
           array([keyword1_i, keyword2_i])
        """
        if keyword1.lower().startswith(('n','i')):
            data_out = np.dstack((self.data.index, self.data[keyword2]))[0]
        elif keyword2.lower().startswith(('n','i')):
            data_out = np.dstack((self.data[keyword1], self.data.index,))[0]
        else:
            data_out = np.dstack((self.data[keyword1], self.data[keyword2]))[0]

        return data_out


class AE_cont(AE):
    def __init__(self, file, threshold, PDT=100, HDT=200, HLT=300):


        self.path = file

        data = pd.read_csv(file, skiprows=3).values[:,0]
        with open(file) as f:
            info = [next(f) for _ in range(3)]

        time_stamp = info[0].split(',')[1].strip().split('.')
        s_time = datetime.datetime.strptime(time_stamp[0], "%m/%d/%Y %H:%M:%S").timestamp() + float('.'+time_stamp[1])
        gain = float(info[2].split(',')[1].strip())
        frequency = float(info[1].split(',')[1].strip())

        time = np.arange(len(data))/frequency
        waveform = np.dstack((time*10**6, data - np.mean(data)))[0]

        points = np.where(waveform[:, 1] >= get_Vt(threshold))[0]
        dt = np.diff(waveform[points, 0])
        events = dt <= (HDT + HLT)
        clusters, nclusters = ndimage.label(events)

        wavelets = []
        event_times = []
        for label in np.arange(1, nclusters+1):
            pos = np.where(clusters==label)[0][[0,-1]]
            if np.diff(pos)[0] >= (PDT*10**-6*frequency):
                start, stop = points[pos[0]], points[pos[1]+1] + int(np.round(HDT*10**-6*frequency)+1)
                wavelet = waveform[start:stop]
                event_times.append(wavelet[0, 0]*10**-6 + s_time)
                wavelet[:, 0] = wavelet[:, 0] - wavelet[0, 0]
                wavelets.append(wavelet)

        super().__init__(wavelets, np.asarray(event_times), threshold, gain)

    def export_data(self, new_path=None):
        output=[]
        for wavelet, e_time in zip(self.wavelets, self.event_times):
            output.append(np.dstack((np.ones(len(wavelet))*e_time, wavelet[:, 0], wavelet[:, 1]))[0])

        if new_path is None:
            new_path = self.path[:-4] + '.dat'

        np.savetxt(new_path, np.vstack(output))


class AE_wavelets(AE):
    def __init__(self, file, threshold, gain=40):
        data = np.loadtxt(file)

        event_times = np.unique(data[:, 0])
        wavelets = [data[np.where(data[:, 0] == event)[0], 1:] for event in event_times]

        super().__init__(wavelets, event_times, threshold, gain)