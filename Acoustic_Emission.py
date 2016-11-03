__author__ = 'Michael Rossol'

__all__ = ['get_dB', 'get_Vt', 'get_t_end', 'AE', 'AE_cont', 'AE_wavelets']

import datetime
import numpy as np
import pandas as pd
from scipy import ndimage
from detect_peaks import detect_peaks


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

        self.data = pd.DataFrame({
            'event_times': self.event_times,
            'counts': self.counts,
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
        """
        Extracts AE wavelets from continuous AE waveform and initiates AE instance
        Parameters
        ----------
        file : 'string;
            file path for .csv containing continuous waveform data
        threshold : 'float'
            threshold for AE event in dB
        PDT : 'float'
            Peak Definition Time = minimum wavelet length in us
        HDT : 'float'
            Hit Definition Time = time till end of wavelet in us after last count
        HLT : 'float'
            Hit Lag Time = minimum time between events
        gain : 'float'
            pre-gain in dB

        Returns
        -------
        self.path : 'string'
            .csv file path
        """

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

        points = np.where(waveform[:, 1] >= get_Vt(threshold, gain=gain))[0]
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

        AE.__init__(self, wavelets, np.asarray(event_times), threshold, gain)

    def export_data(self, filename=None):
        """
        export AE wavelets to .dat
        Parameters
        ----------
        new_path : 'string'
            new filename, default is .csv filename

        Returns
        -------
        """
        output=[]
        for wavelet, e_time in zip(self.wavelets, self.event_times):
            output.append(np.dstack((np.ones(len(wavelet))*e_time, wavelet[:, 0], wavelet[:, 1]))[0])

        if filename is None:
            filename = self.path[:-4] + '.dat'

        np.savetxt(filename, np.vstack(output))


class AE_wavelets(AE):
    def __init__(self, file, threshold, gain=40):
        """
        Loads AE wavelets from .dat file and initiates AE instance
        Parameters
        ----------
        file : 'string'
            .dat filename
        threshold : 'float'
            threshold in dB used to determine AE event
        gain : 'float'
            Pre-gain in dB

        Returns
        -------
        """
        data = np.loadtxt(file)

        event_times = np.unique(data[:, 0])
        wavelets = [data[np.where(data[:, 0] == event)[0], 1:] for event in event_times]

        AE.__init__(self, wavelets, event_times, threshold, gain)