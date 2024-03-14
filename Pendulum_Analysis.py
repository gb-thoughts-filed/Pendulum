import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

class PendulumData:

    def __init__(self, text_file: str):
        self.frames = None
        self.angles = None
        self.times = None
        self.text_file_ingestion(text_file)

        self.frames = list(self.frames)
        self.angles = list(self.angles)
        self.times = list(self.times)

        for i in np.arange(len(self.frames)):
            self.angles[i] = float(self.angles[i])
            self.times[i] = float(self.times[i])

        self.angle_uncertainties = None
        self.time_uncertainties = None
        self.uncertainty_population(len(self.angles))

        self.angles

        self.angle_positive_peaks_indices = sp.signal.find_peaks(self.angles)[0].tolist()
        self.angle_positive_peaks = [self.angles[x] for x in self.angle_positive_peaks_indices]
        self.average_positive_peaks = np.average(self.angle_positive_peaks)
        self.angle_positive_peak_times = [self.times[x] for x in self.angle_positive_peaks_indices]
        self.positive_periods, self.positive_periods_uncertainty = self.period_calculation(self.angle_positive_peaks_indices)

        self.positive_peak_indices_with_negative_sign = self.positive_opposite_sign_peak_removal(self.angle_positive_peaks)

        self.angles_reverse_sign = [(-1)*x for x in self.angles]
        self.angle_negative_peaks_indices = sp.signal.find_peaks(self.angles_reverse_sign)[0].tolist()
        self.angle_negative_peaks = [self.angles[x] for x in self.angle_negative_peaks_indices]
        self.average_negative_peaks = np.average(self.angle_negative_peaks)
        self.angle_negative_peak_times = [self.times[x] for x in self.angle_negative_peaks_indices]
        self.negative_periods, self.negative_periods_uncertainty = self.period_calculation(self.angle_negative_peaks_indices)

        self.negative_peak_indices_with_positive_sign = self.negative_opposite_sign_peak_removal(self.angle_negative_peaks)



        for j in self.positive_peak_indices_with_negative_sign:
            self.angle_positive_peaks_indices.remove(self.angle_positive_peaks_indices[j])
            self.angle_positive_peaks.remove(self.angle_positive_peaks[j])
            self.angle_positive_peak_times.remove(self.angle_positive_peak_times[j])

        for k in self.negative_peak_indices_with_positive_sign:
            self.angle_negative_peaks_indices.remove(self.angle_negative_peaks_indices[k])
            self.angle_negative_peaks.remove(self.angle_negative_peaks[k])
            self.angle_negative_peak_times.remove(self.angle_negative_peak_times[k])

        self.quick_graph()




    def text_file_ingestion(self, text_file:str) -> None:

        self.times, self.angles, self.frames = np.loadtxt(text_file,
                                                          skiprows=2, delimiter=",", dtype=str, unpack=True)

        return



    def uncertainty_population(self, list_length: int) -> None:

        x = np.arange(list_length, dtype=int)
        self.angle_uncertainties = np.full_like(x, 0.05, dtype=float)
        self.time_uncertainties = np.full_like(x, 0.0005, dtype=float)

        return

    def period_calculation(self, angle_peaks_indices: list) -> list:
        periods = []
        periods_uncertainty = []

        for i in np.arange(len(angle_peaks_indices)):
            if i != len(angle_peaks_indices) - 1:
                periods.append(self.times[angle_peaks_indices[i+1]] - self.times[angle_peaks_indices[i]])
                periods_uncertainty.append(
                    np.sqrt(self.time_uncertainties[angle_peaks_indices[i + 1]]**2 + self.time_uncertainties[angle_peaks_indices[i]]**2)
                )
        return periods, periods_uncertainty



    def positive_opposite_sign_peak_removal(self, peaks) -> list:
        opposite_signed_points = []

        for i in np.arange(len(peaks)):
            if peaks[i] < 0:
                opposite_signed_points.append(i)

        opposite_signed_points.reverse()

        return opposite_signed_points

    def negative_opposite_sign_peak_removal(self, peaks) -> list:
        opposite_signed_points = []

        for i in np.arange(len(peaks)):
            if peaks[i] > 0:
                opposite_signed_points.append(i)

        opposite_signed_points.reverse()

        return opposite_signed_points

    def quick_graph(self):

        plt.errorbar(self.times, self.angles, marker="o", ls='', label="Raw Data")
        plt.errorbar(self.angle_positive_peak_times, self.angle_positive_peaks, marker="o", ls='', label="Raw Data Positive Peaks")
        plt.errorbar(self.angle_negative_peak_times, self.angle_negative_peaks, marker="o", ls='',
                     color="green", label="Raw Data Negative Peaks")

def linear(x, a, b):
    return a*x + b

def non_linear(x, c, d, e):
    return c*(x)**d + e

def log(x, c, d, e):
    return c*np.log(d*x) + e

def exponential(x, c, d, e):
    return np.exp(c*(x+d)) + e


def reduced_chi_squared(data_ys, predicted_ys, uncertainties, num_elements, num_parameters):

    dof = num_elements - num_parameters
    residuals = np.square(data_ys - predicted_ys)
    squared_uncertainties = np.square(uncertainties)
    red_chi_squared = (1/dof)*(np.sum(residuals/squared_uncertainties))

    return red_chi_squared

def return_peaks_twenties(peaks: list):
    twenties = []
    for i in peaks:
        if  20 < i < 23:
            twenties.append(i)
    return twenties






