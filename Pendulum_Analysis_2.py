import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

class PendulumData:

    def __init__(self, text_file: str, mass_length, angle_span, points_angle_span):

        # Main data points from text file extracted.
        self.frames = None
        self.angles = None
        self.times = None
        self.xs = None
        self.ys =None
        self.angle_span = angle_span
        self.points_angle_span = points_angle_span
        self.mass_length = mass_length
        self.text_file_str = text_file
        self.text_file_ingestion(text_file)


        # Converts numpy arrays into lists.

        self.frames = list(self.frames)
        self.angles = list(self.angles)
        self.times = list(self.times)
        self.xs = list(self.xs)
        self.ys = list(self.ys)

        # Converts csv list entries from strings to floats

        for i in np.arange(len(self.frames)):
            self.angles[i] = float(self.angles[i])
            self.times[i] = float(self.times[i])
            self.xs[i] = float(self.xs[i])
            self.ys[i] = float(self.ys[i])

        # Xs and Ys are divided by 1000 to reflect mass length measurement in mm.

        self.xs = [i/1000 for i in self.xs]
        self.ys = [i/1000 for i in self.ys]

        # Xs and Ys uncertainties created and populated to match the precision of vernier calipers.

        self.xs_uncertainties = uncertainty_population(len(self.xs), (mass_length/1000)*8)
        self.ys_uncertainties = uncertainty_population(len(self.ys), (mass_length/1000)*8)

        # Angle uncertainties calculated based on x and y measurements.
        self._angle_distance = [np.abs(self.angles[i+1]-self.angles[i]) for i in np.arange(len(self.angles) - 1)]
        self._average_angle_distance = np.average(self._angle_distance)
        self._max_angle_distance = np.max(self._angle_distance)
        #print(self._max_angle_distance)
        #self.angle_uncertainties = angle_uncertainty_calculation(self.xs, self.xs_uncertainties, self.ys, self.ys_uncertainties)
        self.angle_uncertainties = uncertainty_population(len(self.angles), self.angle_span/2)

        # Time uncertainties populated.
        self.time_uncertainties = uncertainty_population(len(self.times), 0.02*self.points_angle_span/2)

        # Conversion of angles and angle uncertainties into radians
        self.angles = [i*(np.pi/180) for i in self.angles]
        self.angle_uncertainties = [i*(np.pi/180) for i in self.angle_uncertainties]

        # Positive peak identification.

        self.angle_positive_peaks_indices = sp.signal.find_peaks(self.angles)[0].tolist()
        self.angle_positive_peaks = [self.angles[x] for x in self.angle_positive_peaks_indices]

        # Average of the positive peaks.
        self.average_positive_peaks = np.average(self.angle_positive_peaks)

        # Positive peak times, that correspond with the positive peaks.
        self.angle_positive_peak_times = [self.times[x] for x in self.angle_positive_peaks_indices]

        # Positive peaks sometimes contain negative numbers. This removes the 'negative'
        # positive peaks.

        self.positive_peak_indices_with_negative_sign = self.positive_opposite_sign_peak_removal(
            self.angle_positive_peaks)

        # Negative peak identification.

        self.angles_reverse_sign = [(-1) * x for x in self.angles]
        self.angle_negative_peaks_indices = sp.signal.find_peaks(self.angles_reverse_sign)[0].tolist()
        self.angle_negative_peaks = [self.angles[x] for x in self.angle_negative_peaks_indices]

        # Negative peak average.

        self.average_negative_peaks = np.average(self.angle_negative_peaks)

        # Negative peak times, that correspond with negative peaks.

        self.angle_negative_peak_times = [self.times[x] for x in self.angle_negative_peaks_indices]

        # 'Positive' negative peak removal/filter.

        self.negative_peak_indices_with_positive_sign = self.negative_opposite_sign_peak_removal(
            self.angle_negative_peaks)

        # Physical removal of conflicting peaks from indices, peaks, and peak times lists,
        # for both negative and postive extremes lists.

        for j in self.positive_peak_indices_with_negative_sign:
            self.angle_positive_peaks_indices.remove(self.angle_positive_peaks_indices[j])
            self.angle_positive_peaks.remove(self.angle_positive_peaks[j])
            self.angle_positive_peak_times.remove(self.angle_positive_peak_times[j])

        for k in self.negative_peak_indices_with_positive_sign:
            self.angle_negative_peaks_indices.remove(self.angle_negative_peaks_indices[k])
            self.angle_negative_peaks.remove(self.angle_negative_peaks[k])
            self.angle_negative_peak_times.remove(self.angle_negative_peak_times[k])

        # self.quick_graph()



    def __str__(self):
        lst=[]
        lst.append(self.text_file_str.split("_"))
        return lst[0][1]

    def text_file_ingestion(self, text_file:str) -> None:

        self.times, self.xs, self.ys, self.angles, self.frames = np.loadtxt(text_file,
                                                          skiprows=2, delimiter=",", dtype=str, unpack=True)

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

        plt.errorbar(self.times, self.angles, yerr=self.angle_uncertainties, marker="o", ls='', label="Raw Data")
        plt.errorbar(self.angle_positive_peak_times, self.angle_positive_peaks, marker="o", ls='', label="Raw Data Positive Peaks")
        plt.errorbar(self.angle_negative_peak_times, self.angle_negative_peaks, marker="o", ls='',
                     color="green", label="Raw Data Negative Peaks")
        plt.title(self.text_file_str)

        plt.figure()


def uncertainty_population(list_length: int, fill_value) -> np.array:

    x = np.arange(list_length, dtype=int)
    populated_uncertainties = np.full_like(x, fill_value, dtype=float)

    return list(populated_uncertainties)

def angle_uncertainty_calculation(x_list, ux_list, y_list, uy_list) -> list:
    product_uncertainties = []
    angle_uncertainties = []

    for i in np.arange(len(x_list)):
        product_uncertainties.append(
            product_uncertainty_propogation((y_list[i]/x_list[i]), x_list[i], y_list[i], ux_list[i], uy_list[i])
        )
    products = [y_list[i]/x_list[i] for i in np.arange(len(x_list))]

    for i in np.arange(len(product_uncertainties)):
        angle_uncertainties.append((1/(1+(products[i]**2)))*np.abs(product_uncertainties[i]))

    return angle_uncertainties


def linear(x, a, b):
    return a*x + b

def non_linear(x, c, d, e):
    return c*(x)**d + e

def log(x, c, d, e):
    return c*np.log(d*x) + e

def exponential(x, a, b, c, d):
    return d*np.exp((-1)*a*x+b) + c

def absolute_value(x, a, b, c):
    return np.abs(a*(x+b)) + c

def puncture_graph(x, a, b, c, d):
    return (-1)*b*(abs(x + c) + a)**2 + d

# def simple_angle_model(tau):

def cos_time_period_model(x, T, a, b):
    return a*np.cos(2*np.pi*(x/T)) + b
def reduced_chi_squared(data_ys, predicted_ys, uncertainties, num_elements, num_parameters):

    dof = num_elements - num_parameters
    residuals = data_ys - predicted_ys
    residuals_squared = np.square(residuals)
    squared_uncertainties = np.square(uncertainties)
    divide = []
    for i in np.arange(len(residuals)):
        divide.append(residuals_squared[i]/squared_uncertainties[i])
    red_chi_squared = (1/dof)*(np.sum(divide))

    return red_chi_squared

def return_peaks_twenties(peaks: list):
    twenties = []
    for i in peaks:
        if  20*(np.pi/180) < i < 23*(np.pi/180):
            twenties.append(i)
    return twenties

def product_uncertainty_propogation(z, x, y, ux, uy) -> float:

    return z*(np.sqrt(((ux/x)**2)+((uy/y)**2)))

def sum_uncertainty_propogation(ux, uy) -> float:

    return np.sqrt(ux**2 + uy**2)

def power_uncertainty_propogation(power, x, ux) -> float:

    return ((power)*x**(power - 1))*ux


