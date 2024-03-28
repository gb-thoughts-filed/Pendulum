
import Pendulum_Analysis

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    r_l1m1 = Pendulum_Analysis.PendulumData("specs_original/R_L1M1.txt")
    # print(r_l1m1.angle_peaks[-14:-1])
    # print(r_l1m1.angle_peak_times[-14:-1])

    # plt.show()

    r_l1m1_no_tail = Pendulum_Analysis.PendulumData("specs_original/R_L1M1_no_tail.txt")
    plt.title("Length 1 Mass 1, Tail Removed")
    # plt.show()

    r_l1m2 = Pendulum_Analysis.PendulumData("specs_original/R_L1M2.txt")
    plt.title("Length 1 Mass 2")
    # plt.show()

    r_l1m3 = Pendulum_Analysis.PendulumData("specs_original/R_L1M3.txt")
    plt.title("Length 1 Mass 3")
    # plt.show()

    r_l1m4 = Pendulum_Analysis.PendulumData("specs_original/R_L1M4.txt")
    plt.title("Length 1 Mass 4")
    # plt.show()

    r_l2m2 = Pendulum_Analysis.PendulumData("specs_original/R_L2M2.txt")
    plt.title("Length 2 Mass 2")
    # plt.show()

    r_l3m2 = Pendulum_Analysis.PendulumData("specs_original/R_L3M2.txt")
    plt.title("Motion of Pendulum with Length (34±1)cm and Mass(199.7±0.1)g: Angle in Degrees v.s. Time(s)")
    plt.xlabel("Time(s)")
    plt.ylabel("Angle(degrees)")
    # plt.show()

    r_l4m2 = Pendulum_Analysis.PendulumData("specs_original/R_L4M2.txt")
    plt.title("Length 4 Mass 2")
    # plt.show()

    r_l5m2 = Pendulum_Analysis.PendulumData("specs_original/R_L5M2.txt")
    plt.title("Length 5 Mass 2")
    # plt.show()

    r_l6m2 = Pendulum_Analysis.PendulumData("specs_original/R_L6M2.txt")
    plt.title("Length 6 Mass 2")
    # plt.show()


# ====================================================================================================================

# Theta Not Investigation
# Using Length 3 Mass 2

# print(len(r_l3m2.angle_positive_peaks))
# print(len(r_l3m2.positive_periods))
# print(len(r_l3m2.angle_negative_peaks))
# print(len(r_l3m2.negative_periods))
#
# print(r_l3m2.angle_positive_peak_times)
# print(r_l3m2.positive_periods)

# print(r_l3m2.angle_positive_peaks)
# print(r_l3m2.angle_negative_peaks)

positive_peaks = r_l3m2.angle_positive_peaks.copy()
positive_peaks.reverse()
positive_periods = r_l3m2.positive_periods[:-1].copy()
positive_periods.reverse()
positive_period_uncertainty = r_l3m2.positive_periods_uncertainty[:-1].copy()
positive_period_uncertainty.reverse()

# print(positive_peaks)

total_peaks = r_l3m2.angle_negative_peaks + positive_peaks
total_periods = r_l3m2.negative_periods + positive_periods
total_period_uncertainty = r_l3m2.negative_periods_uncertainty + positive_period_uncertainty

popt_l3m2_angle_vs_period, pcov_l3m2_angle_vs_period = (
    sp.optimize.curve_fit(Pendulum_Analysis.non_linear, r_l3m2.angle_positive_peaks[:-20], r_l3m2.positive_periods[:-1][:-20],
                          sigma=r_l3m2.positive_periods_uncertainty[:-1][:-20], absolute_sigma=True ))
#print(popt_l3m2_angle_vs_period, pcov_l3m2_angle_vs_period)
plt.figure(2)
plt.errorbar(total_peaks, total_periods, yerr=total_period_uncertainty, marker="o", ls='', label = "Angle v.s. Calculated Period")
plt.title("Angles in Degrees v.s. Pendulum Period(1/s)")
plt.xlabel("Angles in Degrees")
plt.ylabel("Period(1/s")
plt.legend()
plt.show()

plt.errorbar(r_l3m2.angle_negative_peaks[:-20], r_l3m2.negative_periods[:-20], yerr=r_l3m2.negative_periods_uncertainty[:-20], marker="o", ls='', color="blue")
plt.errorbar(r_l3m2.angle_positive_peaks[:-20], r_l3m2.positive_periods[:-1][:-20],
             yerr=r_l3m2.positive_periods_uncertainty[:-1][:-20], marker="o", ls='', color="blue")
plt.title("Angles in Degrees v.s. Pendulum Period(1/s), Points From End of Peak List Removed", wrap=True)
plt.xlabel("Angles in Degrees")
plt.ylabel("Period(1/s)")
plt.show()

# =====================================================================================================================

# Quantitative assessment of asymmetry of pendulum

print(r_l3m2.average_positive_peaks)
print(r_l3m2.average_negative_peaks)

# =====================================================================================================================

# Verify or refute that decay is exponential

popt_l3m2_peakangle_vs_time_nonlin, pcov_l3m2_peakangle_vs_time_nonlin = sp.optimize.curve_fit(Pendulum_Analysis.non_linear,
                                                                                               r_l3m2.angle_positive_peak_times,
                                                                                               r_l3m2.angle_positive_peaks,
                                                                                               sigma=r_l3m2.angle_uncertainties[:104],
                                                                                               absolute_sigma=True)


print(popt_l3m2_peakangle_vs_time_nonlin, pcov_l3m2_peakangle_vs_time_nonlin)



popt_l3m2_peakangle_vs_time_exp, pcov_l3m2_peakangle_vs_time_exp = sp.optimize.curve_fit(Pendulum_Analysis.exponential,
                                                                                               r_l3m2.angle_positive_peak_times,
                                                                                               r_l3m2.angle_positive_peaks,
                                                                                               sigma=r_l3m2.angle_uncertainties[:104],
                                                                                               absolute_sigma=True)
print(popt_l3m2_peakangle_vs_time_exp, pcov_l3m2_peakangle_vs_time_exp)


plt.errorbar(r_l3m2.angle_positive_peak_times, r_l3m2.angle_positive_peaks, yerr=r_l3m2.angle_uncertainties[:104], marker="o", ls='', label = "Raw Peaks")
plt.plot(r_l3m2.angle_positive_peak_times, Pendulum_Analysis.non_linear(r_l3m2.angle_positive_peak_times, *popt_l3m2_peakangle_vs_time_nonlin), label="Predicted Curve")
plt.title("The Positive Peaks of a Wave Depicting The Motion of a Pendulum: Angles(Degrees) v.s. Time(seconds)")
plt.xlabel("Angles(degrees)")
plt.ylabel("Time(s)")
plt.show()

decay_residual_difference = r_l3m2.angle_positive_peaks - (Pendulum_Analysis.non_linear(r_l3m2.angle_positive_peak_times, *popt_l3m2_peakangle_vs_time_nonlin))
# print(decay_residual_difference)
plt.errorbar(r_l3m2.angle_positive_peak_times, decay_residual_difference, yerr=r_l3m2.angle_uncertainties[:104], marker="o", ls='')
plt.title("Residual:The Positive Peaks of a Wave Depicting The Motion of a Pendulum: Angles(Degrees) v.s. Time(seconds)")
plt.ylabel("Raw/Predicted Differences (degrees)")
plt.xlabel("Time(s)")
plt.show()

decay_chi_squared = Pendulum_Analysis.reduced_chi_squared(r_l3m2.angle_positive_peaks,
                                                          Pendulum_Analysis.non_linear(r_l3m2.angle_positive_peak_times, *popt_l3m2_peakangle_vs_time_nonlin),
                                                          r_l3m2.angle_uncertainties[:104],
                                                          len(r_l3m2.angle_positive_peaks),
                                                          3)

decay_goodness_of_fit = 1 - sp.stats.chi2.cdf(decay_chi_squared, 3)
print(decay_chi_squared)
print(decay_goodness_of_fit)


# =====================================================================================================================

lengths = [18, 1], [26, 1], [32, 1], [34, 1], [39, 1], [38, 1]
length_values = [18, 26, 32, 34, 38, 39]

# r_l1m2_twenties = Pendulum_Analysis.return_peaks_twenties(r_l1m2.angle_positive_peaks)
# print(r_l1m2_twenties)
# r_l2m2_twenties = Pendulum_Analysis.return_peaks_twenties(r_l2m2.angle_positive_peaks)
# print(r_l2m2_twenties)
# r_l3m2_twenties = Pendulum_Analysis.return_peaks_twenties(r_l3m2.angle_positive_peaks)
# print(r_l3m2_twenties)
# r_l4m2_twenties = Pendulum_Analysis.return_peaks_twenties(r_l4m2.angle_positive_peaks)
# print(r_l4m2_twenties)
# r_l5m2_twenties = Pendulum_Analysis.return_peaks_twenties(r_l5m2.angle_positive_peaks)
# print(r_l5m2_twenties)
# r_l6m2_twenties = Pendulum_Analysis.return_peaks_twenties(r_l6m2.angle_positive_peaks)
# print(r_l6m2_twenties)

length_theta_not_choice = [21.5, 21.3, 21.4, 21.4, 21.2, 21.5]


length_theta_not_choice_period = [r_l1m2.positive_periods[r_l1m2.angle_positive_peaks.index(21.5)],
                                    r_l2m2.positive_periods[r_l2m2.angle_positive_peaks.index(21.3)],
r_l3m2.positive_periods[r_l3m2.angle_positive_peaks.index(21.4)], r_l4m2.positive_periods[r_l4m2.angle_positive_peaks.index(21.4)],
r_l5m2.positive_periods[r_l5m2.angle_positive_peaks.index(21.2)], r_l6m2.positive_periods[r_l6m2.angle_positive_peaks.index(21.5)]]


popt_length, pcov_length = sp.optimize.curve_fit(Pendulum_Analysis.linear, length_values,
                                                 length_theta_not_choice_period, sigma=np.full(6, 7.07106781*10**(-4)),
                                                 absolute_sigma=True)
print(popt_length, pcov_length)
plt.errorbar(length_values, length_theta_not_choice_period, yerr=np.full(6, 7.07106781*10**(-4)), marker="o", ls='')
plt.plot(length_values, Pendulum_Analysis.linear(np.array(length_values), 0.0190263,0.49784692))
plt.show()

length_difference = length_theta_not_choice_period - Pendulum_Analysis.linear(np.array(length_values), 0.0190263,0.49784692)

plt.errorbar(length_values, length_difference, yerr=np.full(6, 7.07106781*10**(-4)), marker="o", ls='')
plt.show()

length_chi_squared = Pendulum_Analysis.reduced_chi_squared(length_theta_not_choice_period,
                                                           Pendulum_Analysis.linear(np.array(length_values),0.0190263,0.49784692),
np.full(6, 7.07106781*10**(-4)), len(length_theta_not_choice_period), 2)

length_chi_prob = 1 - sp.stats.chi2.cdf(length_chi_squared, 2)

print(length_chi_squared, length_chi_prob)

# =====================================================================================================================