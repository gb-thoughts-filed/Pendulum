
import Pendulum_Analysis_2

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def positive_group_angles_information_extraction(lm_type: Pendulum_Analysis_2.PendulumData, index):

    given_angles = lm_type.angles[lm_type.angle_positive_peaks_indices[index] : lm_type.angle_positive_peaks_indices[index+2]]
    given_angle_uncertainties = lm_type.angle_uncertainties[lm_type.angle_positive_peaks_indices[index] : lm_type.angle_positive_peaks_indices[index+2]]
    given_times = lm_type.times[lm_type.angle_positive_peaks_indices[index] : lm_type.angle_positive_peaks_indices[index+2]]

    return given_angles, given_angle_uncertainties, given_times

def negative_group_angles_information_extraction(lm_type: Pendulum_Analysis_2.PendulumData, index):

    given_angles = lm_type.angles[lm_type.angle_negative_peaks_indices[index] : lm_type.angle_negative_peaks_indices[index+2]]
    given_angle_uncertainties = lm_type.angle_uncertainties[lm_type.angle_negative_peaks_indices[index] : lm_type.angle_negative_peaks_indices[index+2]]
    given_times = lm_type.times[lm_type.angle_negative_peaks_indices[index] : lm_type.angle_negative_peaks_indices[index+2]]

    return given_angles, given_angle_uncertainties, given_times

class PeriodFit:


    def __init__(self, lm_type: Pendulum_Analysis_2.PendulumData, index:int, p_0:list, extraction_function_sign):
        self.angles = None
        self.angle_uncertainties = None
        self.angle_times = None
        self.angles, self.angle_uncertainties, self.angle_times = extraction_function_sign(lm_type, index)



        self.sign = None
        if extraction_function_sign.__str__().split(" ")[1] == "positive_group_angles_information_extraction":
            self.sign = "Positive"
        if extraction_function_sign.__str__().split(" ")[1] == "negative_group_angles_information_extraction":
            self.sign = "Negative"

        self.popt = None
        self.pcov = None
        self.popt, self.pcov = sp.optimize.curve_fit(Pendulum_Analysis_2.cos_time_period_model, self.angle_times,
                                                         self.angles,
                                                         sigma=self.angle_uncertainties, absolute_sigma=True,
                                                         p0=p_0)
        self.p_sigma = np.sqrt(np.diag(self.pcov))
        self.starting_angle = Pendulum_Analysis_2.cos_time_period_model(self.angle_times, *self.popt)[0]
        #print(self.angles[0], self.starting_angle)

        # plt.figure()
        # plt.errorbar(self.angle_times, self.angles, yerr=self.angle_uncertainties, marker="o", ls='', label="Raw Data")
        # plt.plot(self.angle_times, Pendulum_Analysis_2.cos_time_period_model(self.angle_times, *self.popt))
        # plt.title(f"{self.sign} Peaks - {index} to {index+2} - Starting Angle: {self.starting_angle} ")

        self.chi_squared = Pendulum_Analysis_2.reduced_chi_squared(self.angles,
                                                                     Pendulum_Analysis_2.cos_time_period_model(
                                                                         self.angle_times, *self.popt),
                                                                     self.angle_uncertainties, len(self.angles), 3)
        self.chi_squared_prob = (1 - sp.stats.chi2.cdf(self.chi_squared, 3))

        self.period = self.popt[0]/2
        self.period_uncertainty = self.p_sigma[0]/2

class PeriodFit2:

    def __init__(self, lm_type: Pendulum_Analysis_2.PendulumData, index:int, p_0:list, extraction_function_sign):
        self.angles = None
        self.angle_uncertainties = None
        self.angle_times = None
        self.angles, self.angle_uncertainties, self.angle_times = extraction_function_sign(lm_type, index)



        self.sign = None
        if extraction_function_sign.__str__().split(" ")[1] == "positive_group_angles_information_extraction":
            self.sign = "Positive"
        if extraction_function_sign.__str__().split(" ")[1] == "negative_group_angles_information_extraction":
            self.sign = "Negative"

        self.popt = None
        self.pcov = None
        self.popt, self.pcov = sp.optimize.curve_fit(Pendulum_Analysis_2.cos_time_period_model, self.angle_times,
                                                     self.angles,
                                                     sigma=self.angle_uncertainties, absolute_sigma=True,
                                                     p0=p_0)
        self.p_sigma = np.sqrt(np.diag(self.pcov))
        self.starting_angle = Pendulum_Analysis_2.cos_time_period_model(self.angle_times, *self.popt)[0]
        #print(self.angles[0], self.starting_angle)

        # plt.figure()
        # plt.errorbar(self.angle_times, self.angles, yerr=self.angle_uncertainties, marker="o", ls='', label="Raw Data")
        # plt.plot(self.angle_times, Pendulum_Analysis_2.cos_time_period_model(self.angle_times, *self.popt))
        # plt.title(f"{self.sign} Peaks - {index} to {index+2} - Starting Angle: {self.starting_angle} ")

        self.chi_squared = Pendulum_Analysis_2.reduced_chi_squared(self.angles,
                                                                   Pendulum_Analysis_2.cos_time_period_model(
                                                                       self.angle_times, *self.popt),
                                                                   self.angle_uncertainties, len(self.angles), 3)
        self.chi_squared_prob = (1 - sp.stats.chi2.cdf(self.chi_squared, 3))

        self.period = self.popt[0]/2
        self.period_uncertainty = self.p_sigma[0]/2

class DecayFit:

    def __init__(self, lm_type: Pendulum_Analysis_2.PendulumData, p_0):

        self.popt_decay, self.pcov_decay = sp.optimize.curve_fit(Pendulum_Analysis_2.exponential,
                                                       lm_type.angle_positive_peak_times,
                                                       lm_type.angle_positive_peaks,
                                                                 sigma=lm_type.angle_uncertainties[:len(lm_type.angle_positive_peak_times)],
                                                                 absolute_sigma=True, p0=p_0)
        # print(f"Popt Decay {lm_type.__str__()}", self.popt_decay)
        # print(f"Pcov Decay {lm_type.__str__()}", np.sqrt(np.diag(self.pcov_decay)))

        self.p_sigma = np.sqrt(np.diag(self.pcov_decay))

        plt.figure()
        decay_predicted_y = Pendulum_Analysis_2.exponential(np.array(lm_type.angle_positive_peak_times), *self.popt_decay)
        plt.plot(lm_type.angle_positive_peak_times, decay_predicted_y)
        plt.errorbar(lm_type.angle_positive_peak_times, lm_type.angle_positive_peaks,
                     yerr=lm_type.angle_uncertainties[:len(lm_type.angle_positive_peak_times)], marker="o", ls='', label="Raw Data")

        plt.title(f"Decay Graph {lm_type.__str__()}")
        self.decay_chi_squared = Pendulum_Analysis_2.reduced_chi_squared(lm_type.angle_positive_peaks, decay_predicted_y,
                                                                    lm_type.angle_uncertainties[:len(lm_type.angle_positive_peak_times)],
                                                                    len(lm_type.angle_positive_peaks), 4)

        self.decay_chi_prob = (1 - sp.stats.chi2.cdf(self.decay_chi_squared, 4))

        self.tau = 1/self.popt_decay[0]
        self.tau_uncertainty = (-1*(self.tau)**(-2))*self.p_sigma[0]

        # print(f"Decay Chi Squared {lm_type.__str__()}", self.decay_chi_squared)
        # print(f"Decay Chi Prob {lm_type.__str__()}", self.decay_chi_prob)

class DecayFit_ThetaNot:

    def __init__(self, dictionary, key, p_0):

        dictionary_entry = dictionary[key]

        self.popt_decay, self.pcov_decay = sp.optimize.curve_fit(Pendulum_Analysis_2.exponential,
                                                                 dictionary_entry[0],
                                                                 dictionary_entry[1],
                                                                 sigma=dictionary_entry[2],
                                                                 absolute_sigma=True, p0=p_0)

        self.p_sigma = np.sqrt(np.diag(self.pcov_decay))

        plt.figure()
        decay_predicted_y = Pendulum_Analysis_2.exponential(np.array(dictionary_entry[0]), *self.popt_decay)
        plt.plot(dictionary_entry[0], decay_predicted_y)
        plt.errorbar(dictionary_entry[0], dictionary_entry[1],
                     yerr=dictionary_entry[2], marker="o", ls='', label="Raw Data")

        plt.title(f"Decay Graph {key}")
        self.decay_chi_squared = Pendulum_Analysis_2.reduced_chi_squared(dictionary_entry[1], decay_predicted_y,
                                                                         dictionary_entry[2],
                                                                         len(dictionary_entry[1]), 4)

        self.decay_chi_prob = (1 - sp.stats.chi2.cdf(self.decay_chi_squared, 4))

        self.tau = 1/self.popt_decay[0]
        self.tau_uncertainty = (-1*(self.tau)**(-2))*self.p_sigma[0]

        print(f"Popt Decay {key}", self.popt_decay)
        print(f"Pcov Decay {key}", self.tau_uncertainty)

        print(f"Decay Chi Squared {key}", self.decay_chi_squared)
        print(f"Decay Chi Prob {key}", self.decay_chi_prob)



def period_generator(lm_type: Pendulum_Analysis_2.PendulumData, p_0:list, extraction_function_sign, range_length, range_start):

    period_angle_dictionary = {}

    for i in np.arange(range_start, range_length):
        period_angle_dictionary[f"period_{i}"] = PeriodFit(lm_type, i, p_0, extraction_function_sign)

    return period_angle_dictionary


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    r_l1m1 = Pendulum_Analysis_2.PendulumData("R_L1M1_full.txt", 24.59, 3.5, 7)
    #print(r_l1m1.angles)
    #print(r_l1m1.angle_uncertainties)

    r_l1m2 = Pendulum_Analysis_2.PendulumData("R_L1M2_full.txt", 43.90, 3.9, 7)
    r_l1m3 = Pendulum_Analysis_2.PendulumData("R_L1M3_full.txt", 35.51, 4.9, 7)
    r_l1m4 = Pendulum_Analysis_2.PendulumData("R_L1M4_full.txt", 56.65, 3.6, 7)
    r_l1m4.angle_positive_peaks = r_l1m4.angle_positive_peaks[5:]
    r_l1m4.angle_positive_peak_times = r_l1m4.angle_positive_peak_times[5:]
    r_l1m4.angle_positive_peaks_indices = r_l1m4.angle_positive_peaks_indices[5:]

    r_l3m2 = Pendulum_Analysis_2.PendulumData("R_L3M2_full.txt", 43.90, 3.2, 8)
    #print(r_l3m2.angles)
    #print(r_l3m2.angle_uncertainties)

    # Identifying Periods R_L3M2

    # First period
    r_l2m2 = Pendulum_Analysis_2.PendulumData("R_L2M2_full.txt", 43.90, 2.7, 7)
    r_l4m2 = Pendulum_Analysis_2.PendulumData("R_L4M2_full.txt", 43.90, 4.0, 11)
    r_l5m2 = Pendulum_Analysis_2.PendulumData("R_L5M2_full.txt", 43.90, 3.0, 10)
    r_l6m2 = Pendulum_Analysis_2.PendulumData("R_L6M2_full.txt", 43.90, 1.9, 9)


    # l3m2_angles_1, l3m2_angle_uncertainties_1, l3m2_times_1 = positive_group_angles_information_extraction(r_l3m2, 10)
    #
    # popt_l3m2_1, pcov_l3m2_1 = sp.optimize.curve_fit(Pendulum_Analysis_2.cos_time_period_model, l3m2_times_1, l3m2_angles_1,
    #                                                  sigma=l3m2_angle_uncertainties_1, absolute_sigma=True, p0=[1.15, 0.85, 0])
    # p_sigma = np.sqrt(np.diag(pcov_l3m2_1))
    # print(popt_l3m2_1, np.sqrt(np.diag(pcov_l3m2_1)))
    # plt.errorbar(l3m2_times_1, l3m2_angles_1, yerr=l3m2_angle_uncertainties_1, marker="o", ls='', label="Raw Data")
    # plt.plot(l3m2_times_1, Pendulum_Analysis_2.cos_time_period_model(l3m2_times_1, *popt_l3m2_1))
    # #plt.plot(l3m2_times_1, -0.85*np.cos(2*np.pi*(0.867)*(np.array(l3m2_times_1))))
    #
    # l3m2_1_chi_squared = Pendulum_Analysis_2.reduced_chi_squared(l3m2_angles_1,
    #                                                              Pendulum_Analysis_2.cos_time_period_model(l3m2_times_1, *popt_l3m2_1),
    #                                                              l3m2_angle_uncertainties_1, len(l3m2_angles_1), 3)
    # #print(l3m2_1_chi_squared)
    #
    # l3m2_1_chi_squared_prob = (1 - sp.stats.chi2.cdf(l3m2_1_chi_squared, 3))
    # print(l3m2_1_chi_squared_prob)
    #
    # period = (popt_l3m2_1[0]/2, p_sigma[0]/2)
    #
    # plt.figure(3)


    # First Period Class Test

    # pos_period_1 = PeriodFit(r_l3m2, 0, [1.1173184, 0.85], positive_group_angles_information_extraction)
    # print("Period 1", pos_period_1.popt, pos_period_1.p_sigma)
    # print(pos_period_1.starting_angle, pos_period_1.period, pos_period_1.period_uncertainty)
    # print(pos_period_1.chi_squared_prob)




    ''''''

    # ====================================================================================================================

    # Theta Not Investigation
    # Using Length 3 Mass 2

    pos_period_objects_1 = period_generator(r_l3m2, [1.1173184, 0.85, 0], positive_group_angles_information_extraction, 97, 0)
    neg_period_objects_1 = period_generator(r_l3m2, [1.1173184, 0.85, 0], negative_group_angles_information_extraction, 97, 0)

    pos_period_objects_2 = period_generator(r_l3m2, [1.113, 0.85, 0], positive_group_angles_information_extraction, 97, 60)
    neg_period_objects_2 = period_generator(r_l3m2, [1.113, 0.85, 0], negative_group_angles_information_extraction, 97, 60)

    pos_period_objects_3 = period_generator(r_l3m2, [1.109, 0.85, 0], positive_group_angles_information_extraction, 97, 71)
    neg_period_objects_3 = period_generator(r_l3m2, [1.109, 0.85, 0], negative_group_angles_information_extraction, 97, 71)

    pos_period_objects_4 = period_generator(r_l3m2, [1.105, 0.85, 0], positive_group_angles_information_extraction, 97, 83)
    neg_period_objects_4 = period_generator(r_l3m2, [1.105, 0.85, 0], negative_group_angles_information_extraction, 97, 83)


    plt.figure()

    pos_period_angles2_1 = []
    pos_period_periods2_1 = []
    pos_period_uncertainties2_1 = []

    for i in pos_period_objects_1.values():
        pos_period_angles2_1.append(i.starting_angle)
        pos_period_periods2_1.append(i.period)
        pos_period_uncertainties2_1.append(i.period_uncertainty)


    neg_period_angles2_1 = []
    neg_period_periods2_1 = []
    neg_period_uncertainties2_1 = []

    for i in neg_period_objects_1.values():
        neg_period_angles2_1.append(i.starting_angle)
        neg_period_periods2_1.append(i.period)
        neg_period_uncertainties2_1.append(i.period_uncertainty)

    pos_period_angles2_2 = []
    pos_period_periods2_2 = []
    pos_period_uncertainties2_2 = []

    for i in pos_period_objects_2.values():
        pos_period_angles2_2.append(i.starting_angle)
        pos_period_periods2_2.append(i.period)
        pos_period_uncertainties2_2.append(i.period_uncertainty)


    neg_period_angles2_2 = []
    neg_period_periods2_2 = []
    neg_period_uncertainties2_2 = []
    for i in neg_period_objects_2.values():
        neg_period_angles2_2.append(i.starting_angle)
        neg_period_periods2_2.append(i.period)
        neg_period_uncertainties2_2.append(i.period_uncertainty)

    pos_period_angles2_3 = []
    pos_period_periods2_3 = []
    pos_period_uncertainties2_3 = []

    for i in pos_period_objects_3.values():
        pos_period_angles2_3.append(i.starting_angle)
        pos_period_periods2_3.append(i.period)
        pos_period_uncertainties2_3.append(i.period_uncertainty)


    neg_period_angles2_3 = []
    neg_period_periods2_3 = []
    neg_period_uncertainties2_3 = []

    for i in neg_period_objects_3.values():
        neg_period_angles2_3.append(i.starting_angle)
        neg_period_periods2_3.append(i.period)
        neg_period_uncertainties2_3.append(i.period_uncertainty)


    pos_period_angles2_4 = []
    pos_period_periods2_4 = []
    pos_period_uncertainties2_4 = []

    for i in pos_period_objects_4.values():
        pos_period_angles2_4.append(i.starting_angle)
        pos_period_periods2_4.append(i.period)
        pos_period_uncertainties2_4.append(i.period_uncertainty)


    neg_period_angles2_4 = []
    neg_period_periods2_4 = []
    neg_period_uncertainties2_4 = []
    for i in neg_period_objects_4.values():
        neg_period_angles2_4.append(i.starting_angle)
        neg_period_periods2_4.append(i.period)
        neg_period_uncertainties2_4.append(i.period_uncertainty)

    # plt.figure()
    # plt.errorbar(pos_period_angles2_1[:60], pos_period_periods2_1[:60], marker="o", ls='', label=" Positive Angle Raw Data")
    # plt.errorbar(neg_period_angles2_1[:60], neg_period_periods2_1[:60], marker="o", ls='', label=" Negative Angle Raw Data")
    # plt.errorbar(pos_period_angles2_2[:11], pos_period_periods2_2[:11], marker="o", ls='', label=" Positive Angle Raw Data")
    # plt.errorbar(neg_period_angles2_2[:11], neg_period_periods2_2[:11], marker="o", ls='', label=" Negative Angle Raw Data")
    # plt.errorbar(pos_period_angles2_3[:12], pos_period_periods2_3[:12], marker="o", ls='', label=" Positive Angle Raw Data")
    # plt.errorbar(neg_period_angles2_3[:12], neg_period_periods2_3[:12], marker="o", ls='', label=" Negative Angle Raw Data")
    # plt.errorbar(pos_period_angles2_4, pos_period_periods2_4, marker="o", ls='', label=" Positive Angle Raw Data")
    # plt.errorbar(neg_period_angles2_4, neg_period_periods2_4, marker="o", ls='', label=" Negative Angle Raw Data")
    # plt.title("Period - Angle Graph L3M2")

    ppa2_4 = pos_period_angles2_4.copy()
    ppa2_4.reverse()
    ppa2_3 = pos_period_angles2_3[:12].copy()
    ppa2_3.reverse()
    ppa2_2 = pos_period_angles2_2[:11].copy()
    ppa2_2.reverse()
    ppa2_1 = pos_period_angles2_1[:60].copy()
    ppa2_1.reverse()

    all_period_angles = neg_period_angles2_1[:60] + neg_period_angles2_2[:11] + neg_period_angles2_3[:12] + neg_period_angles2_4 + \
                        ppa2_4 + ppa2_3 + ppa2_2 + ppa2_1

    positive_period_angles = pos_period_angles2_4 + pos_period_angles2_3[:12] + pos_period_angles2_2[:11] + pos_period_angles2_1[:60]
    negative_period_angles = neg_period_angles2_1[:60] + neg_period_angles2_2[:11] + neg_period_angles2_3[:12] + neg_period_angles2_4

    #print(len(all_period_angles))

    ppp2_4 = pos_period_periods2_4.copy()
    ppp2_4.reverse()
    ppp2_3 = pos_period_periods2_3[:12].copy()
    ppp2_3.reverse()
    ppp2_2 = pos_period_periods2_2[:11].copy()
    ppp2_2.reverse()
    ppp2_1 = pos_period_periods2_1[:60].copy()
    ppp2_1.reverse()
    all_period_periods = neg_period_periods2_1[:60] + neg_period_periods2_2[:11] + neg_period_periods2_3[:12] + neg_period_periods2_4 + \
                         ppp2_4 + ppp2_3 + ppp2_2 + ppp2_1

    positive_period_periods = pos_period_periods2_4 + pos_period_periods2_3[:12] + pos_period_periods2_2[:11] + pos_period_periods2_1[:60]
    negative_period_periods = neg_period_periods2_1[:60] + neg_period_periods2_2[:11] + neg_period_periods2_3[:12] + neg_period_periods2_4

    ppu2_4 = pos_period_uncertainties2_4.copy()
    ppu2_4.reverse()
    ppu2_3 = pos_period_uncertainties2_3[:12].copy()
    ppu2_3.reverse()
    ppu2_2 = pos_period_uncertainties2_2[:11].copy()
    ppu2_2.reverse()
    ppu2_1 = pos_period_uncertainties2_1[:60].copy()
    ppu2_1.reverse()

    all_period_uncertainties = neg_period_uncertainties2_1[:60] + neg_period_uncertainties2_2[:11] + neg_period_uncertainties2_3[:12] + neg_period_uncertainties2_4 + \
                         ppu2_4 + ppu2_3 + ppu2_2 + ppu2_1

    positive_period_uncertainties = pos_period_uncertainties2_4 + pos_period_uncertainties2_3[:12] + pos_period_uncertainties2_2[:11] + pos_period_uncertainties2_1[:60]
    negative_period_uncertainties = neg_period_uncertainties2_1[:60] + neg_period_uncertainties2_2[:11] + neg_period_uncertainties2_3[:12] + neg_period_uncertainties2_4


    # Fitting Angle Period Graph

    true_period_angle_uncertainty = np.full(194, 0.09899494937)
    angle_uncertainty = np.full(194, r_l3m2.angle_uncertainties[0])

    popt_period_angle, pcov_period_angle = sp.optimize.curve_fit(Pendulum_Analysis_2.puncture_graph, all_period_angles[10:184],
                                                                 all_period_periods[10:184],
                                                                 sigma=all_period_uncertainties[10:184], absolute_sigma=True, p0=[1, 1, -0.001, 0])
    x_model_start = all_period_angles[10]
    x_model_end = all_period_angles[184]
    x_model = np.linspace(x_model_start, x_model_end, 1000)

    period_angle_chi_squared = Pendulum_Analysis_2.reduced_chi_squared(all_period_angles[10:184], Pendulum_Analysis_2.puncture_graph(np.array(all_period_angles[10:184]), *popt_period_angle),
                                                                       all_period_uncertainties[10:184], len(all_period_angles[10:184]), 4)

    period_angle_chi_prob = (1 - sp.stats.chi2.cdf(period_angle_chi_squared, 4))

    print("Period Angle Chi Squared", period_angle_chi_squared)
    print("Period Angle Chi Squared Probability", period_angle_chi_prob)

    # plt.figure()
    # plt.plot(x_model, Pendulum_Analysis_2.puncture_graph(x_model, *popt_period_angle), label="Fit")
    # plt.errorbar(all_period_angles[10:184], all_period_periods[10:184], xerr=angle_uncertainty[10:184],
    #              yerr=all_period_uncertainties[10:184], marker="o", ls='',
    #              label="Angle-Period Data", markersize=10)
    # # plt.plot(all_period_angles, Pendulum_Analysis_2.puncture_graph(np.array(all_period_angles), *popt_period_angle))
    # plt.xlabel("Angles in Radians", fontsize=32, wrap=True)
    # plt.ylabel("Period Length in Seconds", fontsize=32, wrap=True)
    # plt.title("Initial Angle vs Resultant Period Length", fontsize=36, wrap=True)
    # plt.legend(fontsize=24)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    #plt.show()

    print("Popt Period Angle", popt_period_angle)
    print("Pcov, sigma Period Angle", np.sqrt(np.diag(pcov_period_angle)))


    #Residual
    period_angle_differences = np.array(all_period_periods[10:184]) - np.array(Pendulum_Analysis_2.puncture_graph(np.array(all_period_angles[10:184]), *popt_period_angle))
    # plt.figure()
    # plt.title("Residual: Initial Angle vs Resultant Period Length", fontsize=36, wrap=True)
    # plt.xlabel("Angles in Radians", fontsize=32)
    # plt.ylabel("Difference Between Model Period Length and Data in Seconds", wrap=True, fontsize=32)
    # plt.errorbar(period_angle_differences, all_period_periods[10:184],
    #              yerr=all_period_uncertainties[10:184], marker="o", ls='', markersize=10,
    #              label="Angle-Period Data and Model Difference")
    # plt.legend(fontsize=24)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.show()
    # =====================================================================================================================

    # Quantitative assessment of asymmetry of pendulum


    # =====================================================================================================================

    # Verify or refute that decay is exponential, find the time constant tau

    popt_decay_l3m2, pcov_decay_l3m2 = sp.optimize.curve_fit(Pendulum_Analysis_2.exponential,
                                                             r_l3m2.angle_positive_peak_times,
                                                             r_l3m2.angle_positive_peaks,
                                                             sigma=r_l3m2.angle_uncertainties[:len(r_l3m2.angle_positive_peak_times)],
                                                             absolute_sigma=True)
    #print("Popt Decay", popt_decay_l3m2)
    #print("Pcov Decay", np.sqrt(np.diag(pcov_decay_l3m2)))
    tau_p_sigma = np.sqrt(np.diag(pcov_decay_l3m2))
    print("Determined Tau", 1/popt_decay_l3m2[0])
    print("Determined Tau Uncertainty", (-1*(1/popt_decay_l3m2[0])**(-2))*tau_p_sigma[0])

    decay_predicted_y = Pendulum_Analysis_2.exponential(np.array(r_l3m2.angle_positive_peak_times), *popt_decay_l3m2)

    # plt.figure()
    # plt.plot(r_l3m2.angle_positive_peak_times, decay_predicted_y, label="Fit")
    # plt.errorbar(r_l3m2.angle_positive_peak_times, r_l3m2.angle_positive_peaks, xerr=r_l3m2.time_uncertainties[:104],
    #              yerr=r_l3m2.angle_uncertainties[:len(r_l3m2.angle_positive_peak_times)],
    #              marker="o", ls='', label="Raw Positive Peak Data", markersize=10)
    # plt.title("Exponential Decay of a Pendulum with Only Positive Peaks Identified", wrap=True, fontsize=36)
    # plt.xlabel("Time in Seconds", fontsize=32, wrap=True)
    # plt.ylabel("Angles in Radians", fontsize=32, wrap=True)
    # plt.legend(fontsize=24)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)

    decay_chi_squared = Pendulum_Analysis_2.reduced_chi_squared(r_l3m2.angle_positive_peaks, decay_predicted_y,
                                                                r_l3m2.angle_uncertainties[:len(r_l3m2.angle_positive_peak_times)],
                                                                len(r_l3m2.angle_positive_peaks), 4)

    decay_chi_prob = (1 - sp.stats.chi2.cdf(decay_chi_squared, 4))

    print("Decay Chi Squared", decay_chi_squared)
    print("Decay Chi Prob", decay_chi_prob)

    # plt.figure()
    # decay_differences = np.array(r_l3m2.angle_positive_peaks) - decay_predicted_y
    # plt.errorbar(r_l3m2.angle_positive_peak_times, decay_differences, xerr=r_l3m2.time_uncertainties[:104],
    #              yerr=r_l3m2.angle_uncertainties[:len(r_l3m2.angle_positive_peak_times)],
    #              marker="o", ls='', label="Difference Between Peak Data and Model",
    #              markersize=10)
    # plt.title("Residual: Exponential Decay of a Pendulum with Only Positive Peaks Identified", wrap=True, fontsize=36)
    # plt.ylabel("Difference Between Positive Peak Data and Model in Radians", wrap=True, fontsize=32)
    # plt.xlabel("Time in Seconds", wrap=True, fontsize=32)
    # plt.axhline(0, label="0 line", color="black")
    # plt.legend(fontsize=24)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    #
    # plt.show()

    # =====================================================================================================================

    # Verify or refute the claim that the period depends on L+D

    lengths = [18, 1], [26, 1], [32, 1], [34, 1], [39, 1], [38, 1]
    length_uncertainties = np.full(6, 1)
    length_values = np.array([18, 26, 32, 34, 38, 39])

    # r_l1m2_twenties = Pendulum_Analysis_2.return_peaks_twenties(r_l1m2.angle_positive_peaks)
    # print(r_l1m2_twenties)
    # r_l2m2_twenties = Pendulum_Analysis_2.return_peaks_twenties(r_l2m2.angle_positive_peaks)
    # print(r_l2m2_twenties)
    # r_l3m2_twenties = Pendulum_Analysis_2.return_peaks_twenties(r_l3m2.angle_positive_peaks)
    # print(r_l3m2_twenties)
    # r_l4m2_twenties = Pendulum_Analysis_2.return_peaks_twenties(r_l4m2.angle_positive_peaks)
    # print(r_l4m2_twenties)
    # r_l5m2_twenties = Pendulum_Analysis_2.return_peaks_twenties(r_l5m2.angle_positive_peaks)
    # print(r_l5m2_twenties)
    # r_l6m2_twenties = Pendulum_Analysis_2.return_peaks_twenties(r_l6m2.angle_positive_peaks)
    # print(r_l6m2_twenties)

    length_theta_not_choice = [0.3752457891787809, 0.3717551306747922, 0.3735004599267865,
                               0.3735004599267865, 0.3700098014227979, 0.3752457891787809]


    r_l1m2_037_index = r_l1m2.angle_positive_peaks.index(0.3752457891787809)
    r_l2m2_037_index = r_l2m2.angle_positive_peaks.index(0.3717551306747922)
    r_l3m2_037_index = r_l3m2.angle_positive_peaks.index(0.3735004599267865)
    r_l4m2_037_index = r_l4m2.angle_positive_peaks.index(0.3735004599267865)
    r_l5m2_037_index = r_l5m2.angle_positive_peaks.index(0.3700098014227979)
    r_l6m2_037_index = r_l6m2.angle_positive_peaks.index(0.3752457891787809)


    r_l1m2_fit_object = PeriodFit2(r_l1m2, r_l1m2_037_index, [0.82, 1.2, 0], positive_group_angles_information_extraction)
    # print("r_l1m2_fit_object.popt", r_l1m2_fit_object.popt, r_l1m2_fit_object.chi_squared_prob)

    r_l2m2_fit_object = PeriodFit2(r_l2m2, r_l2m2_037_index, [0.985, 1.2, 0], positive_group_angles_information_extraction)
    # print("r_l2m2_fit_object.popt", r_l2m2_fit_object.popt, r_l2m2_fit_object.chi_squared_prob)

    r_l3m2_fit_object = PeriodFit2(r_l3m2, r_l3m2_037_index, [1.1173184, 0.85, 0], positive_group_angles_information_extraction)
    # print("r_l3m2_fit_object.popt", r_l3m2_fit_object.popt, r_l3m2_fit_object.chi_squared_prob)

    r_l4m2_fit_object = PeriodFit2(r_l4m2, r_l4m2_037_index, [1.17, 0.33, 0], positive_group_angles_information_extraction)
    # print("r_l4m2_fit_object.popt", r_l4m2_fit_object.popt, r_l4m2_fit_object.chi_squared_prob)

    r_l5m2_fit_object = PeriodFit2(r_l5m2, r_l5m2_037_index, [1.22, 0.35, 0], positive_group_angles_information_extraction)
    # print("r_l5m2_fit_object.popt", r_l5m2_fit_object.popt, r_l5m2_fit_object.chi_squared_prob)

    r_l6m2_fit_object = PeriodFit2(r_l6m2, r_l6m2_037_index, [1.24, 0.35, 0], positive_group_angles_information_extraction)
    # print("r_l6m2_fit_object.popt", r_l6m2_fit_object.popt, r_l6m2_fit_object.chi_squared_prob)




    #plt.figure()
    # plt.errorbar(length_theta_not_choice, length_theta_not_choice_period, marker="o", ls='', label="Raw Data")
    plt.show()

    length_theta_not_choice_period = np.array([r_l1m2_fit_object.period,
                                      r_l2m2_fit_object.period,
                                      r_l3m2_fit_object.period,
                                      r_l4m2_fit_object.period,
                                      r_l5m2_fit_object.period,
                                      r_l6m2_fit_object.period])

    print(list(zip(length_values, length_theta_not_choice_period)))

    length_theta_not_choice_period_uncertainty = [r_l1m2_fit_object.period_uncertainty,
                                      r_l2m2_fit_object.period_uncertainty,
                                      r_l3m2_fit_object.period_uncertainty,
                                      r_l4m2_fit_object.period_uncertainty,
                                      r_l5m2_fit_object.period_uncertainty,
                                      r_l6m2_fit_object.period_uncertainty]
    length_theta_not_choice_period_uncertainty = true_period_angle_uncertainty = np.full(6, 0.09899494937)
    print(length_theta_not_choice_period_uncertainty)
    length_period_popt, length_period_pcov = sp.optimize.curve_fit(Pendulum_Analysis_2.linear,
                                                                   length_values,
                                                                   length_theta_not_choice_period,
                                                                   sigma=length_theta_not_choice_period_uncertainty,
                                                                   absolute_sigma=True,
                                                                   p0=[0.01, 0.24])



    length_period_chi_squared = Pendulum_Analysis_2.reduced_chi_squared(length_theta_not_choice_period,
                                                                        Pendulum_Analysis_2.linear(length_theta_not_choice_period, *length_period_popt),
                                                                        length_theta_not_choice_period_uncertainty,
                                                                        len(length_theta_not_choice_period), 2)

    length_period_chi_prob = (1 - sp.stats.chi2.cdf(decay_chi_squared, 2))

    print("Length Period Chi Squared", length_period_chi_squared)
    print("Length Period Chi Prob", length_period_chi_prob)



    plt.errorbar(length_values, length_theta_not_choice_period, xerr=length_uncertainties,
                 yerr=length_theta_not_choice_period_uncertainty, markersize=10,
                 marker="o", ls='', label="Raw Data")
    plt.plot(length_values, Pendulum_Analysis_2.linear(length_theta_not_choice_period, *length_period_popt), label="Fit")
    plt.plot(length_values, 0.01*length_values + 0.24, label="Manual Fit")
    plt.title("Increasing Length(cm) v.s. Resultant Period", fontsize=36)
    plt.xlabel("Length(cm)", fontsize=32)
    plt.ylabel("Period(s)", fontsize=32)
    plt.legend(fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

    # =========================================================================

    # Verify or refute the claim that the period depends on mass
    mass_values = [50.0, 99.8, 199.7, 499.4]
    mass_uncertainties = np.full(4, 0.1)
    # r_l1m1_twenties = Pendulum_Analysis_2.return_peaks_twenties(r_l1m1.angle_positive_peaks)
    # print(np.array(r_l1m1_twenties))
    # r_l1m2_twenties = Pendulum_Analysis_2.return_peaks_twenties(r_l1m2.angle_positive_peaks)
    # print(np.array(r_l1m2_twenties))
    # r_l1m3_twenties = Pendulum_Analysis_2.return_peaks_twenties(r_l1m3.angle_positive_peaks)
    # print(np.array(r_l1m3_twenties))
    # r_l1m4_twenties = Pendulum_Analysis_2.return_peaks_twenties(r_l1m4.angle_positive_peaks)
    # print(np.array(r_l1m4_twenties))

    mass_theta_not_choice = [0.36128316, 0.3577925, 0.36302848, 0.36128316]

    r_l1m1_036_index = r_l1m2.angle_positive_peaks.index(0.3752457891787809)
    r_l1m2_036_index = r_l2m2.angle_positive_peaks.index(0.3717551306747922)
    r_l1m3_036_index = r_l3m2.angle_positive_peaks.index(0.3735004599267865)
    r_l1m4_036_index = r_l4m2.angle_positive_peaks.index(0.3735004599267865)

    r_l1m1_fit_object = PeriodFit2(r_l1m1, r_l1m1_036_index, [0.81, 1.2, 0], positive_group_angles_information_extraction)
    # print("r_l1m1_fit_object.popt", r_l1m1_fit_object.popt, r_l1m1_fit_object.chi_squared_prob)

    r_l1m3_fit_object = PeriodFit2(r_l1m3, r_l1m3_036_index, [0.81, 1.2, 0], positive_group_angles_information_extraction)
    # print("r_l1m3_fit_object.popt", r_l1m3_fit_object.popt, r_l1m3_fit_object.chi_squared_prob)

    r_l1m4_fit_object = PeriodFit2(r_l1m4, r_l1m4_036_index, [0.81, 1.2, 0], positive_group_angles_information_extraction)
    # print("r_l1m4_fit_object.popt", r_l1m4_fit_object.popt, r_l1m4_fit_object.chi_squared_prob)

    #plt.show()

    mass_theta_not_choice_period = [r_l1m1_fit_object.period, r_l1m2_fit_object.period,
                                    r_l1m3_fit_object.period, r_l1m4_fit_object.period]

    plt.errorbar(mass_values, mass_theta_not_choice_period, marker="o", ls='', label="Raw Data")
    plt.title("Mass Period Graph")
    plt.show()

    # =========================================================================

    # Investigate the effect of L+D, mass on tau

    r_l1m1_decay_object = DecayFit(r_l1m1, [0.08, 0, 0, 1])
    r_l1m2_decay_object = DecayFit(r_l1m2, [0.08, 0, 0, 1])
    r_l1m3_decay_object = DecayFit(r_l1m3, [0.08, 0, 0, 1])
    r_l1m4_decay_object = DecayFit(r_l1m4, [0.08, 0, 0, 1])

    mass_taus = [r_l1m1_decay_object.tau, r_l1m2_decay_object.tau,
                 r_l1m3_decay_object.tau, r_l1m4_decay_object.tau]





    plt.figure()
    plt.errorbar(mass_values, mass_taus, marker="o", ls='', label="Raw Data")
    plt.title("Mass Tau Graph")
    #print(list(quick_tuple))
    plt.show()

    r_l2m2_decay_object = DecayFit(r_l2m2, [0.08, 0, 0, 1])
    r_l4m2_decay_object = DecayFit(r_l4m2, [0.08, 0, 0, 1])
    r_l5m2_decay_object = DecayFit(r_l5m2, [0.03, 0, 0, 1])
    #quick_tuple = zip(r_l5m2.angle_positive_peak_times, r_l5m2.angle_positive_peaks)
    #print(list(quick_tuple))
    r_l6m2_decay_object = DecayFit(r_l6m2, [0.08, 0, 0, 1])

    length_taus = [r_l1m2_decay_object.tau, r_l1m2_decay_object.tau,
                   popt_decay_l3m2[0], r_l4m2_decay_object.tau,
                   r_l5m2_decay_object.tau, r_l6m2_decay_object.tau]

    plt.figure()
    plt.errorbar(length_values, length_taus, marker="o", ls='', label="Raw Data")
    plt.title("Length Tau Graph")

    plt.show()

    # =========================================================================

    # Investigate the effect of theta not on tau

    r_l3m2_length_pos_peaks = len(r_l3m2.angle_positive_peaks)
    #print(len(r_l3m2.angle_positive_peaks))

    parsed_theta_not_dict = {}


    theta_not_1 = r_l3m2.angle_positive_peaks[0:13]

    for i in np.arange(13):
        start_index = 0
        end_index = 13
        if f"theta_not_{i-1}" in parsed_theta_not_dict:
            start_index = len(parsed_theta_not_dict[f"theta_not_{i-1}"][0])*i
            end_index = start_index + 13
            if end_index == 104:
                end_index = 103

        parsed_theta_not_dict[f"theta_not_{i}"] = [r_l3m2.angle_positive_peak_times[start_index:end_index],
                                                   r_l3m2.angle_positive_peaks[start_index:end_index],
                                                   r_l3m2.angle_uncertainties[:len(r_l3m2.angle_positive_peak_times[start_index:end_index])]]


    #print(parsed_theta_not_dict.keys())

    theta_not_0_decay_fit = DecayFit_ThetaNot(parsed_theta_not_dict, 'theta_not_0', [0.0248951, 0.04613781, -0.0497865, 1.0524421])
    theta_not_1_decay_fit = DecayFit_ThetaNot(parsed_theta_not_dict, 'theta_not_1', [0.0248951, 0.04613781, -0.0497865, 1.0524421])
    theta_not_2_decay_fit = DecayFit_ThetaNot(parsed_theta_not_dict, 'theta_not_2', [0.0248951, 0.04613781, -0.0497865, 1.0524421])
    theta_not_3_decay_fit = DecayFit_ThetaNot(parsed_theta_not_dict, 'theta_not_3', [0.03, 0, 0.05, 1])

    # plt.figure()
    # plt.errorbar(parsed_theta_not_dict[f"theta_not_3"][0], parsed_theta_not_dict[f"theta_not_3"][1], marker="o", ls='', label="Raw Data")

    theta_not_4_decay_fit = DecayFit_ThetaNot(parsed_theta_not_dict, 'theta_not_4', [0.03, 0, 0.05, 1])
    theta_not_5_decay_fit = DecayFit_ThetaNot(parsed_theta_not_dict, 'theta_not_5', [0.03, 0, 0.05, 1])
    theta_not_6_decay_fit = DecayFit_ThetaNot(parsed_theta_not_dict, 'theta_not_6', [0.03, 0, 0.05, 1])
    # theta_not_7_decay_fit = DecayFit_ThetaNot(parsed_theta_not_dict, 'theta_not_7', [0, 5, 0.02, 1])


    # print(list(zip(parsed_theta_not_dict[f"theta_not_7"][0], parsed_theta_not_dict[f"theta_not_7"][1])))


    plt.show()

    theta_not_angle_values = []


    for i in np.arange(7):
        theta_not_angle_values.append(parsed_theta_not_dict[f"theta_not_{i}"][1][0])

    theta_not_tau_values = [theta_not_0_decay_fit.tau,
                            theta_not_1_decay_fit.tau,
                            theta_not_2_decay_fit.tau,
                            theta_not_3_decay_fit.tau,
                            theta_not_4_decay_fit.tau,
                            theta_not_5_decay_fit.tau,
                            theta_not_6_decay_fit.tau]

    print(len(theta_not_angle_values))
    print(len(theta_not_tau_values))

    plt.figure()
    plt.errorbar(theta_not_angle_values, theta_not_tau_values, marker="o", ls='', label="Raw Data")
    plt.title("Theta_0 Tau Graph")

    plt.show()
