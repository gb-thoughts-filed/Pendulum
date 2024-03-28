
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

        self.period = self.popt[0]
        self.period_uncertainty = self.p_sigma[0]

def period_generator(lm_type: Pendulum_Analysis_2.PendulumData, p_0:list, extraction_function_sign, range_length, range_start):

    period_angle_dictionary = {}

    for i in np.arange(range_start, range_length):
        period_angle_dictionary[f"period_{i}"] = PeriodFit(lm_type, i, p_0, extraction_function_sign)

    return period_angle_dictionary


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    r_l1m1 = Pendulum_Analysis_2.PendulumData("R_L1M1_full.txt", 24.59)
    #print(r_l1m1.angles)
    #print(r_l1m1.angle_uncertainties)

    r_l1m2 = Pendulum_Analysis_2.PendulumData("R_L1M2_full.txt", 43.90)
    r_l1m3 = Pendulum_Analysis_2.PendulumData("R_L1M3_full.txt", 35.51)
    r_l1m4 = Pendulum_Analysis_2.PendulumData("R_L1M1_full.txt", 56.65)

    r_l3m2 = Pendulum_Analysis_2.PendulumData("R_L3M2_full.txt", 43.90)
    #print(r_l3m2.angles)
    #print(r_l3m2.angle_uncertainties)



    # Identifying Periods R_L3M2

    # First period
    r_l2m2 = Pendulum_Analysis_2.PendulumData("R_L2M2_full.txt", 43.90)
    r_l4m2 = Pendulum_Analysis_2.PendulumData("R_L4M2_full.txt", 43.90)
    r_l5m2 = Pendulum_Analysis_2.PendulumData("R_L5M2_full.txt", 43.90)
    r_l6m2 = Pendulum_Analysis_2.PendulumData("R_L6M2_full.txt", 43.90)


    l3m2_angles_1, l3m2_angle_uncertainties_1, l3m2_times_1 = positive_group_angles_information_extraction(r_l3m2, 10)

    popt_l3m2_1, pcov_l3m2_1 = sp.optimize.curve_fit(Pendulum_Analysis_2.cos_time_period_model, l3m2_times_1, l3m2_angles_1,
                                                     sigma=l3m2_angle_uncertainties_1, absolute_sigma=True, p0=[1.15, 0.85, 0])
    p_sigma = np.sqrt(np.diag(pcov_l3m2_1))
    print(popt_l3m2_1, np.sqrt(np.diag(pcov_l3m2_1)))
    plt.errorbar(l3m2_times_1, l3m2_angles_1, yerr=l3m2_angle_uncertainties_1, marker="o", ls='', label="Raw Data")
    plt.plot(l3m2_times_1, Pendulum_Analysis_2.cos_time_period_model(l3m2_times_1, *popt_l3m2_1))
    #plt.plot(l3m2_times_1, -0.85*np.cos(2*np.pi*(0.867)*(np.array(l3m2_times_1))))

    l3m2_1_chi_squared = Pendulum_Analysis_2.reduced_chi_squared(l3m2_angles_1,
                                                                 Pendulum_Analysis_2.cos_time_period_model(l3m2_times_1, *popt_l3m2_1),
                                                                 l3m2_angle_uncertainties_1, len(l3m2_angles_1), 3)
    #print(l3m2_1_chi_squared)

    l3m2_1_chi_squared_prob = (1 - sp.stats.chi2.cdf(l3m2_1_chi_squared, 3))
    print(l3m2_1_chi_squared_prob)

    period = (popt_l3m2_1[0]/2, p_sigma[0]/2)

    plt.figure(3)


    # First Period Class Test

    # pos_period_1 = PeriodFit(r_l3m2, 0, [1.1173184, 0.85], positive_group_angles_information_extraction)
    # print("Period 1", pos_period_1.popt, pos_period_1.p_sigma)
    # print(pos_period_1.starting_angle, pos_period_1.period, pos_period_1.period_uncertainty)
    # print(pos_period_1.chi_squared_prob)




    ''''''

    # ====================================================================================================================

    # Theta Not Investigation
    # Using Length 3 Mass 2

    pos_period_objects_0 = period_generator(r_l3m2, [1.15, 0.85, 0], positive_group_angles_information_extraction, 6, 0)
    neg_period_objects_0 = period_generator(r_l3m2, [1.15, 0.85, 0], negative_group_angles_information_extraction, 6, 0)

    pos_period_objects_1 = period_generator(r_l3m2, [1.1173184, 0.85, 0], positive_group_angles_information_extraction, 97, 0)
    neg_period_objects_1 = period_generator(r_l3m2, [1.1173184, 0.85, 0], negative_group_angles_information_extraction, 97, 0)

    pos_period_objects_2 = period_generator(r_l3m2, [1.113, 0.85, 0], positive_group_angles_information_extraction, 97, 60)
    neg_period_objects_2 = period_generator(r_l3m2, [1.113, 0.85, 0], negative_group_angles_information_extraction, 97, 60)

    pos_period_objects_3 = period_generator(r_l3m2, [1.109, 0.85, 0], positive_group_angles_information_extraction, 97, 71)
    neg_period_objects_3 = period_generator(r_l3m2, [1.109, 0.85, 0], negative_group_angles_information_extraction, 97, 71)

    pos_period_objects_4 = period_generator(r_l3m2, [1.105, 0.85, 0], positive_group_angles_information_extraction, 97, 83)
    neg_period_objects_4 = period_generator(r_l3m2, [1.105, 0.85, 0], negative_group_angles_information_extraction, 97, 83)


    plt.figure()

    # print(pos_period_objects["period_0"].starting_angle)
    # print(pos_period_objects.keys())

    pos_period_angles2_0 = []
    pos_period_periods2_0 = []

    for i in pos_period_objects_0.values():
        pos_period_angles2_0.append(i.starting_angle)
        pos_period_periods2_0.append(i.period)


    neg_period_angles2_0 = []
    neg_period_periods2_0 = []
    for i in neg_period_objects_0.values():
        neg_period_angles2_0.append(i.starting_angle)
        neg_period_periods2_0.append(i.period)

    pos_period_angles2_1 = []
    pos_period_periods2_1 = []

    for i in pos_period_objects_1.values():
        pos_period_angles2_1.append(i.starting_angle)
        pos_period_periods2_1.append(i.period)


    neg_period_angles2_1 = []
    neg_period_periods2_1 = []
    for i in neg_period_objects_1.values():
        neg_period_angles2_1.append(i.starting_angle)
        neg_period_periods2_1.append(i.period)

    pos_period_angles2_2 = []
    pos_period_periods2_2 = []

    for i in pos_period_objects_2.values():
        pos_period_angles2_2.append(i.starting_angle)
        pos_period_periods2_2.append(i.period)


    neg_period_angles2_2 = []
    neg_period_periods2_2 = []
    for i in neg_period_objects_2.values():
        neg_period_angles2_2.append(i.starting_angle)
        neg_period_periods2_2.append(i.period)

    pos_period_angles2_3 = []
    pos_period_periods2_3 = []

    for i in pos_period_objects_3.values():
        pos_period_angles2_3.append(i.starting_angle)
        pos_period_periods2_3.append(i.period)


    neg_period_angles2_3 = []
    neg_period_periods2_3 = []
    for i in neg_period_objects_3.values():
        neg_period_angles2_3.append(i.starting_angle)
        neg_period_periods2_3.append(i.period)


    pos_period_angles2_4 = []
    pos_period_periods2_4 = []

    for i in pos_period_objects_4.values():
        pos_period_angles2_4.append(i.starting_angle)
        pos_period_periods2_4.append(i.period)


    neg_period_angles2_4 = []
    neg_period_periods2_4 = []
    for i in neg_period_objects_4.values():
        neg_period_angles2_4.append(i.starting_angle)
        neg_period_periods2_4.append(i.period)

    # plt.errorbar(pos_period_angles2_0, pos_period_periods2_0, marker="o", ls='', label=" Positive Angle Raw Data")
    # plt.errorbar(neg_period_angles2_0, neg_period_periods2_0, marker="o", ls='', label=" Negative Angle Raw Data")
    plt.errorbar(pos_period_angles2_1[:60], pos_period_periods2_1[:60], marker="o", ls='', label=" Positive Angle Raw Data")
    plt.errorbar(neg_period_angles2_1[:60], neg_period_periods2_1[:60], marker="o", ls='', label=" Negative Angle Raw Data")
    plt.errorbar(pos_period_angles2_2[:11], pos_period_periods2_2[:11], marker="o", ls='', label=" Positive Angle Raw Data")
    plt.errorbar(neg_period_angles2_2[:11], neg_period_periods2_2[:11], marker="o", ls='', label=" Negative Angle Raw Data")
    plt.errorbar(pos_period_angles2_3[:12], pos_period_periods2_3[:12], marker="o", ls='', label=" Positive Angle Raw Data")
    plt.errorbar(neg_period_angles2_3[:12], neg_period_periods2_3[:12], marker="o", ls='', label=" Negative Angle Raw Data")
    plt.errorbar(pos_period_angles2_4, pos_period_periods2_4, marker="o", ls='', label=" Positive Angle Raw Data")
    plt.errorbar(neg_period_angles2_4, neg_period_periods2_4, marker="o", ls='', label=" Negative Angle Raw Data")
    plt.title("Period - Angle Graph L3M2")

    plt.show()

    # =====================================================================================================================

    # Quantitative assessment of asymmetry of pendulum


    # =====================================================================================================================

    # Verify or refute that decay is exponential




    # =====================================================================================================================



    # =====================================================================================================================