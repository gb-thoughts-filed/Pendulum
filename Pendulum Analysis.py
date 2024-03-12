import numpy as np
import matplotlib.pyplot as plt

class PendulumData:

    def __init__(self, text_file: str):
        self.frames = None
        self.angles = None
        self.time = None
        self.text_file_ingestion(text_file)

        self.angle_uncertainties = None
        self.time_uncertainties = None
        self.uncertainty_population(len(self.angles))

        self.angles_no_duplicates = list(set(self.angles)).sort()




    def text_file_ingestion(self, text_file:str) -> None:

        self.time, self.angles, self.frames = np.loadtxt(text_file,
                                                         skiprows=2, delimiter=",", dtype=str, unpack=True)

        return

    def uncertainty_population(self, list_length: int) -> None:

        x = np.arange(list_length, dtype=int)
        self.angle_uncertainties = np.full_like(x, 0.05, dtype=float)
        self.time = np.full_like(x, 0.0005)

        return

    def period_calculation(self, angle_no_duplicates: list, angles: list, times: list) -> None:

        for i in angle_no_duplicates:
            angle_index = angles.index(i)

            start_time = times.index(i)

            modified_angles = angles[angle_index+1:]




