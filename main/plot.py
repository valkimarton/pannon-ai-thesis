import matplotlib.pyplot as plt
import pandas as pd

# function to plot stress-strain data
def plot_stress_strain(data: pd.DataFrame, title: str):
    plt.plot(data['strain'], data['stress'])
    plt.title(title)
    plt.xlabel('Strain [-]')
    plt.ylabel('Engineering Stress [MPA]')
    plt.show()