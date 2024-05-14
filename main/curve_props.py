import numpy as np
import pandas as pd
from scipy.integrate import simps
import matplotlib.pyplot as plt

def calculate_curvature(x, y):
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / ((dx_dt**2 + dy_dt**2)**(3/2))
    return curvature

# function for calculating the curvature data for a stress-strain dataframe
def curvature(data: pd.DataFrame):
    x = data['strain']
    y = data['stress']
    return calculate_curvature(x, y)


def total_curvature_with_arc_len(data: pd.DataFrame):
    x = data['strain']
    y = data['stress']
    kappa = calculate_curvature(x, y)
    kappa_mod = kappa[:-1]
    arc_length = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    total_curvature = simps(kappa_mod, arc_length)

    return total_curvature

def total_curvature(data: pd.DataFrame) -> float:
    kappa = curvature(data)
    total_curvature = np.sum(kappa)

    return total_curvature

def total_abs_curvature(data: pd.DataFrame) -> float:
    kappa = curvature(data)
    total_abs_curvature = np.sum(np.abs(kappa))

    return total_abs_curvature

# curvature_ratio = total inferior curvature / total superior curvature
def get_curvature_ratio(data: pd.DataFrame):
    curv = curvature(data)
    total_abs_negative_curvature = np.sum(np.abs(curv[curv < 0]))
    total_abs_positive_curvature = np.sum(np.abs(curv[curv > 0]))

    inferior_curvature = min(total_abs_negative_curvature, total_abs_positive_curvature)
    superior_curvature = max(total_abs_negative_curvature, total_abs_positive_curvature)
    
    return inferior_curvature / superior_curvature

def plot_curvature(data: pd.DataFrame):
    x = data['strain']
    y = data['stress']
    kappa = calculate_curvature(x, y)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, kappa, 'g-')
    ax2.plot(x, y, 'b-')
    ax1.set_xlabel('x')
    ax1.set_ylabel('kappa', color='g')
    ax2.set_ylabel('y', color='b')
    plt.show()
