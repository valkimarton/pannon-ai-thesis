import os
import data_prep
import pandas as pd
import numpy as np
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

RESAMPLED_DATA_FOLDER = './resampled_data/'

BIAX_TENSION_FOLDER = 'biax_tension/'
PLANAR_COMPRESSION_FOLDER = 'planar_compression/'
PLANAR_TENSION_FOLDER = 'planar_tension/'
SIMPLE_SHEAR_FOLDER = 'simple_shear/'
UNIAX_COMPRESSION_FOLDER = 'uniax_compression/'
UNIAX_TENSION_FOLDER = 'uniax_tension/'

NEO_HOOKEAN_FOLDER = 'neohookean/'
MOONEY_RIVLIN_FOLDER = 'mooneyrivlin2/'
OGDEN_FOLDER = 'ogden/'

# function to calculate similarity with dynamic time warping
def dtw_similarity(data1: pd.DataFrame, data2: pd.DataFrame) -> float:
    # calculate dynamic time warping
    n = len(data1)
    m = len(data2)
    dtw = np.zeros((n+1, m+1))
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(data1[i-1] - data2[j-1])
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
    
    return dtw[n][m]

def get_max_first_strain_biax_tension(inputs_nh: dict[str, pd.DataFrame], inputs_mr: dict[str, pd.DataFrame], inputs_og: dict[str, pd.DataFrame]) -> float:
    max_first_strain = 0
    for inputs in [inputs_nh, inputs_mr, inputs_og]:
        for key, data in inputs.items():
            first_strain = data['strain'][0]
            if first_strain > max_first_strain:
                max_first_strain = first_strain
    return max_first_strain

def find_min_and_max_strain_interval_for_stress_type(
        inputs_nh: dict[str, pd.DataFrame], 
        inputs_mr: dict[str, pd.DataFrame], 
        inputs_og: dict[str, pd.DataFrame]) -> tuple[float, float]:
    min_strain_interval = float('inf')
    max_strain_interval = 0
    for inputs in [inputs_nh, inputs_mr, inputs_og]:
        for key, data in inputs.items():
            strain_interval = data['strain'].max() - data['strain'].min()
            if strain_interval < min_strain_interval:
                min_strain_interval = strain_interval
            if strain_interval > max_strain_interval:
                max_strain_interval = strain_interval
    
    print('min_strain_interval for biax_tension: ' + str(min_strain_interval))
    print('max_strain_interval for biax_tension: ' + str(max_strain_interval))
    return min_strain_interval, max_strain_interval

def determine_max_strain_diff_90th_percentile(inputs_nh: dict[str, pd.DataFrame], inputs_mr: dict[str, pd.DataFrame], inputs_og: dict[str, pd.DataFrame]) -> float:
    max_strain_diffs = []
    for inputs in [inputs_nh, inputs_mr, inputs_og]:
        for key, data in inputs.items():
            strain_diff = data['strain'].max() - data['strain'].min()
            max_strain_diffs.append(strain_diff)
    
    max_strain_diff_90th_percentile = np.percentile(max_strain_diffs, 90)
    print('max_strain_diff_90th_percentile for biax_tension: ' + str(max_strain_diff_90th_percentile))
    return max_strain_diff_90th_percentile

def resample_single_biax_tension(input: pd.DataFrame, resampled_strain: np.ndarray) -> pd.DataFrame:
    interp_func = interp1d(input['strain'], input['stress'], kind='linear', fill_value=0, bounds_error=False)
    resampled_stress = interp_func(resampled_strain)
    return pd.DataFrame({'strain': resampled_strain, 'stress': resampled_stress})

def resample_biax_tension(inputs: dict[str, pd.DataFrame], resampled_strain: np.ndarray) -> dict[str, pd.DataFrame]:
    resampled_biax_tension_data: dict[str, pd.DataFrame] = {}
    for key, data in inputs.items():
        resampled_biax_tension_data[key] = resample_single_biax_tension(data, resampled_strain)
    return resampled_biax_tension_data
    

def save_resampled_data(resampled_nh: dict[str, pd.DataFrame], 
                        resampled_mr: dict[str, pd.DataFrame], 
                        resampled_og: dict[str, pd.DataFrame],
                        stress_type_folder: str):
    stress_type_folder_full = RESAMPLED_DATA_FOLDER + stress_type_folder
    for key, data in resampled_nh.items():
        folder = stress_type_folder_full + NEO_HOOKEAN_FOLDER
        if not os.path.exists(folder):
            os.makedirs(folder)
        data.to_csv(folder + key + '.txt', sep=',', index=False)
    for key, data in resampled_mr.items():
        folder = stress_type_folder_full + MOONEY_RIVLIN_FOLDER
        if not os.path.exists(folder):
            os.makedirs(folder)
        data.to_csv(folder + key + '.txt', sep=',', index=False)
    for key, data in resampled_og.items():
        folder = stress_type_folder_full + OGDEN_FOLDER
        if not os.path.exists(folder):
            os.makedirs(folder)
        data.to_csv(folder + key + '.txt', sep=',', index=False)

def create_resampled_inputs():
    inputs_biax_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_biax_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_biax_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.OGDEN_FOLDER)

    min_strain_diff_biax_tension, max_strain_diff_biax_tension = find_min_and_max_strain_interval_for_stress_type(inputs_biax_nh, inputs_biax_mr, inputs_biax_og)
    strain_interval_biax_tension: float = min_strain_diff_biax_tension / 10
    strain_diff_90th_percentile = determine_max_strain_diff_90th_percentile(inputs_biax_nh, inputs_biax_mr, inputs_biax_og)
    sample_size_biax_tension: int = int(strain_diff_90th_percentile / strain_interval_biax_tension)
    max_first_strain_biax_tension: float = get_max_first_strain_biax_tension(inputs_biax_nh, inputs_biax_mr, inputs_biax_og)

    # resample every stress data to strain_interval and sample_size_biax_tension then fill missing values with 0
    resampled_strain = np.linspace(0, strain_interval_biax_tension * sample_size_biax_tension, sample_size_biax_tension)
    resampled_biax_nh: dict[str, pd.DataFrame] = resample_biax_tension(inputs_biax_nh, resampled_strain)
    resampled_biax_mr: dict[str, pd.DataFrame] = resample_biax_tension(inputs_biax_mr, resampled_strain)
    resampled_biax_og: dict[str, pd.DataFrame] = resample_biax_tension(inputs_biax_og, resampled_strain)

    # save resampled data
    save_resampled_data(resampled_biax_nh, resampled_biax_mr, resampled_biax_og, BIAX_TENSION_FOLDER)

def read_resampled_data(data_folder: str) -> dict[str, pd.DataFrame]:
    resampled_data: dict[str, pd.DataFrame] = {}
    for file in os.listdir(data_folder):
        if file.endswith('.txt'):
            data = pd.read_csv(data_folder + file, sep=',')
            data.columns = ['strain', 'stress']
            data_point_number = file[:-4]
            resampled_data[data_point_number] = data
    return resampled_data



if __name__ == '__main__':
    #create_resampled_inputs()

    resampled_biax_nh: dict[str, pd.DataFrame] = read_resampled_data(RESAMPLED_DATA_FOLDER + BIAX_TENSION_FOLDER + NEO_HOOKEAN_FOLDER)
    resampled_biax_mr: dict[str, pd.DataFrame] = read_resampled_data(RESAMPLED_DATA_FOLDER + BIAX_TENSION_FOLDER + MOONEY_RIVLIN_FOLDER)
    resampled_biax_og: dict[str, pd.DataFrame] = read_resampled_data(RESAMPLED_DATA_FOLDER + BIAX_TENSION_FOLDER + OGDEN_FOLDER)

    # calculate similarity with dynamic time warping

    # calculate dynamic time warping similarity between two inputs
    df1 = resampled_biax_nh['biax_tension_neohookean_dp0']
    df2 = resampled_biax_nh['biax_tension_neohookean_dp1']
    df3 = resampled_biax_mr['biax_tension_mooneyrivlin2_dp0']
    df4 = resampled_biax_og['biax_tension_ogden_dp0']
    df5 = resampled_biax_og['biax_tension_ogden_dp1']

    # convert 'stress' column to numpy array
    stress1 = df1['stress'].to_numpy()
    stress2 = df2['stress'].to_numpy()
    stress3 = df3['stress'].to_numpy()
    stress4 = df4['stress'].to_numpy()
    stress5 = df5['stress'].to_numpy()



    # calculate dynamic time warping similarity between two inputs

    distance, path = fastdtw(stress1, stress2)
    print('Distance between df1 and df2: ' + str(distance))

    distance, path = fastdtw(stress1, stress3)
    print('Distance between df1 and df3: ' + str(distance))

    distance, path = fastdtw(stress1, stress4)
    print('Distance between df1 and df4: ' + str(distance))

    distance, path = fastdtw(stress4, stress5)
    print('Distance between df4 and df5: ' + str(distance))
