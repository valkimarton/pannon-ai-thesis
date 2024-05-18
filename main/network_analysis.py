import data_prep
import pandas as pd
import numpy as np
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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

def max_first_strain_biax_tension(inputs_nh: dict[str, pd.DataFrame], inputs_mr: dict[str, pd.DataFrame], inputs_og: dict[str, pd.DataFrame]) -> float:
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
        interp_func = interp1d(data['strain'], data['stress'], kind='linear', fill_value=0, bounds_error=False)
        resampled_stress = interp_func(resampled_strain)
        resampled_biax_tension_data[key] = pd.DataFrame({'strain': resampled_strain, 'stress': resampled_stress})


if __name__ == '__main__':
    inputs_biax_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_biax_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_biax_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.OGDEN_FOLDER)

    # TODO:
    # KNN gráffal gráf a mintákból -> megnézni hogy kialakul-e a három klaszter
    # Hasonlóság:
    # -	Dinamikus idősorvetemítéssel
    # Interpolálni a görbéket úgy hogy azonos strain értékekkel dolgozzak: 
    # - Kitölteni a rövideket 0-kkal és ezután mehet a Dinamikus idősorvetemítés

    # find min and max strain interval for each stress type
    min_strain_diff_biax_tension, max_strain_diff_biax_tension = find_min_and_max_strain_interval_for_stress_type(inputs_biax_nh, inputs_biax_mr, inputs_biax_og)
    # find optimal strain interval before resampling data
    strain_interval_biax_tension: float = min_strain_diff_biax_tension / 10
    # determine number of points for resampling - round down to nearest integer max / interval


    # determine 90th percentile of max_strain_diff
    strain_diff_90th_percentile = determine_max_strain_diff_90th_percentile(inputs_biax_nh, inputs_biax_mr, inputs_biax_og)

    sample_size_biax_tension: int = int(strain_diff_90th_percentile / strain_interval_biax_tension)
    max_first_strain_biax_tension: float = max_first_strain_biax_tension(inputs_biax_nh, inputs_biax_mr, inputs_biax_og)

    # resample every stress data to strain_interval and sample_size_biax_tension then fill missing values with 0
    resampled_strain = np.linspace(
        max_first_strain_biax_tension, 
        max_first_strain_biax_tension + strain_interval_biax_tension * sample_size_biax_tension, 
        sample_size_biax_tension)
    # resampled_biax_nh: dict[str, pd.DataFrame] = resample_biax_tension(inputs_biax_nh, resampled_strain)
    # resampled_biax_mr: dict[str, pd.DataFrame] = resample_biax_tension(inputs_biax_mr, resampled_strain)
    # resampled_biax_og: dict[str, pd.DataFrame] = resample_biax_tension(inputs_biax_og, resampled_strain)

    bt_nh_0: pd.DataFrame = inputs_biax_nh['biax_tension_neohookean_dp0']
    bt_mr_0: pd.DataFrame = inputs_biax_mr['biax_tension_mooneyrivlin2_dp0']
    bt_og_56: pd.DataFrame = inputs_biax_og['biax_tension_ogden_dp56']

    resampled_bt_nh_0: pd.DataFrame = resample_single_biax_tension(bt_nh_0, resampled_strain)
    resampled_bt_mr_0: pd.DataFrame = resample_single_biax_tension(bt_mr_0, resampled_strain)
    resampled_bt_og_56: pd.DataFrame = resample_single_biax_tension(bt_og_56, resampled_strain)

    # plot original and resampled data
    plt.plot(bt_nh_0['strain'], bt_nh_0['stress'], label='neohookean dp0')
    plt.plot(resampled_bt_nh_0['strain'], resampled_bt_nh_0['stress'], label='resampled neohookean dp0')
    plt.legend()
    plt.show()

    plt.plot(bt_mr_0['strain'], bt_mr_0['stress'], label='mooneyrivlin2 dp0')
    plt.plot(resampled_bt_mr_0['strain'], resampled_bt_mr_0['stress'], label='resampled mooneyrivlin2 dp0')
    plt.legend()
    plt.show()

    plt.plot(bt_og_56['strain'], bt_og_56['stress'], label='ogden dp56')
    plt.plot(resampled_bt_og_56['strain'], resampled_bt_og_56['stress'], label='resampled ogden dp56')
    plt.legend()
    plt.show()


    
    # inputs_planarc_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_COMPRESSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    # inputs_planarc_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_COMPRESSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    # inputs_planarc_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_COMPRESSION_FOLDER, data_prep.OGDEN_FOLDER)

    # inputs_planart_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    # inputs_planart_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    # inputs_planart_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_TENSION_FOLDER, data_prep.OGDEN_FOLDER)

    # inputs_shear_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.SIMPLE_SHEAR_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    # inputs_shear_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.SIMPLE_SHEAR_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    # inputs_shear_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.SIMPLE_SHEAR_FOLDER, data_prep.OGDEN_FOLDER)
    
    # inputs_uniaxc_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_COMPRESSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    # inputs_uniaxc_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_COMPRESSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    # inputs_uniaxc_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_COMPRESSION_FOLDER, data_prep.OGDEN_FOLDER)

    # inputs_uniaxt_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    # inputs_uniaxt_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    # inputs_uniaxt_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.OGDEN_FOLDER)