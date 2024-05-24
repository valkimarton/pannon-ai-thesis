import os
import data_prep
import matplotlib.pyplot as plt
import pandas as pd

# function that saves the plot to a file
def save_plot_stress_strain(data: pd.DataFrame, title: str, file_name: str):
    plt.plot(data['strain'], data['stress'])
    plt.title(title)
    plt.xlabel('Strain [-]')
    plt.ylabel('Engineering Stress [MPA]')
    plt.savefig(file_name)
    plt.close()

# function to save image of all data for an tension-type
def generate_diagrams(inputs: dict, save_folder: str):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for key, value in inputs.items():
        file_name = key + '.png'
        save_plot_stress_strain(value, key, save_folder + file_name)

def generate_all_diagrams():
    inputs_biax_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_biax_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_biax_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.OGDEN_FOLDER)

    inputs_planarc_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_COMPRESSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_planarc_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_COMPRESSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_planarc_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_COMPRESSION_FOLDER, data_prep.OGDEN_FOLDER)

    inputs_planart_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_planart_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_planart_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_TENSION_FOLDER, data_prep.OGDEN_FOLDER)

    inputs_shear_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.SIMPLE_SHEAR_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_shear_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.SIMPLE_SHEAR_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_shear_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.SIMPLE_SHEAR_FOLDER, data_prep.OGDEN_FOLDER)
    
    inputs_uniaxc_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_COMPRESSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_uniaxc_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_COMPRESSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_uniaxc_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_COMPRESSION_FOLDER, data_prep.OGDEN_FOLDER)

    inputs_uniaxt_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_uniaxt_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_uniaxt_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.OGDEN_FOLDER)

    generate_diagrams(inputs_biax_nh, 'diagrams/biax_tension/neohoohean/')
    generate_diagrams(inputs_biax_mr, 'diagrams/biax_tension/mooneyrivlin2/')
    generate_diagrams(inputs_biax_og, 'diagrams/biax_tension/ogden/')
    generate_diagrams(inputs_planarc_nh, 'diagrams/planar_compression/neohoohean/')
    generate_diagrams(inputs_planarc_mr, 'diagrams/planar_compression/mooneyrivlin2/')
    generate_diagrams(inputs_planarc_og, 'diagrams/planar_compression/ogden/')
    generate_diagrams(inputs_planart_nh, 'diagrams/planar_tension/neohoohean/')
    generate_diagrams(inputs_planart_mr, 'diagrams/planar_tension/mooneyrivlin2/')
    generate_diagrams(inputs_planart_og, 'diagrams/planar_tension/ogden/')
    generate_diagrams(inputs_shear_nh, 'diagrams/simple_shear/neohoohean/')
    generate_diagrams(inputs_shear_mr, 'diagrams/simple_shear/mooneyrivlin2/')
    generate_diagrams(inputs_shear_og, 'diagrams/simple_shear/ogden/')
    generate_diagrams(inputs_uniaxc_nh, 'diagrams/uniax_compression/neohoohean/')
    generate_diagrams(inputs_uniaxc_mr, 'diagrams/uniax_compression/mooneyrivlin2/')
    generate_diagrams(inputs_uniaxc_og, 'diagrams/uniax_compression/ogden/')
    generate_diagrams(inputs_uniaxt_nh, 'diagrams/uniax_tension/neohoohean/')
    generate_diagrams(inputs_uniaxt_mr, 'diagrams/uniax_tension/mooneyrivlin2/')
    generate_diagrams(inputs_uniaxt_og, 'diagrams/uniax_tension/ogden/')

# generate_all_diagrams()
    