import os
import data_prep
import pandas as pd
import curve_props
import plot

SAMPLE_NAME = 'sample_name'
TOTAL_CURVATURE = 'total_curvature'
CURVATURE_RATIO = 'curvature_ratio'
FINAL_STRAIN = 'final_strain'
FINAL_STRESS = 'final_stress'
CLASS = 'class'

EXTRACTED_FEATURES_DATA_FOLDER = './extracted_features_data/'

# function to extract biax tension features for all data models
def extract_all_biax_tension_features() -> pd.DataFrame:
    # Features for Biax Tension
    inputs_biax_nh = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_biax_mr = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_biax_og = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.OGDEN_FOLDER)
    # extract features for biax tension
    features_biax_nh = extract_biax_tension_features_for_model(inputs_biax_nh)
    features_biax_mr = extract_biax_tension_features_for_model(inputs_biax_mr)
    features_biax_og = extract_biax_tension_features_for_model(inputs_biax_og)
    # concatenate all features
    features_biax = pd.concat([features_biax_nh, features_biax_mr, features_biax_og])
    return features_biax

# function to extract planar compression features for all data models
def extract_all_planar_compression_features() -> pd.DataFrame:
    # Features for Planar Compression
    inputs_planarc_nh = data_prep.read_inputs(data_prep.PLANAR_COMPRESSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_planarc_mr = data_prep.read_inputs(data_prep.PLANAR_COMPRESSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_planarc_og = data_prep.read_inputs(data_prep.PLANAR_COMPRESSION_FOLDER, data_prep.OGDEN_FOLDER)
    # extract features for planar compression
    features_planarc_nh = extract_planar_compression_features_for_model(inputs_planarc_nh)
    features_planarc_mr = extract_planar_compression_features_for_model(inputs_planarc_mr)
    features_planarc_og = extract_planar_compression_features_for_model(inputs_planarc_og)
    # concatenate all features
    features_planarc = pd.concat([features_planarc_nh, features_planarc_mr, features_planarc_og])
    return features_planarc

# function to extract planar tension features for all data models
def extract_all_planar_tension_features() -> pd.DataFrame:
    # Features for Planar Tension
    inputs_planart_nh = data_prep.read_inputs(data_prep.PLANAR_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_planart_mr = data_prep.read_inputs(data_prep.PLANAR_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_planart_og = data_prep.read_inputs(data_prep.PLANAR_TENSION_FOLDER, data_prep.OGDEN_FOLDER)
    # extract features for planar tension
    features_planart_nh = extract_planar_tension_features_for_model(inputs_planart_nh)
    features_planart_mr = extract_planar_tension_features_for_model(inputs_planart_mr)
    features_planart_og = extract_planar_tension_features_for_model(inputs_planart_og)
    # concatenate all features
    features_planart = pd.concat([features_planart_nh, features_planart_mr, features_planart_og])
    return features_planart

# function to extract simple shear features for all data models
def extract_all_simple_shear_features() -> pd.DataFrame:
    # Features for Simple Shear
    inputs_shear_nh = data_prep.read_inputs(data_prep.SIMPLE_SHEAR_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_shear_mr = data_prep.read_inputs(data_prep.SIMPLE_SHEAR_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_shear_og = data_prep.read_inputs(data_prep.SIMPLE_SHEAR_FOLDER, data_prep.OGDEN_FOLDER)
    # extract features for simple shear
    features_shear_nh = extract_simple_shear_features_for_model(inputs_shear_nh)
    features_shear_mr = extract_simple_shear_features_for_model(inputs_shear_mr)
    features_shear_og = extract_simple_shear_features_for_model(inputs_shear_og)
    # concatenate all features
    features_shear = pd.concat([features_shear_nh, features_shear_mr, features_shear_og])
    return features_shear

# function to extract uniax compression features for all data models
def extract_all_uniax_compression_features() -> pd.DataFrame:
    # Features for Uniax Compression
    inputs_uniaxc_nh = data_prep.read_inputs(data_prep.UNIAX_COMPRESSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_uniaxc_mr = data_prep.read_inputs(data_prep.UNIAX_COMPRESSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_uniaxc_og = data_prep.read_inputs(data_prep.UNIAX_COMPRESSION_FOLDER, data_prep.OGDEN_FOLDER)
    # extract features for uniax compression
    features_uniaxc_nh = extract_uniax_compression_features_for_model(inputs_uniaxc_nh)
    features_uniaxc_mr = extract_uniax_compression_features_for_model(inputs_uniaxc_mr)
    features_uniaxc_og = extract_uniax_compression_features_for_model(inputs_uniaxc_og)
    # concatenate all features
    features_uniaxc = pd.concat([features_uniaxc_nh, features_uniaxc_mr, features_uniaxc_og])
    return features_uniaxc

# function to extract uniax tension features for all data models
def extract_all_uniax_tension_features() -> pd.DataFrame:
    # Features for Uniax Tension
    inputs_uniaxt_nh = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_uniaxt_mr = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_uniaxt_og = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.OGDEN_FOLDER)
    # extract features for uniax tension
    features_uniaxt_nh = extract_uniax_tension_features_for_model(inputs_uniaxt_nh)
    features_uniaxt_mr = extract_uniax_tension_features_for_model(inputs_uniaxt_mr)
    features_uniaxt_og = extract_uniax_tension_features_for_model(inputs_uniaxt_og)
    # concatenate all features
    features_uniaxt = pd.concat([features_uniaxt_nh, features_uniaxt_mr, features_uniaxt_og])
    return features_uniaxt


# function to extract biax tension features for a data model type
def extract_biax_tension_features_for_model(inputs: dict) -> pd.DataFrame:
    features = pd.DataFrame(columns=[TOTAL_CURVATURE, CURVATURE_RATIO, FINAL_STRAIN, FINAL_STRESS, CLASS])
    for key, value in inputs.items():
        total_curvature = curve_props.total_curvature(value)
        curvature_ratio = curve_props.get_curvature_ratio(value)
        final_strain = value['strain'].iloc[-1]
        final_stress = value['stress'].iloc[-1]
        class_label = get_class(key)
        features.loc[key] = [total_curvature, curvature_ratio, final_strain, final_stress, class_label]
        features.index.name = SAMPLE_NAME
    
    return features

# function to extract planar compression features for a data model type
def extract_planar_compression_features_for_model(inputs: dict) -> pd.DataFrame:
    features = pd.DataFrame(columns=[TOTAL_CURVATURE, FINAL_STRAIN, CLASS])
    for key, value in inputs.items():
        total_curvature = curve_props.total_curvature(value)
        final_strain = value['strain'].iloc[-1]
        class_label = get_class(key)
        features.loc[key] = [total_curvature, final_strain, class_label]
        features.index.name = SAMPLE_NAME
    
    return features

# function to extract planar tension features for a data model type
def extract_planar_tension_features_for_model(inputs: dict) -> pd.DataFrame:
    features = pd.DataFrame(columns=[TOTAL_CURVATURE, CURVATURE_RATIO, FINAL_STRAIN, CLASS])
    for key, value in inputs.items():
        total_curvature = curve_props.total_curvature(value)
        curvature_ratio = curve_props.get_curvature_ratio(value)
        final_strain = value['strain'].iloc[-1]
        class_label = get_class(key)
        features.loc[key] = [total_curvature, curvature_ratio, final_strain, class_label]
        features.index.name = SAMPLE_NAME
    
    return features

# function to extract simple shear features for a data model type
def extract_simple_shear_features_for_model(inputs: dict) -> pd.DataFrame:
    features = pd.DataFrame(columns=[TOTAL_CURVATURE, CURVATURE_RATIO, FINAL_STRAIN, CLASS])
    for key, value in inputs.items():
        total_curvature = curve_props.total_curvature(value)
        curvature_ratio = curve_props.get_curvature_ratio(value)
        final_strain = value['strain'].iloc[-1]
        class_label = get_class(key)
        features.loc[key] = [total_curvature, curvature_ratio, final_strain, class_label]
        features.index.name = SAMPLE_NAME
    
    return features

# function to extract uniax compression features for a data model type
def extract_uniax_compression_features_for_model(inputs: dict) -> pd.DataFrame:
    features = pd.DataFrame(columns=[TOTAL_CURVATURE, CURVATURE_RATIO, FINAL_STRAIN, CLASS])
    for key, value in inputs.items():
        total_curvature = curve_props.total_curvature(value)
        curvature_ratio = curve_props.get_curvature_ratio(value)
        final_strain = value['strain'].iloc[-1]
        class_label = get_class(key)
        features.loc[key] = [total_curvature, curvature_ratio, final_strain, class_label]
        features.index.name = SAMPLE_NAME
    
    return features

# function to extract uniax tension features for a data model type
def extract_uniax_tension_features_for_model(inputs: dict) -> pd.DataFrame:
    features = pd.DataFrame(columns=[TOTAL_CURVATURE, CURVATURE_RATIO, FINAL_STRAIN, FINAL_STRESS, CLASS])
    for key, value in inputs.items():
        print(key)
        total_curvature = curve_props.total_curvature(value)
        curvature_ratio = curve_props.get_curvature_ratio(value)
        final_strain = value['strain'].iloc[-1]
        final_stress = value['stress'].iloc[-1]
        class_label = get_class(key)
        features.loc[key] = [total_curvature, curvature_ratio, final_strain, final_stress, class_label]
        features.index.name = SAMPLE_NAME
    
    return features

def get_class(key: str) -> str:
    if 'neohookean' in key:
        return 'nh'
    elif 'mooneyrivlin2' in key:
        return 'mr'
    elif 'ogden' in key:
        return 'og'
    else:
        raise ValueError('Cannot determine class')


def extract_features_for_all():
    df_biax_tension = extract_all_biax_tension_features()
    df_planar_compression = extract_all_planar_compression_features()
    df_planar_tension = extract_all_planar_tension_features()
    df_simple_shear = extract_all_simple_shear_features()
    df_uniax_compression = extract_all_uniax_compression_features()
    df_uniax_tension = extract_all_uniax_tension_features()

    extracted_features_data_folder = EXTRACTED_FEATURES_DATA_FOLDER
    if not os.path.exists(extracted_features_data_folder):
        os.makedirs(extracted_features_data_folder)
        
    df_biax_tension.to_csv(extracted_features_data_folder + 'biax_tension_features.csv')
    df_planar_compression.to_csv(extracted_features_data_folder + 'planar_compression_features.csv')
    df_planar_tension.to_csv(extracted_features_data_folder + 'planar_tension_features.csv')
    df_simple_shear.to_csv(extracted_features_data_folder + 'simple_shear_features.csv')
    df_uniax_compression.to_csv(extracted_features_data_folder + 'uniax_compression_features.csv')
    df_uniax_tension.to_csv(extracted_features_data_folder + 'uniax_tension_features.csv')

    plot.create_correlation_matrix(df_biax_tension[[TOTAL_CURVATURE, CURVATURE_RATIO, FINAL_STRAIN, FINAL_STRESS]])
    plot.create_correlation_matrix(df_planar_compression[[TOTAL_CURVATURE, FINAL_STRAIN]])
    plot.create_correlation_matrix(df_planar_tension[[TOTAL_CURVATURE, CURVATURE_RATIO, FINAL_STRAIN]])
    plot.create_correlation_matrix(df_simple_shear[[TOTAL_CURVATURE, CURVATURE_RATIO, FINAL_STRAIN]])
    plot.create_correlation_matrix(df_uniax_compression[[TOTAL_CURVATURE, CURVATURE_RATIO, FINAL_STRAIN]])
    plot.create_correlation_matrix(df_uniax_tension[[TOTAL_CURVATURE, CURVATURE_RATIO, FINAL_STRAIN]])

if __name__ == '__main__':
    extract_features_for_all()