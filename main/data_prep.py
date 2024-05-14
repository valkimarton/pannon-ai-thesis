import pandas as pd
import numpy as np
import os

DATA_FOLDER = './data/'
BIAX_TENSION_FOLDER = 'biax_tension/'
PLANAR_COMPRESSION_FOLDER = 'planar_compression/'
PLANAR_TENSION_FOLDER = 'planar_tension/'
SIMPLE_SHEAR_FOLDER = 'simple_shear/'
UNIAX_COMPRESSION_FOLDER = 'uniax_compression/'
UNIAX_TENSION_FOLDER = 'uniax_tension/'

NEO_HOOKEAN_FOLDER = 'neohookean/'
MOONEY_RIVLIN_FOLDER = 'mooneyrivlin2/'
OGDEN_FOLDER = 'ogden/'

# read biax_tension data into dataFrames
# - neo-hookean
def read_biax_tension_neo_hookean() -> dict:
    biax_tension_neo_hookean = {}
    data_folder = './data/biax_tension/neohookean'
    for file in os.listdir(data_folder):
        if file.endswith('.txt'):
            # read data. values are separated by simple space
            data = pd.read_csv(data_folder + '/' + file, sep=' ', header=None)
            data.columns = ['X', 'Y']
            data_point_number = file[-7:-4]
            biax_tension_neo_hookean[data_point_number] = data
            return biax_tension_neo_hookean
# - mooney-rivlin
# - ogden

def read_inputs(tension_folder: str, mat_model_folder: str) -> dict:
    inputs = {}
    data_folder = DATA_FOLDER + tension_folder + mat_model_folder
    for file in os.listdir(data_folder):
        if file.endswith('.txt'):
            data = pd.read_csv(data_folder + '/' + file, sep='  ', header=None, engine='python')
            # remove rows where all values are 0
            data = data[(data.T != 0).any()]
            # remove if there is a 3rd columsn
            if len(data.columns) == 3:
                data = data.drop(data.columns[2], axis=1)
            data.columns = ['stress', 'strain']   #TODO
            data_point_number = file[:-4]
            inputs[data_point_number] = data
    return inputs



