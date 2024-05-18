import data_prep

if __name__ == '__main__':
    inputs_biax_nh = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_biax_mr = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_biax_og = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.OGDEN_FOLDER)

    inputs_planarc_nh = data_prep.read_inputs(data_prep.PLANAR_COMPRESSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_planarc_mr = data_prep.read_inputs(data_prep.PLANAR_COMPRESSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_planarc_og = data_prep.read_inputs(data_prep.PLANAR_COMPRESSION_FOLDER, data_prep.OGDEN_FOLDER)

    inputs_planart_nh = data_prep.read_inputs(data_prep.PLANAR_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_planart_mr = data_prep.read_inputs(data_prep.PLANAR_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_planart_og = data_prep.read_inputs(data_prep.PLANAR_TENSION_FOLDER, data_prep.OGDEN_FOLDER)

    inputs_shear_nh = data_prep.read_inputs(data_prep.SIMPLE_SHEAR_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_shear_mr = data_prep.read_inputs(data_prep.SIMPLE_SHEAR_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_shear_og = data_prep.read_inputs(data_prep.SIMPLE_SHEAR_FOLDER, data_prep.OGDEN_FOLDER)
    
    inputs_uniaxc_nh = data_prep.read_inputs(data_prep.UNIAX_COMPRESSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_uniaxc_mr = data_prep.read_inputs(data_prep.UNIAX_COMPRESSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_uniaxc_og = data_prep.read_inputs(data_prep.UNIAX_COMPRESSION_FOLDER, data_prep.OGDEN_FOLDER)

    inputs_uniaxt_nh = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_uniaxt_mr = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_uniaxt_og = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.OGDEN_FOLDER)


'''
TODO:

    - Network Analysis

    - CNN
'''