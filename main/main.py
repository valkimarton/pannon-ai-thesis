import data_prep
import plot
import diagram_generator
import curve_props

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

    # TODO: handle uniax_tension's 3 columns
    # inputs_uniaxt_nh = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    # inputs_uniaxt_mr = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    # inputs_uniaxt_og = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.OGDEN_FOLDER)

    # diagram_generator.generate_all_diagrams() # runs for about 5-7 minutes

    # nh dp0
    kappa_nh_dp0 = curve_props.total_curvature(inputs_biax_nh['biax_tension_neohookean_dp0'])
    kappa_nh_dp0_with_arc_len = curve_props.total_curvature_with_arc_len(inputs_biax_nh['biax_tension_neohookean_dp0'])
    curve_props.plot_curvature(inputs_biax_nh['biax_tension_neohookean_dp0'])

    # mr dp0
    kappa_mr_dp0 = curve_props.total_curvature(inputs_biax_mr['biax_tension_mooneyrivlin2_dp0'])
    kappa_mr_dp0_with_arc_len = curve_props.total_curvature_with_arc_len(inputs_biax_mr['biax_tension_mooneyrivlin2_dp0'])
    curve_props.plot_curvature(inputs_biax_mr['biax_tension_mooneyrivlin2_dp0'])

    # og dp0
    kappa_og_dp0 = curve_props.total_curvature(inputs_biax_og['biax_tension_ogden_dp0'])
    kappa_og_dp0_with_arc_len = curve_props.total_curvature_with_arc_len(inputs_biax_og['biax_tension_ogden_dp0'])
    curve_props.plot_curvature(inputs_biax_og['biax_tension_ogden_dp0'])

    # og dp20
    kappa_og_dp20 = curve_props.total_curvature(inputs_biax_og['biax_tension_ogden_dp20'])
    kappa_og_dp20_with_arc_len = curve_props.total_curvature_with_arc_len(inputs_biax_og['biax_tension_ogden_dp20'])
    curve_props.plot_curvature(inputs_biax_og['biax_tension_ogden_dp20'])

    print(kappa_nh_dp0, kappa_nh_dp0_with_arc_len)
    print(kappa_mr_dp0, kappa_mr_dp0_with_arc_len)
    print(kappa_og_dp0, kappa_og_dp0_with_arc_len)
    print(kappa_og_dp20, kappa_og_dp20_with_arc_len)


    '''Baseline properties:
     - Max strain
     - Max stress
     - Total signed curvature (use simple total_curvature function)
     - Total absolute curvature (use simple total_curvature function)
    '''



    


'''
TODO:
    - Add class label to data
    - Add data to a single DataFrame

    - feature extraction:
        curvature / multiple segment curvature
    
    - traditional classification (random forest, SVM, etc.) as baseline

    - CNN

    - Investigate: Functional Data Analysis
        - functional data representation
        - functional data classification
'''