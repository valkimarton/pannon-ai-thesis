import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from seaborn import heatmap

import data_prep
import feature_extractor

EVALUATE_ENABLED = False

'''
TODO

DATA PREPROCESSING
modify inputs to have 100 datapoints
 IF more: interpolate to 100
 IF less: pad with zeros / (or interpolate to 100 IF  I have time)
Scale inputs

MODEL
Build 2D CNN
  FIX HYPERPARAMETERS:
    layers: 1 conv, 0 pooling, 2 dense --> Nem túl sok paraméter ez?
    activation in hidden layers: ReLU
    activation in output layer: softmax
    optimizer: adam

  HYPERPARAMETER TUNING
    - Number of filters --> 2-16
    - Filter size (MxM) --> 2-16
'''

BIAX_TENSION = 'biax_tension'
PLANAR_COMPRESSION = 'planar_compression'
PLANAR_TENSION = 'planar_tension'
SIMPLE_SHEAR = 'simple_shear'
UNIAX_COMPRESSION = 'uniax_compression'
UNIAX_TENSION = 'uniax_tension'

NEO_HOOKEAN = 0
MOONEY_RIVLIN = 1
OGDEN = 2

DATA_POINT = 'data_point'

TRAIN_INPUTS = 'train_inputs'
TRAIN_LABELS = 'train_labels'
TEST_INPUTS = 'test_inputs'
TEST_LABELS = 'test_labels'
VAL_INPUTS = 'val_inputs'
VAL_LABELS = 'val_labels'

class InputData:
    def __init__(self, data_point: int, data: pd.DataFrame, class_label: int):
        self.data_point = data_point
        self.data = data
        self.class_label = class_label
    
    def __str__(self):
        return f'Data point: {self.data_point}, Class: {self.class_label}, Data: {self.data}'   

def make_input_lengths_equal(inputs: list[InputData]):
    # Modify inputs to have 100 datapoints
    # IF more: interpolate to 100
    # IF less: pad with zeros
    for input_data in inputs:
        if len(input_data.data) < 100:
            # pad with zeros
            input_data.data = input_data.data.reindex(range(100), fill_value=0)
        elif len(input_data.data) > 100:
            # downsample to 100
            # select first 100
            input_data.data = input_data.data.iloc[:100]  # TODO: enhance this

def scale_dataframe(data:pd.DataFrame):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns)
    return scaled_data

def scale_data(inputs: dict[str, pd.DataFrame]):
    for key, data in inputs.items():
        scaled_data = scale_dataframe(data)
        inputs[key] = scaled_data
    return inputs

def get_class(key: str) -> int:
    if 'neohookean' in key:
        return NEO_HOOKEAN
    elif 'mooneyrivlin' in key:
        return MOONEY_RIVLIN
    elif 'ogden' in key:
        return OGDEN
    else:
        raise ValueError('Cannot determine class')

def get_input_data(inputs: dict[str, pd.DataFrame]) -> list[InputData]:
    input_data: list[InputData] = []
    for data_point_name, df in inputs.items():
        data_point = int(data_point_name.split('_')[-1][2:])
        # if data_point_name contains 'mooneyrivlin' add 100 to data_point, if 'ogden' add 200
        if data_point_name.find('mooneyrivlin') != -1:
            data_point += 100
        if data_point_name.find('ogden') != -1:
            data_point += 200
        class_label = get_class(data_point_name)
        input_data.append(InputData(data_point, df, class_label))
    return input_data

def get_test_data(input_data: list[InputData]) -> list[InputData]:
    test_data: list[InputData] = []
    for data in input_data:
        if data.data_point % 6 == 0:
            test_data.append(data)
    return test_data

def get_validation_data(input_data: list[InputData]) -> list[InputData]:
    validation_data: list[InputData] = []
    for data in input_data:
        if data.data_point % 6 == 1:
            validation_data.append(data)
    return validation_data

def get_extreme_values(input_datas: list[InputData]) -> tuple[float]:
    max_strain = max([input_data.data['strain'].max() for input_data in input_datas])
    min_strain = min([input_data.data['strain'].min() for input_data in input_datas])
    max_stress = max([input_data.data['stress'].max() for input_data in input_datas])
    min_stress = min([input_data.data['stress'].min() for input_data in input_datas])
    return max_strain, min_strain, max_stress, min_stress

def normalize_input_data(input_datas: list[InputData], max_strain: float, min_strain: float, max_stress: float, min_stress: float):
    for input_data in input_datas:
        input_data.data['strain'] = (input_data.data['strain'] - min_strain) / (max_strain - min_strain)
        input_data.data['stress'] = (input_data.data['stress'] - min_stress) / (max_stress - min_stress)

def preprocess_data_biax():
    inputs_biax_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_biax_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_biax_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.BIAX_TENSION_FOLDER, data_prep.OGDEN_FOLDER)

    input_data_biax_nh: list[InputData] = get_input_data(inputs_biax_nh)
    input_data_biax_mr: list[InputData] = get_input_data(inputs_biax_mr)
    input_data_biax_og: list[InputData] = get_input_data(inputs_biax_og)

    input_data_biax: list[InputData] = input_data_biax_nh + input_data_biax_mr + input_data_biax_og

    # Modify inputs to have 100 datapoints
    # IF more: interpolate to 100
    # IF less: pad with zeros
    make_input_lengths_equal(input_data_biax)

    input_data_biax_test = get_test_data(input_data_biax)
    input_data_biax_val = get_validation_data(input_data_biax)
    input_data_biax_train = [data for data in input_data_biax if data not in input_data_biax_test and data not in input_data_biax_val]

    # Normalize inputData.data of input_data_biax_train
    max_strain, min_strain, max_stress, min_stress = get_extreme_values(input_data_biax_train)
    normalize_input_data(input_data_biax_train, max_strain, min_strain, max_stress, min_stress)
    normalize_input_data(input_data_biax_test, max_strain, min_strain, max_stress, min_stress)
    normalize_input_data(input_data_biax_val, max_strain, min_strain, max_stress, min_stress)

    # create inputs and labels for CNN
    # input data: 4D tensor (samples, rows, cols, channels)
    # labels: 1D tensor
    train_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_biax_train])
    train_labels = np.array([input_data.class_label for input_data in input_data_biax_train]).reshape(-1, 1)
    test_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_biax_test])
    test_labels = np.array([input_data.class_label for input_data in input_data_biax_test]).reshape(-1, 1)
    val_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_biax_val])
    val_labels = np.array([input_data.class_label for input_data in input_data_biax_val]).reshape(-1, 1)

    result = {
        TRAIN_INPUTS: train_inputs,
        TRAIN_LABELS: train_labels,
        TEST_INPUTS: test_inputs,
        TEST_LABELS: test_labels,
        VAL_INPUTS: val_inputs,
        VAL_LABELS: val_labels
    }

    return result

def preprocess_data_planar_comp():
    inputs_planar_comp_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_COMPRESSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_planar_comp_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_COMPRESSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_planar_comp_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_COMPRESSION_FOLDER, data_prep.OGDEN_FOLDER)

    input_data_planar_comp_nh: list[InputData] = get_input_data(inputs_planar_comp_nh)
    input_data_planar_comp_mr: list[InputData] = get_input_data(inputs_planar_comp_mr)
    input_data_planar_comp_og: list[InputData] = get_input_data(inputs_planar_comp_og)

    input_data_planar_comp: list[InputData] = input_data_planar_comp_nh + input_data_planar_comp_mr + input_data_planar_comp_og

    # Modify inputs to have 100 datapoints
    # IF more: interpolate to 100
    # IF less: pad with zeros
    make_input_lengths_equal(input_data_planar_comp)

    input_data_planar_comp_test = get_test_data(input_data_planar_comp)
    input_data_planar_comp_val = get_validation_data(input_data_planar_comp)
    input_data_planar_comp_train = [data for data in input_data_planar_comp if data not in input_data_planar_comp_test and data not in input_data_planar_comp_val]

    # Normalize inputData.data of input_data_planar_comp_train
    max_strain, min_strain, max_stress, min_stress = get_extreme_values(input_data_planar_comp_train)
    normalize_input_data(input_data_planar_comp_train, max_strain, min_strain, max_stress, min_stress)
    normalize_input_data(input_data_planar_comp_test, max_strain, min_strain, max_stress, min_stress)
    normalize_input_data(input_data_planar_comp_val, max_strain, min_strain, max_stress, min_stress)

    # create inputs and labels for CNN
    train_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_planar_comp_train])
    train_labels = np.array([input_data.class_label for input_data in input_data_planar_comp_train]).reshape(-1, 1)
    test_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_planar_comp_test])
    test_labels = np.array([input_data.class_label for input_data in input_data_planar_comp_test]).reshape(-1, 1)
    val_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_planar_comp_val])
    val_labels = np.array([input_data.class_label for input_data in input_data_planar_comp_val]).reshape(-1, 1)

    result = {
        TRAIN_INPUTS: train_inputs,
        TRAIN_LABELS: train_labels,
        TEST_INPUTS: test_inputs,
        TEST_LABELS: test_labels,
        VAL_INPUTS: val_inputs,
        VAL_LABELS: val_labels
    }
    return result

def preprocess_data_planar_tension():
    inputs_planar_tension_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_planar_tension_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_planar_tension_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.PLANAR_TENSION_FOLDER, data_prep.OGDEN_FOLDER)

    input_data_planar_tension_nh: list[InputData] = get_input_data(inputs_planar_tension_nh)
    input_data_planar_tension_mr: list[InputData] = get_input_data(inputs_planar_tension_mr)
    input_data_planar_tension_og: list[InputData] = get_input_data(inputs_planar_tension_og)

    input_data_planar_tension: list[InputData] = input_data_planar_tension_nh + input_data_planar_tension_mr + input_data_planar_tension_og

    # Modify inputs to have 100 datapoints
    # IF more: interpolate to 100
    # IF less: pad with zeros
    make_input_lengths_equal(input_data_planar_tension)

    input_data_planar_tension_test = get_test_data(input_data_planar_tension)
    input_data_planar_tension_val = get_validation_data(input_data_planar_tension)
    input_data_planar_tension_train = [data for data in input_data_planar_tension if data not in input_data_planar_tension_test and data not in input_data_planar_tension_val]

    # Normalize inputData.data of input_data_planar_tension_train
    max_strain, min_strain, max_stress, min_stress = get_extreme_values(input_data_planar_tension_train)
    normalize_input_data(input_data_planar_tension_train, max_strain, min_strain, max_stress, min_stress)
    normalize_input_data(input_data_planar_tension_test, max_strain, min_strain, max_stress, min_stress)
    normalize_input_data(input_data_planar_tension_val, max_strain, min_strain, max_stress, min_stress)

    # create inputs and labels for CNN
    train_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_planar_tension_train])
    train_labels = np.array([input_data.class_label for input_data in input_data_planar_tension_train]).reshape(-1, 1)
    test_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_planar_tension_test])
    test_labels = np.array([input_data.class_label for input_data in input_data_planar_tension_test]).reshape(-1, 1)
    val_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_planar_tension_val])
    val_labels = np.array([input_data.class_label for input_data in input_data_planar_tension_val]).reshape(-1, 1)

    result = {
        TRAIN_INPUTS: train_inputs,
        TRAIN_LABELS: train_labels,
        TEST_INPUTS: test_inputs,
        TEST_LABELS: test_labels,
        VAL_INPUTS: val_inputs,
        VAL_LABELS: val_labels
    }
    return result

def preprocess_data_simple_shear():
    inputs_simple_shear_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.SIMPLE_SHEAR_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_simple_shear_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.SIMPLE_SHEAR_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_simple_shear_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.SIMPLE_SHEAR_FOLDER, data_prep.OGDEN_FOLDER)

    input_data_simple_shear_nh: list[InputData] = get_input_data(inputs_simple_shear_nh)
    input_data_simple_shear_mr: list[InputData] = get_input_data(inputs_simple_shear_mr)
    input_data_simple_shear_og: list[InputData] = get_input_data(inputs_simple_shear_og)

    input_data_simple_shear: list[InputData] = input_data_simple_shear_nh + input_data_simple_shear_mr + input_data_simple_shear_og

    # Modify inputs to have 100 datapoints
    # IF more: interpolate to 100
    # IF less: pad with zeros
    make_input_lengths_equal(input_data_simple_shear)

    input_data_simple_shear_test = get_test_data(input_data_simple_shear)
    input_data_simple_shear_val = get_validation_data(input_data_simple_shear)
    input_data_simple_shear_train = [data for data in input_data_simple_shear if data not in input_data_simple_shear_test and data not in input_data_simple_shear_val]

    # Normalize inputData.data of input_data_simple_shear_train
    max_strain, min_strain, max_stress, min_stress = get_extreme_values(input_data_simple_shear_train)
    normalize_input_data(input_data_simple_shear_train, max_strain, min_strain, max_stress, min_stress)
    normalize_input_data(input_data_simple_shear_test, max_strain, min_strain, max_stress, min_stress)
    normalize_input_data(input_data_simple_shear_val, max_strain, min_strain, max_stress, min_stress)

    # create inputs and labels for CNN
    train_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_simple_shear_train])
    train_labels = np.array([input_data.class_label for input_data in input_data_simple_shear_train]).reshape(-1, 1)
    test_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_simple_shear_test])
    test_labels = np.array([input_data.class_label for input_data in input_data_simple_shear_test]).reshape(-1, 1)
    val_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_simple_shear_val])
    val_labels = np.array([input_data.class_label for input_data in input_data_simple_shear_val]).reshape(-1, 1)

    result = {
        TRAIN_INPUTS: train_inputs,
        TRAIN_LABELS: train_labels,
        TEST_INPUTS: test_inputs,
        TEST_LABELS: test_labels,
        VAL_INPUTS: val_inputs,
        VAL_LABELS: val_labels
    }
    return result

def preprocess_data_uniax_comp():
    inputs_uniax_comp_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_COMPRESSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_uniax_comp_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_COMPRESSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_uniax_comp_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_COMPRESSION_FOLDER, data_prep.OGDEN_FOLDER)

    input_data_uniax_comp_nh: list[InputData] = get_input_data(inputs_uniax_comp_nh)
    input_data_uniax_comp_mr: list[InputData] = get_input_data(inputs_uniax_comp_mr)
    input_data_uniax_comp_og: list[InputData] = get_input_data(inputs_uniax_comp_og)

    input_data_uniax_comp: list[InputData] = input_data_uniax_comp_nh + input_data_uniax_comp_mr + input_data_uniax_comp_og

    # Modify inputs to have 100 datapoints
    # IF more: interpolate to 100
    # IF less: pad with zeros
    make_input_lengths_equal(input_data_uniax_comp)

    input_data_uniax_comp_test = get_test_data(input_data_uniax_comp)
    input_data_uniax_comp_val = get_validation_data(input_data_uniax_comp)
    input_data_uniax_comp_train = [data for data in input_data_uniax_comp if data not in input_data_uniax_comp_test and data not in input_data_uniax_comp_val]

    # Normalize inputData.data of input_data_uniax_comp_train
    max_strain, min_strain, max_stress, min_stress = get_extreme_values(input_data_uniax_comp_train)
    normalize_input_data(input_data_uniax_comp_train, max_strain, min_strain, max_stress, min_stress)
    normalize_input_data(input_data_uniax_comp_test, max_strain, min_strain, max_stress, min_stress)
    normalize_input_data(input_data_uniax_comp_val, max_strain, min_strain, max_stress, min_stress)

    # create inputs and labels for CNN
    train_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_uniax_comp_train])
    train_labels = np.array([input_data.class_label for input_data in input_data_uniax_comp_train]).reshape(-1, 1)
    test_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_uniax_comp_test])
    test_labels = np.array([input_data.class_label for input_data in input_data_uniax_comp_test]).reshape(-1, 1)
    val_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_uniax_comp_val])
    val_labels = np.array([input_data.class_label for input_data in input_data_uniax_comp_val]).reshape(-1, 1)

    result = {
        TRAIN_INPUTS: train_inputs,
        TRAIN_LABELS: train_labels,
        TEST_INPUTS: test_inputs,
        TEST_LABELS: test_labels,
        VAL_INPUTS: val_inputs,
        VAL_LABELS: val_labels
    }
    return result

def preprocess_data_uniax_tension():
    inputs_uniax_tension_nh: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.NEO_HOOKEAN_FOLDER)
    inputs_uniax_tension_mr: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.MOONEY_RIVLIN_FOLDER)
    inputs_uniax_tension_og: dict[str, pd.DataFrame] = data_prep.read_inputs(data_prep.UNIAX_TENSION_FOLDER, data_prep.OGDEN_FOLDER)

    input_data_uniax_tension_nh: list[InputData] = get_input_data(inputs_uniax_tension_nh)
    input_data_uniax_tension_mr: list[InputData] = get_input_data(inputs_uniax_tension_mr)
    input_data_uniax_tension_og: list[InputData] = get_input_data(inputs_uniax_tension_og)

    input_data_uniax_tension: list[InputData] = input_data_uniax_tension_nh + input_data_uniax_tension_mr + input_data_uniax_tension_og

    # Modify inputs to have 100 datapoints
    # IF more: interpolate to 100
    # IF less: pad with zeros
    make_input_lengths_equal(input_data_uniax_tension)

    input_data_uniax_tension_test = get_test_data(input_data_uniax_tension)
    input_data_uniax_tension_val = get_validation_data(input_data_uniax_tension)
    input_data_uniax_tension_train = [data for data in input_data_uniax_tension if data not in input_data_uniax_tension_test and data not in input_data_uniax_tension_val]

    # Normalize inputData.data of input_data_uniax_tension_train
    max_strain, min_strain, max_stress, min_stress = get_extreme_values(input_data_uniax_tension_train)
    normalize_input_data(input_data_uniax_tension_train, max_strain, min_strain, max_stress, min_stress)
    normalize_input_data(input_data_uniax_tension_test, max_strain, min_strain, max_stress, min_stress)
    normalize_input_data(input_data_uniax_tension_val, max_strain, min_strain, max_stress, min_stress)

    # create inputs and labels for CNN
    train_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_uniax_tension_train])
    train_labels = np.array([input_data.class_label for input_data in input_data_uniax_tension_train]).reshape(-1, 1)
    test_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_uniax_tension_test])
    test_labels = np.array([input_data.class_label for input_data in input_data_uniax_tension_test]).reshape(-1, 1)
    val_inputs = np.array([input_data.data.to_numpy().reshape(100, 2, 1) for input_data in input_data_uniax_tension_val])
    val_labels = np.array([input_data.class_label for input_data in input_data_uniax_tension_val]).reshape(-1, 1)

    result = {
        TRAIN_INPUTS: train_inputs,
        TRAIN_LABELS: train_labels,
        TEST_INPUTS: test_inputs,
        TEST_LABELS: test_labels,
        VAL_INPUTS: val_inputs,
        VAL_LABELS: val_labels
    }
    return result

def build_cnn(filters:int=2, kernel_length:int=5):
    # Build 2D CNN
    model = Sequential([
        Conv2D(filters=filters, kernel_size=(kernel_length, 2), activation='relu', input_shape=(100, 2, 1)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def train_cnn(model, inputs_and_labest: dict[str, np.ndarray], epochs:int=10):
    history = model.fit(
        inputs_and_labest[TRAIN_INPUTS],
        inputs_and_labest[TRAIN_LABELS],
        epochs=epochs,
        validation_data=(inputs_and_labest[VAL_INPUTS], inputs_and_labest[VAL_LABELS])
    )
    return history

def evaluate_cnn_results(history, model, inputs_and_labest: dict[str, np.ndarray], stress_type):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and validation accuracy for ' + stress_type)
    plt.ylim([0.4, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_accuracy = model.evaluate(inputs_and_labest[TEST_INPUTS], inputs_and_labest[TEST_LABELS])
    print(f'Test accuracy: {test_accuracy}')

    # plot confusion matrix
    predictions = model.predict(inputs_and_labest[TEST_INPUTS])
    predictions = np.argmax(predictions, axis=1)
    # use heatmap to plot confusion matrix
    cm = confusion_matrix(inputs_and_labest[TEST_LABELS], predictions)
    # use heatmap to plot confusion matrix
    heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {stress_type}')
    plt.show()

def train_model_with_params(filters:int, kernel_length:int, epochs:int, inputs_and_labest: dict[str, np.ndarray]):
    model = build_cnn(filters, kernel_length)
    history = train_cnn(model, inputs_and_labest, epochs)
    return history, model

def train_model_with_param_ranges(filter_range:tuple[int], kernel_length_range:tuple[int], epochs:int, inputs_and_labest: dict[str, np.ndarray]):
    best_accuracy = 0
    best_params = (0, 0)
    best_model_and_history = (None, None)
    results: dict[tuple[int, int], float] = {}
    for filters in filter_range:
        for kernel_length in kernel_length_range:
            history, model = train_model_with_params(filters, kernel_length, epochs, inputs_and_labest)
            test_loss, test_accuracy = model.evaluate(inputs_and_labest[TEST_INPUTS], inputs_and_labest[TEST_LABELS])
            if EVALUATE_ENABLED:
                evaluate_cnn_results(history, model, inputs_and_labest, f'biax tension {filters} filters, {kernel_length} kernel length')
            # append results to dataframe
            results[(filters, kernel_length)] = test_accuracy
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_params = (filters, kernel_length)
                best_model_and_history = (model, history)
    
    # heatmap for results
    heatmap_data = pd.DataFrame(columns=kernel_length_range)
    for filters in filter_range:
        row = []
        for kernel_length in kernel_length_range:
            row.append(results[(filters, kernel_length)])
        heatmap_data.loc[filters] = row
    heatmap(heatmap_data, annot=True)
    plt.xlabel('Kernel length')
    plt.ylabel('Number of filters')
    plt.title('Accuracy for different hyperparameters')
    plt.show()
    print(f'Best accuracy: {best_accuracy} with hyperparameters: {best_params}')
    
    return best_model_and_history
    

if __name__ == '__main__':
    biax_data: dict[str, np.ndarray] = preprocess_data_biax()
    best_biax_cnn_model, history_biax = train_model_with_param_ranges(range(3, 12, 4), range(3, 12, 4), 200, biax_data)
    evaluate_cnn_results(history_biax, best_biax_cnn_model, biax_data, 'biax tension')
    
    planar_comp_data: dict[str, np.ndarray] = preprocess_data_planar_comp()
    best_planar_comp_cnn_model, history_planar_comp = train_model_with_param_ranges(range(3, 12, 4), range(3, 12, 4), 250, planar_comp_data)
    evaluate_cnn_results(history_planar_comp, best_planar_comp_cnn_model, planar_comp_data, 'planar compression')

    planar_tension_data: dict[str, np.ndarray] = preprocess_data_planar_tension()
    best_planar_tension_cnn_model, history_planar_tension = train_model_with_param_ranges(range(3, 12, 4), range(3, 12, 4), 250, planar_tension_data)
    evaluate_cnn_results(history_planar_tension, best_planar_tension_cnn_model, planar_tension_data, 'planar tension')

    simple_shear_data: dict[str, np.ndarray] = preprocess_data_simple_shear()
    best_simple_shear_cnn_model, history_simple_shear = train_model_with_param_ranges(range(3, 12, 4), range(3, 12, 4), 250, simple_shear_data)
    evaluate_cnn_results(history_simple_shear, best_simple_shear_cnn_model, simple_shear_data, 'simple shear')

    uniax_comp_data: dict[str, np.ndarray] = preprocess_data_uniax_comp()
    best_uniax_comp_cnn_model, history_uniax_comp = train_model_with_param_ranges(range(3, 12, 4), range(3, 12, 4), 400, uniax_comp_data)
    evaluate_cnn_results(history_uniax_comp, best_uniax_comp_cnn_model, uniax_comp_data, 'uniax compression')

    uniax_tension_data: dict[str, np.ndarray] = preprocess_data_uniax_tension()
    best_uniax_tension_cnn_model, history_uniax_tension = train_model_with_param_ranges(range(3, 12, 4), range(3, 12, 4), 250, uniax_tension_data)
    evaluate_cnn_results(history_uniax_tension, best_uniax_tension_cnn_model, uniax_tension_data, 'uniax tension')

    # TODO: 
    #  vote with best models
    #  early stopping ...
    #  alter CNN architecture --> multiple conv layers, pooling layers

    print('done')

