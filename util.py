import numpy as np
from keras import Sequential, Model
from keras.utils import to_categorical


def binarize_and_balance(inputs, outputs, target):
    dataset_dict = dict()
    rows_count = inputs.shape[0]
    for row_index in range(0, rows_count):
        label = np.argmax(outputs[row_index])
        if label not in dataset_dict:
            dataset_dict[label] = []
        dataset_dict[label].append(inputs[row_index])
    balanced_inputs = dataset_dict[target]
    target_points_sz = len(balanced_inputs)
    balanced_bin_outputs = [1 for _ in range(target_points_sz)]
    for i in range(target_points_sz):
        for label, points in dataset_dict.items():
            if label == target or i >= len(points):
                continue
            balanced_inputs.append(points[i])
            balanced_bin_outputs.append(0)
    return np.asarray(balanced_inputs), to_categorical(np.asarray(balanced_bin_outputs), 2)


def is_softmax_classifier(model):
    output_layer = model.layers[-1]
    return hasattr(output_layer, 'activation') and output_layer.activation.__name__ == 'softmax'


def model_ok(model):
    return isinstance(model, Sequential) or isinstance(model, Model)


def jaccard_sim(set_a, set_b):
    intersection = len(set(set_a).intersection(set_b))
    union = (len(set_a) + len(set_b)) - intersection
    return float(intersection) / union
