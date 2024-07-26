from network import Dataset
import numpy as np


class MutationScore:
    def __init__(self, model_name, model, mutations, dataset, dataset_name, mutation_level):
        self._model_name = model_name
        self._model = model
        self._mutations = mutations
        self._dataset = dataset
        self._dataset_name = dataset_name
        self._mutation_level = mutation_level
        self._num_of_classes = self._model.layers[-1].units
        self._correct_test_points = None
        self._mutation_score = None

    def get_mutation_score(self):
        return self._mutation_score

    def get_correct_test_points(self):  # , dataset):
        x = self._dataset.get_x_test()
        y = self._dataset.get_y_test()

        x_new = []
        y_new = []

        predictions = self._model.predict(x)

        for i, row in enumerate(predictions):
            if np.argmax(y[i]) == np.argmax(row):
                x_new += [x[i]]
                y_new += [y[i]]

        return x_new, y_new

    def killed_classes(self):
        classes_list = [1] * self._num_of_classes
        sum = 0
        for m_prime in self._mutations:
            predictions = m_prime.get_model().predict(np.array(self._correct_test_points[0]))
            for i, predicted_label in enumerate(predictions):
                if np.argmax(self._correct_test_points[1][i]) != np.argmax(predicted_label):
                    classes_list[np.argmax(predicted_label)] *= 0
                    print("Class killed - " + str(np.argmax(predicted_label)))
            sum += self._num_of_classes - np.sum(classes_list)
            classes_list = [1] * self._num_of_classes
        return sum

    def run(self):
        x, y = self.get_correct_test_points()
        self._correct_test_points = [x, y]
        kc = self.killed_classes()
        print("Killed classes: " + str(kc))
        self._mutation_score = kc / (len(self._mutations) * self._num_of_classes)
