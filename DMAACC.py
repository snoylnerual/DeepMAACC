from os import path
import os
import shutil
import numpy as np
from keras.models import load_model
import pickle
#from mutation_consumer import MutationCounter
from mutation_generator import MutationGenerator
#from slicer import OptimizedRankedSlicer
from unit_clusterer import UnitClustering
from util import binarize_and_balance, jaccard_sim


class DMAACC:
    def __init__(self):
        self._model_filename = None
        self._model = None
        # self._train_in = None
        # self._train_out = None
        # self._test_in = None
        # self._test_out = None
        # self._fraction = 0.1
        self._mutation_level = 'neuron'
        self._mutation_ratio = 0.15
        self._mutator_list = ['GF', 'NAI', 'NEB']
        self._standard_deviation = 0.2
        self._clusters_per_layer = 5
        self._one_unit_per_cluster = False
        self._threshold = 2
        # self._slicer = OptimizedRankedSlicer
        # self._slicer_name = 'optimized'
        # self._optimizer = 'adam'
        # self._epochs = 64
        self.__need_mutation_analysis = True

    def load_model(self, model_filename):
        self._model_filename = model_filename
        self._model = load_model(model_filename)

    # def load_train_inputs(self, train_inputs_filename):
    #     self._train_in = np.load(train_inputs_filename)
    #
    # def load_train_outputs(self, train_outputs_filename):
    #     self._train_out = np.load(train_outputs_filename)
    #
    # def load_test_inputs(self, test_inputs_filename):
    #     self._test_in = np.load(test_inputs_filename)
    #
    # def load_test_outputs(self, test_outputs_filename):
    #     self._test_out = np.load(test_outputs_filename)
    #
    # def set_mutation_weight_fraction(self, fraction):
    #     self._fraction = fraction

    def set_mutation_level(self, mutation_level):
        self._mutation_level = mutation_level

    def set_mutation_ratio(self, mutation_ratio):
        self._mutation_ratio = mutation_ratio

    def set_standard_deviation(self, std):
        self._standard_deviation = std

    def set_clusters_per_layer(self, cluster_size):
        self._clusters_per_layer = cluster_size

    def set_one_unit_per_cluster(self, u):
        self._one_unit_per_cluster = u

    def set_cluster_selection_threshold(self, threshold):
        self._threshold = threshold

    # def set_slicer(self, slicer_name):
    #     sn = slicer_name.strip().lower()
    #     if sn == 'optimized':
    #         self._slicer = OptimizedRankedSlicer
    #         self.__need_mutation_analysis = True
    #     else:
    #         raise ValueError('Unknown slicer: ' + slicer_name)
    #     self._slicer_name = sn
    #
    # def set_optimizer(self, optimizer):
    #     self._optimizer = optimizer
    #
    # def set_epochs(self, epochs):
    #     self._epochs = epochs

    def __get_outputs_count(self):
        return self._model.layers[-1].units

    def run(self):
        #counter = MutationCounter()
        #expanded_contrib_matrix = None
        if self.__need_mutation_analysis:
            # cm_filename = 'matrix-%s-%d-%.2f.cm' % ('all' if self._one_unit_per_cluster
            #                                         else str(self._clusters_per_layer),
            #                                         self._threshold,
            #                                         self._fraction)
            # if path.exists(cm_filename):
            #     print('Skipped mutation analysis, loading %s...' % cm_filename)
            #     with open(cm_filename, 'rb') as f:
            #         expanded_contrib_matrix = pickle.load(f)
            # else:
            unit_clustering = UnitClustering(self._model)
            clusters = unit_clustering.get_clusters(self._clusters_per_layer,
                                                        one_unit_per_cluster=self._one_unit_per_cluster)
            # ncm = NeuronContributionMatrix(self._model, self._train_in)
            for index, mu in enumerate(clusters):
                print(index)
                print(str(mu.get_layer_index())+str(mu.get_unit_indices()))
            #      mu.add(.1)
            mg = MutationGenerator(self._model_filename, self._model, clusters, self._mutation_level, self._mutation_ratio)
            mg.get_mutations(self._mutator_list)
            """ #   expanded_contrib_matrix = ncm.get_expanded_contrib_matrix()
                with open(cm_filename, 'wb') as f:
                    pickle.dump(expanded_contrib_matrix, f)

        counter.reset()
        out_dir = 'modules-%s-%s-%d-%.2f' % (self._slicer_name,
                                             'all' if self._one_unit_per_cluster else str(self._clusters_per_layer),
                                             self._threshold,
                                             self._fraction)
        if path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        original_predictions = self._model.predict(self._test_in, verbose=None)
        selected_units = dict()
        for target in range(self.__get_outputs_count()):
            bin_train_in, bin_train_out = binarize_and_balance(self._train_in, self._train_out, target)
            bin_test_in, bin_test_out = binarize_and_balance(self._test_in, self._test_out, target)
            if expanded_contrib_matrix is None:
                slicer = self._slicer(self._model_filename,
                                      self._clusters_per_layer,
                                      self._threshold,
                                      train_data=(bin_train_in, bin_train_out),
                                      validation_data=(bin_test_in, bin_test_out),
                                      optimizer=self._optimizer,
                                      epochs=self._epochs,
                                      one_unit_per_cluster=self._one_unit_per_cluster)  # noqa
            else:
                slicer = self._slicer(self._model_filename,
                                      expanded_contrib_matrix,
                                      self._threshold,
                                      train_data=(bin_train_in, bin_train_out),
                                      validation_data=(bin_test_in, bin_test_out),
                                      optimizer=self._optimizer,
                                      epochs=self._epochs)
            sliced_model = slicer.get_slice(target)
            selected_units[target] = slicer.get_selected_units()
            module_name = 'module-%d' % target
            sliced_model.save(path.join(out_dir, module_name + '.h5'))
            with open(path.join(out_dir, module_name + '-perf.txt'), 'w') as f:
                class_indices = np.where(np.argmax(self._test_out, axis=1) == target)[0]
                original_class_predictions = np.argmax(original_predictions[class_indices], axis=1)
                class_true_labels = np.argmax(self._test_out[class_indices])
                original_model_acc = np.sum(original_class_predictions == class_true_labels) / len(class_indices)
                module_acc = sliced_model.evaluate(bin_test_in, bin_test_out, verbose=None)[1]
                f.write('Module Accuracy: %f\n' % module_acc)
                f.write('Original Model Accuracy: %f\n' % original_model_acc)
        with open(path.join(out_dir, 'stats.txt'), 'w') as f:
            f.write(str(counter))
            f.write('==============\n')
            f.write('Jaccard similarities:')
            count = 0
            ji_sum = 0
            for m1 in range(self.__get_outputs_count()):
                for m2 in range(m1 + 1, self.__get_outputs_count()):
                    ji = jaccard_sim(selected_units[m1], selected_units[m2])
                    f.write('Module %d vs. Module %d: %f\n' % (m1, m2, ji))
                    ji_sum += ji
                    count += 1
            f.write('--------------\n')
            f.write('Average: %f\n' % (ji_sum / count))"""

