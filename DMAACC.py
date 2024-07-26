from mutation_generator import MutationGenerator
from unit_clusterer import UnitClustering
from mutation_score import MutationScore
from keras.models import load_model
from network import Dataset
import time


class DMAACC:
    def __init__(self):
        self._model_filename = None
        self._model = None
        self._mutation_level = 'neuron'
        self._mutator_list = ['CW', 'NAI', 'NEB']
        self._standard_deviation = 0.2
        self._cluster_size = 5
        self._one_unit_per_cluster = False
        self._threshold = 2
        self.__need_mutation_analysis = True
        self._mutation_percent = 0.1
        self._dataset = None

    def load_model(self, model_filename):
        self._model_filename = model_filename
        self._model = load_model(model_filename)

    def get_model(self):
        return self._model

    def set_mutation_level(self, mutation_level):
        self._mutation_level = mutation_level

    def set_mutation_percent(self, mutation_percent):
        self._mutation_percent = mutation_percent

    def set_mutator_list(self, mutator_list):
        self._mutator_list = mutator_list

    def set_standard_deviation(self, std):
        self._standard_deviation = std

    def set_cluster_size(self, cluster_size):
        self._cluster_size = cluster_size

    def set_one_unit_per_cluster(self, u):
        self._one_unit_per_cluster = u

    def set_cluster_selection_threshold(self, threshold):
        self._threshold = threshold

    def set_dataset(self, dataset):
        self._dataset = dataset

    def __get_outputs_count(self):
        return self._model.layers[-1].units

    def run_approach_1(self):
        # counter = MutationCounter()
        if self.__need_mutation_analysis:
            clusters = []
            start = time.time()
            if self._mutation_level == 'cluster':
                unit_clustering = UnitClustering(self._model)

                #SO WHEN I PASS CLUSTER SIZE HERE, I NEED TO DO THE MATH BEFORE IT
                clusters = unit_clustering.get_clusters(self._cluster_size,
                                                            one_unit_per_cluster=self._one_unit_per_cluster)
                for index, mu in enumerate(clusters):
                   print('layer '+str(mu.get_layer_index())+' cluster '+str(index)+' '+str(mu.get_unit_indices()))
            mg = MutationGenerator(self._model_filename, self._model, clusters, self._mutation_level)
            mg.set_mutation_percent(self._mutation_percent)
            mutations = mg.get_mutations(self._mutator_list)  # this is M'
            # self.model is the original model
            end = time.time()
            self._dataset = Dataset(self._model_filename.split('-')[1].split('.')[0])  # this is T
            ms = MutationScore(self._model_filename.split('.')[0], self._model, mutations, self._dataset,
                               self._dataset.get_dataset_name(), self._mutation_level)
            ms.run()
            print('Mutation Score: ' + str(ms.get_mutation_score()))
            print('Number of mutations: ' + str(len(mutations)))
            print('Average size of clusters per layer: ' + str(self._cluster_size))
            #accs = []
            # for mut in mutations:
            #     accs += [mut.evaluate(x=self._dataset.get_x_test(), y=self._dataset.get_y_test(), return_dict=True)['accuracy']]

            return start, end, ms.get_mutation_score() #, accs

    def run_approach_2(self):
        mg = MutationGenerator(self._model_filename, self._model, [], 'neuron')
        mg.set_mutation_percent(self._mutation_percent)
        mutations = mg.get_mutations(self._mutator_list)
        ms = MutationScore(self._model_filename.split('.')[0], self._model, mutations, self._dataset,
                           self._dataset.get_dataset_name(), self._mutation_level)

        if self._mutation_level == 'cluster':
            pass


