from mutation_generator import MutationGenerator
from unit_clusterer import UnitClustering
from mutation_score import MutationScore
from keras.models import load_model
from network import Dataset
import numpy as np
import pandas as pd
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
        self._mutation_percent = 0.1
        self._dataset = None
        self._thresholds = [n/2 for n in range(6, 15)]

    def load_model(self, model_filename):
        self._model_filename = model_filename
        self._model = load_model(model_filename)

    def get_model(self):
        return self._model

    def set_mutation_level(self, mutation_level):
        self._mutation_level = mutation_level

    def get_mutation_level(self, mutation_level):
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

    def set_ParHAC_thresholds(self, threshold):
        self._thresholds = threshold

    def __get_outputs_count(self):
        return self._model.layers[-1].units

    def run_vanilla(self):
        df_vanilla = pd.DataFrame(
            columns=['Model_Type', 'Dataset', 'Mutation_Level', 'Mutate_time',
                     'Number_of_Mutants', 'Mutation_Score', 'MS_time', 'Total_time'])
        clusters = []
        start = time.time()
        m_start = time.time()
        mg = MutationGenerator(self._model_filename, self._model, clusters, self._mutation_level)
        mg.set_mutation_percent(self._mutation_percent)
        mutations = mg.get_mutations(self._mutator_list)  # this is M'
        m_end = time.time()
        self._dataset = Dataset(self._model_filename.split('-')[1].split('.')[0])  # this is T
        ms_start = time.time()
        ms = MutationScore(self._model_filename.split('.')[0], self._model, mutations, self._dataset,
                           self._dataset.get_dataset_name(), self._mutation_level)
        ms.run()
        ms_end = time.time()
        df_vanilla.loc[len(df_vanilla.index)] = [self._model_filename, self._dataset.get_dataset_name(),
                             self._mutation_level, m_end - m_start, len(mutations),
                             ms.get_mutation_score(), ms_end - ms_start, ms_end - start]

        return df_vanilla

    def run_approach_1(self):
        df_clusters = pd.DataFrame(
            columns=['Model_Type', 'Dataset', 'Mutation_Level', 'Mutate_time',
                     'Number_of_Mutants', 'Number_of_Clusters', 'Clusters_per_layer',
                     'Max_Cluster_Sz', 'Min_Cluster_Sz', 'Mean_Cluster_Sz',
                     'Cluster_time', 'Mutation_Score', 'MS_time', 'Total_time'])
        clusters = []
        start = time.time()
        c_start = 0
        c_end = 0
        unit_clustering = UnitClustering(self._model)
        if self._mutation_level == 'cluster':
            c_start = time.time()
            clusters = unit_clustering.get_clusters(self._cluster_size)
            c_end = time.time()
            for index, mu in enumerate(clusters):
               print('layer '+str(mu.get_layer_index())+' cluster '+str(index)+' '+str(mu.get_unit_indices()))
        m_start = time.time()
        mg = MutationGenerator(self._model_filename, self._model, clusters, self._mutation_level)
        mg.set_mutation_percent(self._mutation_percent)
        mutations = mg.get_mutations(self._mutator_list)  # this is M'
        # self.model is the original model
        m_end = time.time()
        self._dataset = Dataset(self._model_filename.split('-')[1].split('.')[0])  # this is T
        ms_start = time.time()
        ms = MutationScore(self._model_filename.split('.')[0], self._model, mutations, self._dataset,
                           self._dataset.get_dataset_name(), self._mutation_level)
        ms.run()
        ms_end = time.time()
        print('Mutation Score: ' + str(ms.get_mutation_score()))
        print('Number of mutations: ' + str(len(mutations)))
        print('Average size of clusters per layer: ' + str(self._cluster_size))
        df_clusters.loc[len(df_clusters.index)] = [self._model_filename, self._dataset.get_dataset_name(),
                           self._mutation_level, m_end - m_start, len(mutations), ms.get_cluster_amount(),
                           self._cluster_size, unit_clustering.get_max_cluster_size(),
                           unit_clustering.get_min_cluster_size(), unit_clustering.get_mean_cluster_size(),
                           c_end - c_start, ms.get_mutation_score(), ms_end - ms_start, ms_end - start]

        return df_clusters

    def run_approach_2(self):
        start = time.time()
        mg = MutationGenerator(self._model_filename, self._model, [], 'neuron')
        mg.set_mutation_percent(self._mutation_percent)
        mutations = mg.get_mutations(self._mutator_list)
        m_end = time.time()
        self._dataset = Dataset(self._model_filename.split('-')[1].split('.')[0])

        ms = MutationScore(self._model_filename.split('.')[0], self._model, mutations, self._dataset,
                           self._dataset.get_dataset_name(), self._mutation_level)

        if 'cluster' in self._mutation_level:
            df_cluster = pd.DataFrame(columns=['Model_Type', 'Dataset', 'Mutable_Layers', 'Mutation_Level',
                                               'Mutate_time', 'Number_of_Mutants', 'ParHAC_Threshold',
                                               'Number_of_Clusters', 'Max_Cluster_Sz', 'Min_Cluster_Sz',
                                               'Mean_Cluster_Sz', 'Cluster_MS_Score', 'Cluster_time',
                                               'Mutation_Score', 'MS_time', 'Total_time'])

            unit_clustering = UnitClustering(self._model)
            for threshold in self._thresholds:
                ms_start = time.time()
                graph_clusters = unit_clustering.get_graph_clusters(mutations, threshold)
                ms.set_clusters(graph_clusters)
                ms.cluster_run()
                ms_end = time.time()
                df_cluster.loc[len(df_cluster.index)] = [self._model_filename, self._dataset.get_dataset_name(), len(graph_clusters),
                                   'cluster', m_end-start, len(mutations), threshold, ms.get_cluster_amount(),
                                   unit_clustering.get_max_cluster_size(), unit_clustering.get_min_cluster_size(),
                                   unit_clustering.get_mean_cluster_size(), ms.get_cluster_mutation_score(),
                                   ms_end-ms_start, (ms_end-ms_start)+(m_end-start)]

        # if 'neuron' in self._mutation_level:
        #     ms_start = time.time()
        #     df_vanilla = pd.DataFrame(
        #         columns=['Model_Type', 'Dataset', 'Mutatable_Layers', 'Mutation_Level', 'Mutate_time',
        #                  'Number_of_Mutants', 'Mutation_Score', 'MS_time', 'Total_time'])
        #     ms.run()
        #     ms_end = time.time()
        #     df_vanilla.concat([self._model_filename, self._dataset.get_dataset_name(), 'neuron',
        #                        m_end - start, len(mutations), ms.get_mutation_score(), ms_end - ms_start,
        #                        (ms_end - ms_start) + (m_end - start)])

        return [], df_cluster




