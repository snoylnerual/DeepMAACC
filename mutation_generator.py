import numpy as np
from tqdm import tqdm
import utils
from unit_clusterer import MutableUnitCluster
class MutationOperator:
    def __init__(self):
        self._mutant_number = 0
        self._utils = utils.GeneralUtils()
        self._model_utils = utils.ModelUtils()
        self._check = utils.ExaminationalUtils()
        self._MGOp_utils = MutationOperatorsUtils()

    def Gaussian_Fuzzing_Mutator(self, model, clusters, mutation_level='neuron', prob_distribution='normal', STD=0.1, lower_bound=None, upper_bound=None, lam=None):
        #TODO: Haven't changed this to new clustering method
        valid_prob_distribution_types = ['normal', 'uniform']
        assert prob_distribution in valid_prob_distribution_types, 'The probability distribution type ' + prob_distribution + ' is not implemented in GF mutation operator'
        if prob_distribution == 'uniform' and ((lower_bound is None) or (upper_bound is None)):
            raise ValueError('In uniform distribution, users are required to specify the lower bound and upper bound of noises')
        if prob_distribution == 'exponential' and (lam is None):
            raise ValueError('In exponential distribution, users are required to specify the lambda value')

        valid_mutation_levels = ['neuron', 'cluster']
        assert mutation_level in valid_mutation_levels, 'The mutation level ' + mutation_level + ' is not implemented in GF mutation operator'

        GF_model = self._model_utils.model_copy(model, 'GF')
        # layers = []
        # for i in range(0, len(GF_model.layers) - 1):
        #     layers += [GF_model.layers[i]]
        # layers = [l for l in GF_model.layers][1:-1]
        layers = [l for l in GF_model.layers]

        num_of_layers = len(layers)
        # print('------------------------------------')
        # print("num of layers " + str(num_of_layers))
        self._check.valid_indices_of_mutated_layers_check(num_of_layers, [])
        layers_should_be_mutated = self._model_utils.get_booleans_of_layers_should_be_mutated(num_of_layers,[])
        layers = [l for b, l in zip(layers_should_be_mutated, layers) if b]

        if mutation_level == 'cluster':
            for index, layer in enumerate(layers):
                weights = layer.get_weights()
                # print('weights len: ' + str(len(weights)))
                # print('weights[0] : '+str(weights[0].shape))
                if not (len(weights) == 0):  # and layers_should_be_mutated[index]:
                    layer_clusters = clusters[index]  # gives the clusters for this layer
                    for ci, cluster in enumerate(layer_clusters):
                        # print("layer " + str(mutated_layer_indices[index]) + " cluster " + str(ci))
                        # print(cluster)
                        for i, v in enumerate(weights):  # weights for each layer are made up of weights and biases
                            cluster_weights = v[..., cluster]
                            cluster_shape = cluster_weights.shape
                            flat_cluster = cluster_weights.flatten()
                            GF_flat_cluster = self._MGOp_utils.GF_on_list(flat_cluster, prob_distribution, STD,
                                                                          lower_bound, upper_bound, lam)
                            GF_cluster_weights = GF_flat_cluster.reshape(cluster_shape)

                            if i==0:
                                weights[0][..., cluster] = GF_cluster_weights
                            elif i==1:
                                weights[1][cluster] = GF_cluster_weights
                    layer.set_weights(weights)

        else:
            for index, layer in enumerate(layers):
                weights = layer.get_weights()
                if not (len(weights) == 0) and layers_should_be_mutated[index]:
                    val = weights[0]
                    bias = weights[1]
                    val_shape = val.shape
                    for i in range(val_shape[1]):
                        neuron = val[..., i]
                        neuron_shape = neuron.shape
                        flat_neuron = neuron.flatten()
                        GF_flat_neuron = self._MGOp_utils.GF_on_list(flat_neuron, prob_distribution, STD,
                                                                     lower_bound, upper_bound, lam)
                        GF_neuron = GF_flat_neuron.reshape(neuron_shape)

                        GF_bias = self._MGOp_utils.GF_on_list([bias[i]], prob_distribution, STD,
                                                              lower_bound, upper_bound, lam)
                        weights[0][..., i] = GF_neuron
                        weights[1][i] = GF_bias
                    layer.set_weights(weights)

        return GF_model

    def Change_Weight_Mutator(self, model, clusters, mutation_level='neuron', mutation_percent=0.1):
        list_of_mutants = []

        if mutation_level == 'cluster':
            for cluster in clusters:
                CW_model = self._model_utils.model_copy(model, 'CW')
                cluster_indices = np.array(cluster.get_unit_indices())
                layer_index = cluster.get_layer_index()
                layer_name = type(CW_model.layers[layer_index]).__name__
                weights = CW_model.layers[layer_index].get_weights()
                # temp_b = weights[1][cluster_indices]
                if layer_name == 'Conv2D':
                    # temp_w = np.array(weights[0][:, :, :, cluster_indices])
                    weights[0][:, :, :, cluster_indices] *= (mutation_percent+1)
                elif layer_name == 'Dense':
                    # temp_w = np.array(weights[0][:, cluster_indices])
                    weights[0][:, cluster_indices] *= (mutation_percent+1)
                else:
                    pass
                weights[1][cluster_indices] *= (mutation_percent+1)
                CW_model.layers[layer_index].set_weights(weights)
                list_of_mutants.append(Mutant(self._mutant_number, CW_model, 'CW_Cluster', layer_index, cluster_indices))
                self._mutant_number += 1
                # if layer_name == 'Conv2D':
                #     weights[0][:, :, :, cluster_indices] = temp_w
                # elif layer_name == 'Dense':
                #     weights[0][:, cluster_indices] = temp_w
                # weights[1][cluster_indices] = temp_b
                # CW_model.layers[layer_index].set_weights(weights)

        elif mutation_level == 'neuron':
            for layer_index, layer in enumerate(model.layers):
                weights = layer.get_weights()
                if not (len(weights) == 0):  # weights will have length of zero if they shouldn't be edited
                    layer_name = type(layer).__name__
                    CONV2D = layer_name == 'Conv2D'
                    DENSE = layer_name == 'Dense'
                    if CONV2D: i = 3
                    elif DENSE: i = 1
                    else: pass
                    for neuron_index in range(weights[0].shape[i]):
                        CW_model = self._model_utils.model_copy(model, 'CW')
                        # temp_b = weights[1][neuron_index]
                        if CONV2D:
                            # temp_w = np.array(weights[0][:, :, :, neuron_index])
                            weights[0][:, :, :, neuron_index] *= (mutation_percent + 1)
                        elif DENSE:
                            # temp_w = np.array(weights[0][:, neuron_index])
                            weights[0][:, neuron_index] *= (mutation_percent + 1)
                        weights[1][neuron_index] *= (mutation_percent + 1)
                        CW_model.layers[layer_index].set_weights(weights)
                        list_of_mutants.append(Mutant(self._mutant_number, CW_model, 'CW', layer_index, neuron_index))
                        self._mutant_number += 1
                        # if CONV2D:
                        #     weights[0][:, :, :, neuron_index] = temp_w
                        # elif DENSE:
                        #     weights[0][:, neuron_index] = temp_w
                        # weights[1][neuron_index] = temp_b
                        # CW_model.layers[layer_index].set_weights(weights)

        return list_of_mutants
        # CW_model = self._model_utils.model_copy(model, 'CW')
        # layers = [l for l in CW_model.layers]
        #
        # num_of_layers = len(layers)
        # self._check.valid_indices_of_mutated_layers_check(num_of_layers, mutated_layer_indices)
        # layers_should_be_mutated = self._model_utils.get_booleans_of_layers_should_be_mutated(num_of_layers,
        #                                                                                       mutated_layer_indices)
        # layers = [l for b, l in zip(layers_should_be_mutated, layers) if b]
        # multiplication_factor = mutation_percent+1
        # if mutation_level == 'cluster':
        #     for index, layer in enumerate(layers):
        #         weights = layer.get_weights()
        #         layer_name = type(layer).__name__
        #         if not (len(weights) == 0):
        #             layer_clusters = clusters[index]
        #             for ci, cluster in enumerate(layer_clusters):
        #                 # print("layer " + str(mutated_layer_indices[index]) + " cluster " + str(ci))
        #                 vals = weights[0]
        #                 vals_shape = vals.shape
        #                 if (len(vals_shape) != 1):
        #                     if layer_name == 'Conv2D':
        #                         vals[:, :, :, cluster] *= multiplication_factor
        #                     elif layer_name == 'Dense':
        #                         vals[:, cluster] *= multiplication_factor
        #                     else:
        #                         pass
        #                 weights[0] = vals
        #             layer.set_weights(weights)
        #
        # elif mutation_level == 'neuron':
        #     for index, layer in enumerate(layers):
        #         weights = layer.get_weights()
        #         layer_name = type(layer).__name__
        #         new_weights = []
        #         if not (len(weights) == 0):
        #             vals = weights[0]
        #             vals_shape = vals.shape
        #             # for val in weights:
        #
        #             if (len(vals_shape) != 1) and layers_should_be_mutated[index]:
        #                 if layer_name == 'Conv2D':
        #                     for neuron_index in range(vals_shape[3]):
        #                         vals[:, :, :, neuron_index] *= multiplication_factor
        #                 elif layer_name == 'Dense':
        #                     for neuron_index in range(vals_shape[1]):
        #                         vals[:, neuron_index] *= multiplication_factor
        #                 else:
        #                     pass
        #             layer.set_weights(weights)
        # return CW_model

    def Neuron_Activation_Inversion_Mutation(self, model, clusters, mutation_level='neuron'):
        list_of_mutants = []

        if mutation_level == 'cluster':
            for cluster in clusters:
                NAI_model = self._model_utils.model_copy(model, 'NAI')
                cluster_indices = np.array(cluster.get_unit_indices())
                layer_index = cluster.get_layer_index()
                layer_name = type(NAI_model.layers[layer_index]).__name__
                weights = NAI_model.layers[layer_index].get_weights()
                #temp_b = weights[1][cluster_indices]
                if layer_name == 'Conv2D':
                    # temp_w = np.array(weights[0][:, :, :, cluster_indices])
                    weights[0][:, :, :, cluster_indices] *= -1
                elif layer_name == 'Dense':
                    # temp_w = np.array(weights[0][:, cluster_indices])
                    weights[0][:, cluster_indices] *= -1
                else:
                    pass
                # weights[1][cluster_indices] *= -1
                NAI_model.layers[layer_index].set_weights(weights)
                list_of_mutants.append(Mutant(self._mutant_number, NAI_model, 'NAI_Cluster', layer_index, cluster_indices))
                self._mutant_number += 1
                # if layer_name == 'Conv2D':
                #     weights[0][:, :, :, cluster_indices] = temp_w
                # elif layer_name == 'Dense':
                #     weights[0][:, cluster_indices] = temp_w
                # # weights[1][cluster_indices] = temp_b
                # NAI_model.layers[layer_index].set_weights(weights)

        elif mutation_level == 'neuron':
            for layer_index, layer in enumerate(model.layers):
                weights = layer.get_weights()
                if not (len(weights) == 0):  # weights will have length of zero if they shouldn't be edited
                    layer_name = type(layer).__name__
                    CONV2D = layer_name == 'Conv2D'
                    DENSE = layer_name == 'Dense'
                    if CONV2D:
                        i = 3
                    elif DENSE:
                        i = 1
                    else:
                        pass
                    for neuron_index in range(weights[0].shape[i]):
                        NAI_model = self._model_utils.model_copy(model, 'NAI')
                        # temp_b = weights[1][neuron_index]
                        if CONV2D:
                            # temp_w = np.array(weights[0][:, :, :, neuron_index])
                            weights[0][:, :, :, neuron_index] *= -1
                        elif DENSE:
                            # temp_w = np.array(weights[0][:, neuron_index])
                            weights[0][:, neuron_index] *= -1
                        # weights[1][neuron_index] *= -1
                        NAI_model.layers[layer_index].set_weights(weights)
                        list_of_mutants.append(Mutant(self._mutant_number, NAI_model, 'NAI', layer_index, neuron_index))
                        self._mutant_number += 1
                        # if CONV2D:
                        #     weights[0][:, :, :, neuron_index] = temp_w
                        # elif DENSE:
                        #     weights[0][:, neuron_index] = temp_w
                        # #weights[1][neuron_index] = temp_b
                        # NAI_model.layers[layer_index].set_weights(weights)

        return list_of_mutants

        #
        #
        # # layers = []
        # # for i in range(0, len(NAI_model.layers) - 1):
        # #     layers += [NAI_model.layers[i]]
        #
        # layers = [l for l in NAI_model.layers]
        #
        # num_of_layers = len(layers)
        # self._check.valid_indices_of_mutated_layers_check(num_of_layers, mutated_layer_indices)
        # layers_should_be_mutated = self._model_utils.get_booleans_of_layers_should_be_mutated(num_of_layers, mutated_layer_indices)
        # layers = [l for b, l in zip(layers_should_be_mutated, layers) if b]
        # # TODO might need to change layers back for 'neuron'
        #
        # if mutation_level == 'cluster':
        #     for index, layer in enumerate(layers):
        #         weights = layer.get_weights()
        #         layer_name = type(layer).__name__
        #         #new_weights = []
        #         if not (len(weights) == 0):
        #             layer_clusters = clusters[index]
        #             #clusters_to_be_mutated = self._utils.generate_permutation(len(layer_clusters), mutation_ratio)
        #             ## print(clusters_to_be_mutated)  # permutation)
        #             # clusters_to_be_mutated = permutation
        #
        #             for ci, cluster in enumerate(layer_clusters):
        #                 #if ci in clusters_to_be_mutated:
        #                 # print("layer " + str(mutated_layer_indices[index]) + " cluster " + str(ci))
        #                 #neuron_index = np.random.choice(cluster)
        #                 #for val in weights:
        #                 #val_shape = val.shape
        #                 vals = weights[0]
        #                 vals_shape = vals.shape
        #                 #neuron = weights[0][..., neuron_index]
        #                 if (len(vals_shape) != 1): # and layers_should_be_mutated[index]:
        #                     if layer_name == 'Conv2D':
        #                         # filter_width, filter_height, num_of_input_channels, num_of_output_channels = vals_shape
        #                         # permutation = self._utils.generate_permutation(len(cluster), mutation_ratio)#num_of_output_channels, mutation_ratio)
        #                         vals[:, :, :, cluster] *= -1
        #                         # for output_channel_index in cluster[permutation]:
        #                         #    neuron[:, :, :, output_channel_index] *= -1
        #                     elif layer_name == 'Dense':
        #                         #input_dim, output_dim = neuron_shape
        #                         #permutation = self._utils.generate_permutation(len(cluster), mutation_ratio)
        #                         #for output_dim_index in cluster[permutation]:
        #                         #    neuron[:, output_dim_index] *= -1
        #                         vals[:, cluster] *= -1
        #                     else:
        #                         pass
        #                 weights[0] = vals # I believe this is redundant
        #                 #new_weights.append(neuron)
        #
        #             layer.set_weights(weights)#new_weights)
        #
        # elif mutation_level == 'neuron':
        #     for index, layer in enumerate(layers):
        #         weights = layer.get_weights()
        #         layer_name = type(layer).__name__
        #         new_weights = []
        #         if not (len(weights) == 0):
        #             vals = weights[0]
        #             vals_shape = vals.shape
        #             #for val in weights:
        #
        #             if (len(vals_shape) != 1) and layers_should_be_mutated[index]:
        #                 if layer_name == 'Conv2D':
        #                     # filter_width, filter_height, num_of_input_channels, num_of_output_channels = val_shape
        #                     # permutation = self._utils.generate_permutation(num_of_output_channels, mutation_ratio)
        #                     # for output_channel_index in permutation:
        #                     #     val[:, :, :, output_channel_index] *= -1
        #                     for neuron_index in range(vals_shape[3]):
        #                         vals[:, :, :, neuron_index] *= -1
        #                 elif layer_name == 'Dense':
        #                     # input_dim, output_dim = val_shape
        #                     # permutation = self._utils.generate_permutation(output_dim, mutation_ratio)
        #                     # for output_dim_index in permutation:
        #                     #    val[:, output_dim_index] *= -1
        #                     for neuron_index in range(vals_shape[1]):
        #                         vals[:, neuron_index] *= -1
        #                 else:
        #                     pass
        #                 #new_weights.append(val)
        #             layer.set_weights(weights)#new_weights)
        # return NAI_model

    def Neuron_Effect_Blocking_Mutation(self, model, clusters, mutation_level='neuron'):
        list_of_mutants = []

        if mutation_level == 'cluster':
            for cluster in clusters:
                NEB_model = self._model_utils.model_copy(model, 'NEB')
                cluster_indices = np.array(cluster.get_unit_indices())
                layer_index = cluster.get_layer_index()
                layer_name = type(NEB_model.layers[layer_index]).__name__
                weights = NEB_model.layers[layer_index].get_weights()  # [..., cluster_indices]
                vals_shape = weights[0].shape
                # we do the [...,cluster_indices] because we don't filter on that later
                # temp_b = weights[1][cluster_indices]
                if layer_name == 'Conv2D':
                    cluster_input_indexes = np.array([num for num in range(vals_shape[2])])
                    # temp_w = np.array(weights[0][:, :, :, cluster_indices])
                    weights[0][:, :, cluster_input_indexes][cluster_indices] = 0
                    #TODO: I have feeling this will not work either so need to check on that?
                elif layer_name == 'Dense':
                    # cluster_input_indexes = np.array([num for num in range(vals_shape[0])])
                    # temp_w = np.array(weights[0][:, cluster_indices])
                    weights[0][..., cluster_indices] *= 0
                else:
                    pass
                # weights[1][cluster_indices] *= -1
                NEB_model.layers[layer_index].set_weights(weights)
                list_of_mutants.append(Mutant(self._mutant_number, NEB_model, 'NEB_Cluster', layer_index, cluster_indices))
                # self._mutant_number += 1
                # if layer_name == 'Conv2D':
                #     weights[0][:, :, :, cluster_indices] = temp_w
                # elif layer_name == 'Dense':
                #     weights[0][:, cluster_indices] = temp_w
                # # weights[1][cluster_indices] = temp_b
                # NEB_model.layers[layer_index].set_weights(weights)

        elif mutation_level == 'neuron':
            for layer_index, layer in enumerate(model.layers):
                weights = layer.get_weights()
                if not (len(weights) == 0):  # weights will have length of zero if they shouldn't be edited
                    layer_name = type(layer).__name__
                    CONV2D = layer_name == 'Conv2D'
                    DENSE = layer_name == 'Dense'
                    if CONV2D:
                        i = 3
                    elif DENSE:
                        i = 1
                    else:
                        pass
                    for neuron_index in range(weights[0].shape[i]):
                        NEB_model = self._model_utils.model_copy(model, 'NEB')
                        # temp_b = weights[1][neuron_index]
                        for val in weights:
                            val_shape = val.shape
                            if (len(val.shape) != 1):
                                if CONV2D:
                                    # temp_w = np.array(weights[0][:, :, :, neuron_index])
                                    input_neuron_indices = [n for n in range(val_shape[2])]
                                    weights[0][:, :, input_neuron_indices, neuron_index] = 0
                                elif DENSE:
                                    # temp_w = np.array(weights[0][:, neuron_index])
                                    input_neuron_indices = [n for n in range(val_shape[0])]
                                    weights[0][input_neuron_indices, neuron_index] = 0
                                # weights[1][neuron_index] *= -1
                                NEB_model.layers[layer_index].set_weights(weights)
                                list_of_mutants.append(Mutant(self._mutant_number, NEB_model, 'NEB', layer_index, neuron_index))
                                self._mutant_number += 1
                                # if CONV2D:
                                #     weights[0][:, :, :, neuron_index] = temp_w
                                # elif DENSE:
                                #     weights[0][:, neuron_index] = temp_w
                                # weights[1][neuron_index] = temp_b
                    # NEB_model.layers[layer_index].set_weights(weights)

        return list_of_mutants
        #
        # layers = [l for l in NEB_model.layers]
        # # layers = []
        # # for i in range(0, len(NEB_model.layers) - 1):
        # #     layers += [NEB_model.layers[i]]
        #
        # num_of_layers = len(layers)
        # self._check.valid_indices_of_mutated_layers_check(num_of_layers, mutated_layer_indices)
        # layers_should_be_mutated = self._model_utils.get_booleans_of_layers_should_be_mutated(num_of_layers, mutated_layer_indices)
        # layers = [l for b, l in zip(layers_should_be_mutated, layers) if b]
        # # TODO might need to change layers back for 'neuron'
        #
        # if mutation_level == 'cluster':
        #     for index, layer in enumerate(layers):
        #         weights = layer.get_weights()
        #         # print('weights len: ' + str(len(weights)))
        #         # print('weights[0] : ' + str(weights[0].shape))
        #         # print('weights[1] : ' + str(weights[1].shape))
        #         layer_name = type(layer).__name__
        #         #new_weights = []
        #         if not (len(weights) == 0):
        #             layer_clusters = clusters[index]
        #             #clusters_to_be_mutated = self._utils.generate_permutation(len(layer_clusters), mutation_ratio)
        #             ## print(clusters_to_be_mutated)  # permutation)
        #
        #             for ci, cluster in enumerate(layer_clusters):
        #                 #if ci in clusters_to_be_mutated:
        #                 # print("layer " + str(mutated_layer_indices[index]) + " cluster " + str(ci))
        #                 neuron_index = np.random.choice(cluster)
        #                 vals = weights[0][..., cluster]
        #                 vals_shape = vals.shape
        #                 if (len(vals_shape) != 1): # and layers_should_be_mutated[index]:
        #                     if layer_name == 'Conv2D':
        #                         # filter_width, filter_height, num_of_input_channels, num_of_output_channels = neuron_shape
        #                         # permutation = self._utils.generate_permutation(num_of_input_channels, mutation_ratio) #num_of_input_channels, mutation_ratio)
        #                         # for input_channel_index in permutation:#cluster[permutation]:
        #                         #     neuron[:, :, input_channel_index, :] = 0
        #                         #for neuron_index in range(vals_shape[3]):
        #                         cluster_input_indexes = np.array([num for num in range(vals_shape[2])])
        #                         # # print('=====')
        #                         # # print(vals[:, :, cluster_input_indexes, :])
        #                         vals[:, :, cluster_input_indexes, :] = 0
        #                         # # print(vals[:, :, cluster_input_indexes, :])
        #                         # # print('=====')
        #                         # # print(vals[...,])
        #                         # # print('===============')
        #                     elif layer_name == 'Dense':
        #                         # input_dim, output_dim = neuron_shape
        #                         # permutation = self._utils.generate_permutation(len(cluster), mutation_ratio)#input_dim, mutation_ratio)
        #                         # for input_index in cluster[permutation]:
        #                         #     neuron[input_index] = 0
        #                         #for neuron_index in range(vals_shape[1]):
        #                         cluster_input_indexes = np.array([num for num in range(vals_shape[0])])
        #                         vals[cluster_input_indexes] = 0
        #                     else:
        #                         pass
        #                 #weights[0] = vals # I believe this is redundant
        #             layer.set_weights(weights)
        #
        #
        # elif mutation_level == 'neuron':
        #     for index, layer in enumerate(layers):
        #         weights = layer.get_weights()
        #         layer_name = type(layer).__name__
        #         # new_weights = []
        #         if not (len(weights) == 0):
        #             for vals in weights:
        #                 #vals = weights[0]
        #                 vals_shape = vals.shape
        #                 if (len(vals_shape) != 1) and layers_should_be_mutated[index]:
        #                     if layer_name == 'Conv2D':
        #                         # filter_width, filter_height, num_of_input_channels, num_of_output_channels = val_shape
        #                         # permutation = self._utils.generate_permutation(num_of_input_channels, mutation_ratio)
        #                         # for input_channel_index in permutation:
        #                         #    val[:, :, input_channel_index, :] = 0
        #                         for input_neuron_index in range(vals_shape[2]):
        #                             vals[:, :, input_neuron_index, :] = 0
        #                     elif layer_name == 'Dense':
        #                         # input_dim, output_dim = val_shape
        #                         # permutation = self._utils.generate_permutation(input_dim, mutation_ratio)
        #                         # for input_index in permutation:
        #                         #     val[input_index] = 0
        #                         for input_neuron_index in range(vals_shape[0]):
        #                             vals[input_neuron_index] = 0
        #                     else:
        #                         pass
        #                 #new_weights.append(val)
        #             layer.set_weights(weights)#new_weights)
        #
        # return NEB_model


from Mutant import Mutant


class MutationOperatorsUtils:
    def __init__(self):
        self._utils = utils.GeneralUtils()
        self.LD_mut_candidates = ['Dense']
        self.LAm_mut_candidates = ['Dense']

    def GF_on_list(self, lst, prob_distribution, STD, lower_bound, upper_bound, lam):
        copy_lst = lst.copy()
        number_of_data = len(copy_lst)
        # if mutation_ratio < 1:
        #     permutation = self._utils.generate_permutation(number_of_data, mutation_ratio)
        # else:
        #     permutation = [x for x in range(number_of_data)]

        if prob_distribution == 'normal':
            ## print("permutation")
            ## print(permutation)

            ## print(permutation.shape)
            ## print(type(permutation))
            ## print("copy_list 1")
            ## print(copy_lst.shape)
            ## print(copy_lst[:15])
            ## print(type(copy_lst))
            ## print(copy_lst)
            copy_lst += np.random.normal(scale=STD, size=len(copy_lst))
            #copy_lst[permutation] += np.random.normal(scale=STD, size=len(permutation))
            ## print("copy_list 2")
            ## print(copy_lst[:15])
            ## print(copy_lst.shape)
            ## print(copy_lst)
        elif prob_distribution == 'uniform':
            copy_lst += np.random.uniform(low=lower_bound, high=upper_bound, size=len(copy_lst))
        elif prob_distribution == 'exponential':
            assert lam != 0
            scale = 1 / lam
            copy_lst += np.random.exponential(scale=scale, size=len(copy_lst))
        else:
            pass

        return copy_lst

    def cluster_splitting(self, clusters, num_of_layers):
        # nums of layers, list of number of neurons
        # [#, [#,#,#]]
        new_cluster_array = []
        #layer_length = cluster_shapes[0]
        c = 0
        temp_arr = []
        for v in clusters:
            n = v.get_layer_index()
            if n == c:
                temp_arr.append(np.array(v.get_unit_indices()))
            else:
                if len(temp_arr) != 0:
                    new_cluster_array.append(temp_arr)
                temp_arr = [np.array(v.get_unit_indices())]
            c = n

        if len(temp_arr) != 0:
            new_cluster_array.append(temp_arr)
        return new_cluster_array


class MutationGenerator:

    def __init__(self, model_name, model, clusters, mutation_level):
        self._model_name = model_name
        self._model = model #load_model(model_filename)
        #if not isinstance(model, Sequential):
        #    raise ValueError('An instance of Sequential is expected')
        self._MO = MutationOperator()
        self._MGOpUtils = MutationOperatorsUtils()
        self._clusters = clusters  # self._MGOpUtils.cluster_splitting(clusters, len(model.layers))
        self._mutation_level = mutation_level
        self._mutation_percent = 0.1
        self._consumers = []
        self.model_utils = utils.ModelUtils()

    def set_mutation_percent(self, mutation_percent):
        self._mutation_percent = mutation_percent

    def get_mutations(self, mutator_list):#, consumers):
        # if not consumers:
        #     self._consumers = []
        # else:
        #     self._consumers = consumers
        return self._apply_mutators(mutator_list)

    # def _submit_mutation(self, uid, location, model, description):
    #     mutation = Mutation(uid, location, model, description)
    #     for consumer in self._consumers:
    #         consumer.consume(mutation)

    def _apply_mutators(self, mutator_list):
        mutated_models = []
        # GF_model = self.model_utils.model_copy(self._model, 'GF')
        if 'fcnn' in self._model_name:
            mCW = [1, 2, 3]
            mNAI = [1, 2, 3] #4] exclude last layer bc it excludes it in the unit clusterer
            mNEB = [1, 2, 3] #[0, cant do first one because it is just input layer with no neurons
        elif 'lenet' in self._model_name:
            mCW = [0, 2, 5, 6]
            mNAI = [0, 2, 5, 6] #7] exclude last layer bc it excludes it in the unit clusterer
            mNEB = [0, 2, 5, 6]
        elif 'resnet18' in self._model_name:
            mCW = [0, 2, 5, 6]
            mNAI = [0, 2, 5, 6]
            mNEB = [0, 2, 5, 6]
        elif 'alexnet' in self._model_name:
            mCW = [1, 4, 7, 9, 11, 15, 18, 21]
            mNAI = [1, 4, 7, 9, 11, 15, 18, 21]
            mNEB = [1, 4, 7, 9, 11, 15, 18, 21]
        elif 'vggnet16' in self._model_name:
            mCW = [0, 1, 4, 5, 8, 9, 10, 13, 14, 15, 18, 19, 20, 24, 26]
            mNAI = [0, 1, 4, 5, 8, 9, 10, 13, 14, 15, 18, 19, 20, 24, 26]
            mNEB = [0, 1, 4, 5, 8, 9, 10, 13, 14, 15, 18, 19, 20, 24, 26]
        else:
            m = None

        if 'GF' in mutator_list:
            mGF = [1, 2, 3]
            GF_model = self._MO.Gaussian_Fuzzing_Mutator(self._model, self._clusters, self._mutation_level,
                                                         prob_distribution='normal', STD=0.1, lower_bound=None, upper_bound=None, lam=None, mutated_layer_indices=mGF)
            GF_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            mutated_models.append(GF_model)
        # print("\n-------------------------------------------\n")
        if 'CW' in mutator_list:
            CW_models = self._MO.Change_Weight_Mutator(self._model, self._clusters, self._mutation_level, self._mutation_percent)  # , mutated_layer_indices=mCW)
            for CW_mutant in CW_models:
                CW_model = CW_mutant.get_model()
                CW_model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
                mutated_models.append(CW_mutant)
        # print("\n-------------------------------------------\n")
        if 'NEB' in mutator_list:
            NEB_models = self._MO.Neuron_Effect_Blocking_Mutation(self._model, self._clusters, self._mutation_level)  # , mNEB)
                #self.model_utils.model_copy(self._model, 'NEB'))
            for NEB_mutant in NEB_models:
                NEB_model = NEB_mutant.get_model()
                NEB_model.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
                mutated_models.append(NEB_mutant)
        # print("\n-------------------------------------------\n")
        if 'NAI' in mutator_list:
            NAI_models = self._MO.Neuron_Activation_Inversion_Mutation(self._model, self._clusters, self._mutation_level)  # , mNAI)
            #= self.model_utils.model_copy(self._model, 'NAI')
            for NAI_mutant in NAI_models:
                NAI_model = NAI_mutant.get_model()
                NAI_model.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
                mutated_models.append(NAI_mutant)
        return mutated_models

        #progress_bar = tqdm(total=3 * len(self._mutable_units), position=0, leave=True, desc="Analyzing Mutants")
        # for mu in self._mutable_units:
        #     location = Location(mu.get_layer_index(), mu.get_unit_indices())
        #     mu.add(self._fraction)
        #     self._submit_mutation(make_uid(mu),
        #                           location,
        #                           mu.get_model(),
        #                           'Incremented weights by %.2f%%' % (self._fraction * 100))
        #     mu.reset()
        #     progress_bar.update(1)
        #     mu.add(-self._fraction)
        #     self._submit_mutation(make_uid(mu),
        #                           location,
        #                           mu.get_model(),
        #                           'Decremented weights by %.2f%%' % (self._fraction * 100))
        #     mu.reset()
        #     progress_bar.update(1)
        #     mu.add(-1.0)
        #     self._submit_mutation(make_uid(mu),
        #                           location,
        #                           mu.get_model(),
        #                           'Deleted weights')
        #     mu.reset()
        #     progress_bar.update(1)



