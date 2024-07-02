import numpy as np
from tqdm import tqdm
import utils
from unit_clusterer import MutableUnitCluster

class MutationGenerator():
    def __init__(self, model_name, model, clusters, mutation_level, mutation_ratio):
        self._model_name = model_name
        self._model = model #load_model(model_filename)
        #if not isinstance(model, Sequential):
        #    raise ValueError('An instance of Sequential is expected')
        self._MO = MutationOperator()
        self._MGOpUtils = MutationOperatorsUtils()
        self._clusters = self._MGOpUtils.cluster_splitting(clusters, len(model.layers))
        self._mutation_level = mutation_level
        self._mutation_ratio = mutation_ratio
        self._consumers = []
        #self._fraction = fraction
        self.model_utils = utils.ModelUtils()

    def get_mutations(self, mutator_list):#, consumers):
        # if not consumers:
        #     self._consumers = []
        # else:
        #     self._consumers = consumers
        self._apply_mutators(mutator_list)

    # def _submit_mutation(self, uid, location, model, description):
    #     mutation = Mutation(uid, location, model, description)
    #     for consumer in self._consumers:
    #         consumer.consume(mutation)

    def _apply_mutators(self, mutator_list):

        if self._mutation_level in 'cluster':
            self._mutation_ratio = 1
        # GF_model = self.model_utils.model_copy(self._model, 'GF')
        if 'fcnn' in self._model_name:
            m = [1, 2, 3]
        elif 'lenet' in self._model_name:
            m = [0, 2, 5, 6]
        else:
            m = None
        clusters = self._clusters
        if 'GF' in mutator_list:
            GF_model = self._MO.Gaussian_Fuzzing_Mutator(self._model, self._clusters, self._mutation_ratio, self._mutation_level,
                                                         prob_distribution='normal', STD=0.1, lower_bound=None, upper_bound=None, lam=None, mutated_layer_indices=m)
        print("\n-------------------------------------------\n")
        if 'NEB' in mutator_list:
            NEB_model = self._MO.Neuron_Effect_Blocking_Mutation(self._model, self._clusters, self._mutation_ratio, self._mutation_level, m)
                #self.model_utils.model_copy(self._model, 'NEB'))
        print("\n-------------------------------------------\n")
        if 'NAI' in mutator_list:
            NAI_model = self._MO.Neuron_Activation_Inversion_Mutation(self._model, self._clusters, self._mutation_ratio, self._mutation_level, m)
            #= self.model_utils.model_copy(self._model, 'NAI')

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


class MutationOperator():
    def __init__(self):
        self._utils = utils.GeneralUtils()
        self._model_utils = utils.ModelUtils()
        self._check = utils.ExaminationalUtils()
        self._MGOp_utils = MutationOperatorsUtils()

    def Gaussian_Fuzzing_Mutator(self, model, clusters, mutation_ratio, mutation_level='neuron', prob_distribution='normal', STD=0.1, lower_bound=None, upper_bound=None, lam=None, mutated_layer_indices=None):
        self._check.mutation_ratio_range_check(mutation_ratio)

        valid_prob_distribution_types = ['normal', 'uniform']
        assert prob_distribution in valid_prob_distribution_types, 'The probability distribution type ' + prob_distribution + ' is not implemented in GF mutation operator'
        if prob_distribution == 'uniform' and ((lower_bound is None) or (upper_bound is None)):
            raise ValueError('In uniform distribution, users are required to specify the lower bound and upper bound of noises')
        if prob_distribution == 'exponential' and (lam is None):
            raise ValueError('In exponential distribution, users are required to specify the lambda value')

        valid_mutation_levels = ['neuron', 'cluster']
        assert mutation_level in valid_mutation_levels, 'The mutation level ' + mutation_level + ' is not implemented in GF mutation operator'

        GF_model = self._model_utils.model_copy(model, 'GF')
        layers = []
        for i in range(0, len(GF_model.layers) - 1):
            layers += [GF_model.layers[i]]
        # layers = [l for l in GF_model.layers][1:-1]

        num_of_layers = len(layers)
        print("num of layers" + str(num_of_layers))
        self._check.valid_indices_of_mutated_layers_check(num_of_layers, mutated_layer_indices)
        layers_should_be_mutated = self._model_utils.get_booleans_of_layers_should_be_mutated(num_of_layers,
                                                                                              mutated_layer_indices)
        layers = [l for b, l in zip(layers_should_be_mutated, layers) if b]

        if mutation_level == 'cluster':
            for index, layer in enumerate(layers):
                weights = layer.get_weights()
                if not (len(weights) == 0): # and layers_should_be_mutated[index]:
                    layer_clusters = clusters[index]
                    clusters_to_be_mutated = self._utils.generate_permutation(len(layer_clusters), mutation_ratio)
                    print(clusters_to_be_mutated) # permutation)
                    #clusters_to_be_mutated = permutation
                    for ci, cluster in enumerate(layer_clusters):
                        if ci in clusters_to_be_mutated:
                            print("layer " + str(mutated_layer_indices[index]) + " cluster " + str(ci))
                            #neuron_indexes = cluster
                            neuron_index = np.random.choice(cluster)
                            neuron = weights[0][..., neuron_index]
                            neuron_shape = neuron.shape
                            flat_neuron = neuron.flatten()
                            GF_flat_neuron = self._MGOp_utils.GF_on_list(flat_neuron, mutation_ratio, prob_distribution, STD,
                                                                       lower_bound, upper_bound, lam)
                            GF_neuron_weights = GF_flat_neuron.reshape(neuron_shape)

                            neuron = weights[1][neuron_index]
                            neuron_shape = neuron.shape
                            flat_neuron = neuron.flatten()
                            GF_flat_neuron = self._MGOp_utils.GF_on_list(flat_neuron, mutation_ratio, prob_distribution, STD,
                                                                       lower_bound, upper_bound, lam)
                            GF_neuron_biases = GF_flat_neuron.reshape(neuron_shape)

                            for ni in cluster:
                                weights[0][..., ni] = GF_neuron_weights
                                weights[1][ni] = GF_neuron_biases
                            #print("weights")
                            #print(weights[0][..., neuron_indexes[0]][:15])
                            #print(weights[1][neuron_indexes[0]])
                            #print(weights[0][..., neuron_indexes[1]][:15])
                            #print(weights[1][neuron_indexes[1]])
                    layer.set_weights(weights)
        else:
            layers = [l for l in GF_model.layers]
            #print("layers")
            #print(layers)
            #print()

            num_of_layers = len(layers)
            self._check.valid_indices_of_mutated_layers_check(num_of_layers, mutated_layer_indices)
            layers_should_be_mutated = self._model_utils.get_booleans_of_layers_should_be_mutated(num_of_layers,
                                                                                                  mutated_layer_indices)

            layers = [l for b, l in zip(layers_should_be_mutated, layers) if b]

            for index, layer in enumerate(layers):
                weights = layer.get_weights()
                new_weights = []
                #print("layer " + str(index))
                #print("layer shape: " + str(len(weights)))
                #numb = 0
                if not (len(weights) == 0) and layers_should_be_mutated[index]:
                    for val in weights:
                        val_shape = val.shape
                        #print(str(numb) + ":")
                        #print(val_shape)
                        #print("SHAPE")
                        #print(val[..., numb].shape)
                        flat_val = val.flatten()
                        GF_flat_val = self._MGOp_utils.GF_on_list(flat_val, mutation_ratio, prob_distribution, STD,
                                                                lower_bound, upper_bound, lam)
                        GF_val = GF_flat_val.reshape(val_shape)
                        new_weights.append(GF_val)
                        #numb += 1
                    layer.set_weights(new_weights)

        return GF_model

    def Neuron_Activation_Inversion_Mutation(self, model, clusters, mutation_ratio, mutation_level='neuron', mutated_layer_indices=None):
        self._check.mutation_ratio_range_check(mutation_ratio)

        NAI_model = self._model_utils.model_copy(model, 'NAI')

        layers = []
        for i in range(0, len(NAI_model.layers) - 1):
            layers += [NAI_model.layers[i]]

        #layers = [l for l in NAI_model.layers]

        num_of_layers = len(layers)
        self._check.valid_indices_of_mutated_layers_check(num_of_layers, mutated_layer_indices)
        layers_should_be_mutated = self._model_utils.get_booleans_of_layers_should_be_mutated(num_of_layers, mutated_layer_indices)
        layers = [l for b, l in zip(layers_should_be_mutated, layers) if b]
        # TODO might need to change layers back for 'neuron'

        if mutation_level == 'cluster':
            for index, layer in enumerate(layers):
                weights = layer.get_weights()
                layer_name = type(layer).__name__
                #new_weights = []
                if not (len(weights) == 0):
                    layer_clusters = clusters[index]
                    clusters_to_be_mutated = self._utils.generate_permutation(len(layer_clusters), mutation_ratio)
                    print(clusters_to_be_mutated)  # permutation)
                    # clusters_to_be_mutated = permutation

                    for ci, cluster in enumerate(layer_clusters):
                        if ci in clusters_to_be_mutated:
                            print("layer " + str(mutated_layer_indices[index]) + " cluster " + str(ci))
                            neuron_index = np.random.choice(cluster)
                            #for val in weights:
                            #val_shape = val.shape
                            neuron = weights[0]
                            neuron_shape = neuron.shape
                            #neuron = weights[0][..., neuron_index]
                            if (len(neuron.shape) is not 1): # and layers_should_be_mutated[index]:
                                if layer_name == 'Conv2D':
                                    filter_width, filter_height, num_of_input_channels, num_of_output_channels = neuron_shape
                                    permutation = self._utils.generate_permutation(len(cluster), mutation_ratio)#num_of_output_channels, mutation_ratio)
                                    for output_channel_index in cluster[permutation]:
                                        neuron[:, :, :, output_channel_index] *= -1
                                elif layer_name == 'Dense':
                                    input_dim, output_dim = neuron_shape
                                    permutation = self._utils.generate_permutation(len(cluster), mutation_ratio)
                                    for output_dim_index in cluster[permutation]:
                                        neuron[:, output_dim_index] *= -1
                                else:
                                    pass
                            weights[0] = neuron # I believe this is redundant
                            #new_weights.append(neuron)

                    layer.set_weights(weights)#new_weights)

        elif mutation_level == 'neuron':
            for index, layer in enumerate(layers):
                weights = layer.get_weights()
                layer_name = type(layer).__name__
                new_weights = []
                if not (len(weights) == 0):
                    for val in weights:
                        val_shape = val.shape
                        if (len(val.shape) is not 1) and layers_should_be_mutated[index]:
                            if layer_name == 'Conv2D':
                                filter_width, filter_height, num_of_input_channels, num_of_output_channels = val_shape
                                permutation = self._utils.generate_permutation(num_of_output_channels, mutation_ratio)
                                for output_channel_index in permutation:
                                    val[:, :, :, output_channel_index] *= -1
                            elif layer_name == 'Dense':
                                input_dim, output_dim = val_shape
                                permutation = self._utils.generate_permutation(output_dim, mutation_ratio)
                                for output_dim_index in permutation:
                                    val[:, output_dim_index] *= -1
                            else:
                                pass
                        new_weights.append(val)
                    layer.set_weights(new_weights)

        return NAI_model

    def Neuron_Effect_Blocking_Mutation(self, model, clusters, mutation_ratio, mutation_level='neuron', mutated_layer_indices=None):
        self._check.mutation_ratio_range_check(mutation_ratio)

        NEB_model = self._model_utils.model_copy(model, 'NEB')
        #layers = [l for l in NEB_model.layers]
        layers = []
        for i in range(0, len(NEB_model.layers) - 1):
            layers += [NEB_model.layers[i]]

        num_of_layers = len(layers)
        self._check.valid_indices_of_mutated_layers_check(num_of_layers, mutated_layer_indices)
        layers_should_be_mutated = self._model_utils.get_booleans_of_layers_should_be_mutated(num_of_layers, mutated_layer_indices)
        layers = [l for b, l in zip(layers_should_be_mutated, layers) if b]
        # TODO might need to change layers back for 'neuron'

        if mutation_level == 'cluster':
            for index, layer in enumerate(layers):
                weights = layer.get_weights()
                layer_name = type(layer).__name__
                #new_weights = []
                if not (len(weights) == 0):
                    layer_clusters = clusters[index]
                    clusters_to_be_mutated = self._utils.generate_permutation(len(layer_clusters), mutation_ratio)
                    print(clusters_to_be_mutated)  # permutation)

                    for ci, cluster in enumerate(layer_clusters):
                        if ci in clusters_to_be_mutated:
                            print("layer " + str(mutated_layer_indices[index]) + " cluster " + str(ci))
                            neuron_index = np.random.choice(cluster)
                            neuron = weights[0]
                            neuron_shape = neuron.shape
                            if (len(neuron_shape) is not 1): # and layers_should_be_mutated[index]:
                                if layer_name == 'Conv2D':
                                    filter_width, filter_height, num_of_input_channels, num_of_output_channels = neuron_shape
                                    permutation = self._utils.generate_permutation(num_of_input_channels, mutation_ratio) #num_of_input_channels, mutation_ratio)
                                    for input_channel_index in permutation:#cluster[permutation]:
                                        neuron[:, :, input_channel_index, :] = 0
                                elif layer_name == 'Dense':
                                    input_dim, output_dim = neuron_shape
                                    permutation = self._utils.generate_permutation(len(cluster), mutation_ratio)#input_dim, mutation_ratio)
                                    for input_index in cluster[permutation]:
                                        neuron[input_index] = 0
                                else:
                                    pass
                            weights[0] = neuron # I believe this is redundant
                        layer.set_weights(weights)


        elif mutation_level == 'neuron':
            for index, layer in enumerate(layers):
                weights = layer.get_weights()
                layer_name = type(layer).__name__
                new_weights = []
                if not (len(weights) == 0):
                    for val in weights:
                        val_shape = val.shape
                        if (len(val.shape) is not 1) and layers_should_be_mutated[index]:
                            if layer_name == 'Conv2D':
                                filter_width, filter_height, num_of_input_channels, num_of_output_channels = val_shape
                                permutation = self._utils.generate_permutation(num_of_input_channels, mutation_ratio)
                                for input_channel_index in permutation:
                                    val[:, :, input_channel_index, :] = 0
                            elif layer_name == 'Dense':
                                input_dim, output_dim = val_shape
                                permutation = self._utils.generate_permutation(input_dim, mutation_ratio)
                                for input_index in permutation:
                                    val[input_index] = 0
                            else:
                                pass
                        new_weights.append(val)
                    layer.set_weights(new_weights)

        return NEB_model


class MutationOperatorsUtils():
    def __init__(self):
        self._utils = utils.GeneralUtils()
        self.LD_mut_candidates = ['Dense']
        self.LAm_mut_candidates = ['Dense']

    def GF_on_list(self, lst, mutation_ratio, prob_distribution, STD, lower_bound, upper_bound, lam):
        copy_lst = lst.copy()
        number_of_data = len(copy_lst)
        if mutation_ratio < 1:
            permutation = self._utils.generate_permutation(number_of_data, mutation_ratio)
        else:
            permutation = [x for x in range(number_of_data)]

        if prob_distribution == 'normal':
            #print("permutation")
            #print(permutation)
            #print(permutation.shape)
            #print(type(permutation))
            #print("copy_list 1")
            #print(copy_lst.shape)
            #print(copy_lst[:15])
            #print(type(copy_lst))
            #print(copy_lst)
            copy_lst[permutation] += np.random.normal(scale=STD, size=len(permutation))
            #copy_lst[permutation] += np.random.normal(scale=STD, size=len(permutation))
            #print("copy_list 2")
            #print(copy_lst[:15])
            #print(copy_lst.shape)
            #print(copy_lst)
        elif prob_distribution == 'uniform':
            copy_lst[permutation] += np.random.uniform(low=lower_bound, high=upper_bound, size=len(permutation))
        elif prob_distribution == 'exponential':
            assert lam is not 0
            scale = 1 / lam
            copy_lst[permutation] += np.random.exponential(scale=scale, size=len(permutation))
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



