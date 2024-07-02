# Main
from argparse import ArgumentParser
from keras.models import load_model
from unit_clusterer import UnitClustering
import numpy as np
from DMAACC import DMAACC


# Model, Mutate, Test
# Model, Cluster, Mutate, Test

def print_messages_MMM_generators(self, mode, network=None, test_datas=None, test_labels=None, model=None, mutated_model=None, STD=0.1, mutation_ratio=0):
    if mode in ['GF', 'WS', 'NEB', 'NAI', 'NS']:
        print('Before ' + mode)
        network.evaluate_model(model, test_datas, test_labels)
        print('After ' + mode + ', where the mutation ratio is', mutation_ratio)
        network.evaluate_model(mutated_model, test_datas, test_labels, mode)
    elif mode in ['LD', 'LAm', 'AFRm']:
        print('Before ' + mode)
        model.summary()
        network.evaluate_model(model, test_datas, test_labels)

        print('After ' + mode)
        mutated_model.summary()
        network.evaluate_model(mutated_model, test_datas, test_labels, mode)
    else:
        pass



if __name__ == "__main__":
    '''nums = np.array([110.1,111.1,112.1,113.1,114.1,115.1,116.1,117.1,118.1,119.1,1110.1])
    nums = nums[1:-1]
    print(nums)
    permutation = np.random.permutation(len(nums))
    permutation = permutation[:6]
    print(permutation)
    print(nums[permutation])
    #nums = nums[permutation]
    nums = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
    nums[permutation] += np.random.normal(scale=0.2, size=len(permutation))
    print(nums[permutation])'''


    parser = ArgumentParser()
    parser.add_argument('-m',
                        '--model',
                        dest='model_filename',
                        help='Model file in .h5 format',
                        required=True)
    parser.add_argument('-ml',
                        '--mutation_level',
                        dest='mutation_level',
                        help='Potential values are \'neuron\' or \'cluster\'',
                        required=True)
    parser.add_argument('-mr',
                        '--mutation_ratio',
                        dest='mutation_ratio',
                        help='Ratio of weights to be mutated',
                        default=0.15,
                        required=False)
    parser.add_argument('-std',
                        '--standard_deviation',
                        dest='standard_deviation',
                        help='Standard deviation used for Gauzzian Fuzzing mutator',
                        default=0.2,
                        required=False)
    # TODO add these into arguments prob_distribution='normal', lower_bound=None, upper_bound=None, lam=None, mutated_layer_indices=None
    # parser.add_argument('-tri',
    #                     '--train-in',
    #                     dest='train_in_filename',
    #                     help='Train input file in .npy format',
    #                     required=True)
    # parser.add_argument('-tro',
    #                     '--train-out',
    #                     dest='train_out_filename',
    #                     help='Train output file in .npy format',
    #                     required=True)
    # parser.add_argument('-tsi',
    #                     '--test-in',
    #                     dest='test_in_filename',
    #                     help='Test input file in .npy format',
    #                     required=True)
    # parser.add_argument('-tso',
    #                     '--test-out',
    #                     dest='test_out_filename',
    #                     help='Test output file in .npy format',
    #                     required=True)
    parser.add_argument('-n',
                        '--top',
                        dest='top_n',
                        default='2',
                        help='Top n groups be selected from each layer (default: 2)',
                        required=False)
    # parser.add_argument('-w',
    #                     '--weight-fraction',
    #                     dest='weight_fraction',
    #                     default='0.1',
    #                     help='The fraction to add/subtract when mutating the weights (default: 0.1)',
    #                     required=False)
    parser.add_argument('-c',
                        '--clusters-per-layer',
                        dest='clusters_per_layer',
                        default='5',
                        help='Number of clusters per layer. Ignored if one-unit-per-cluster activated.',
                        required=False)
    parser.add_argument('-u',
                        '--one-unit-per-cluster',
                        dest='one_unit_per_cluster',
                        default='False',
                        help='One unit per cluster (default: False)',
                        required=False)
    # parser.add_argument('-s',
    #                     '--slicer',
    #                     dest='slicer',
    #                     help='Slicer to be used (default: optimized). '
    #                          'Potential values are \'random\', \'ranked\', and \'optimized\'',
    #                     default='optimized',
    #                     required=False)
    # parser.add_argument('-o',
    #                     '--optimizer',
    #                     dest='optimizer',
    #                     help='Optimizer for re-training the last layer of the slice, e.g., \'adam\', \'sgd\', etc. '
    #                          '(default: adam)',
    #                     default='adam',
    #                     required=False)
    # parser.add_argument('-e',
    #                     '--epochs',
    #                     dest='epochs',
    #                     help='Number of epochs used for re-training the last layer of the slice (default: 5)',
    #                     default='64',
    #                     required=False)
    args = parser.parse_args()

    dmaacc_run = DMAACC()

    dmaacc_run.load_model(args.model_filename)
    dmaacc_run.set_mutation_level(args.mutation_level)
    dmaacc_run.set_mutation_ratio(float(args.mutation_ratio))
    dmaacc_run.set_standard_deviation(float(args.standard_deviation))
    dmaacc_run.set_cluster_selection_threshold(int(args.top_n))
    dmaacc_run.set_clusters_per_layer(int(args.clusters_per_layer))
    dmaacc_run.set_one_unit_per_cluster(args.one_unit_per_cluster.lower() == 'true')
    dmaacc_run.run()

    # model = load_model(args.model_filename)
    # unit_clustering = UnitClustering(model)
    # clusters = np.array(unit_clustering.get_clusters(5,False))
    #     #self._clusters_per_layer, one_unit_per_cluster=self._one_unit_per_cluster)
    # for index, mu in enumerate(clusters):
    #      print(index)
    #      print(str(mu.get_layer_index())+str(mu.get_unit_indices()))
    #      mu.add(.1)
    # print(type(clusters[0].get_unit_indices()))
    # c = np.array(clusters[0].get_unit_indices())
    # print(c)
    # print(type(c))
    # add progress bar later with tqdm
    #clusters = np.array_split(clusters, len(clusters)//5)#cluster_sz)
    #print(clusters)
    #layers * 5

    # import importlib
    # import utils
    # importlib.reload(utils)
    import model_mut_operators
    #model_muts_opts = model_mut_operators.ModelMutationOperators()
    #GF_model = model_muts_opts.GF_mut(model, clusters, mutation_ratio=args.mutation_ratio, STD=args.standard_deviation, mutation_level=args.mutation_level)
    # print(GF_model)
    #GF_mut has mutated_layer_indices=None
    #TODO: maybe this is what I should use with the clusters or something. further checking needed

    #print_messages_MMM_generators('GF', network=)
    # need to recompile trained model after we mutate it?"""
