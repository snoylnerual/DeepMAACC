import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from argparse import ArgumentParser
from network import Dataset
from network import Network
from DMAACC import DMAACC
import os.path
import pandas as pd


if __name__ == "__main__":
    # run_type = 'vanilla'
    # run_type = 'neuron_clustering'
    # run_type = 'mutant_clustering'
    # run_type = 'random_mutation_selection'
    run_type = 'boundary_sampling'

    # arch_type = 'all'
    arch_type = 'one_by_one'

    # ========================================================================================================
    if run_type == 'vanilla': # normal mutation analysis
        dmaacc_run = DMAACC()
        dmaacc_run.set_mutation_percent(0.1)
        dmaacc_run.set_mutator_list(['CW', 'NAI', 'NEB'])
        model_list = [['fcnn-mnist.keras', 'lenet5-mnist.keras'],
                      ['fcnn-fmnist.keras', 'lenet5-fmnist.keras'],
                      ['fcnn-kmnist.keras', 'lenet5-kmnist.keras'],
                      ['fcnn-emnist.keras', 'lenet5-emnist.keras']]
        dataset_list = ['mnist', 'fmnist', 'kmnist', 'emnist']

        for ds, model_l in zip(dataset_list, model_list):
            d = Dataset(ds)
            dmaacc_run.set_dataset(d)
            for model_n in model_l:
                dmaacc_run.load_model('examples/' + ds + '/' + model_n)
                model_current = dmaacc_run.get_model()
                for i in range(30):
                    dmaacc_run.set_mutation_level('neuron')
                    if arch_type == 'one_by_one':
                        df_clusters = dmaacc_run.run_vanilla()
                        csvfile = 'exp_vanilla.csv'
                        if os.path.isfile(csvfile):
                            df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
                        else:
                            df_clusters.to_csv(csvfile, mode='w', header=True, index=False)

    # ========================================================================================================
    elif run_type == 'neuron_clustering':
        dmaacc_run = DMAACC()
        dmaacc_run.set_mutation_percent(0.1)
        dmaacc_run.set_mutator_list(['CW', 'NAI', 'NEB'])
        n_list = [2, 4, 6, 8, 10]

        model_list = [['fcnn-mnist.keras', 'lenet5-mnist.keras'],
                      ['fcnn-fmnist.keras', 'lenet5-fmnist.keras'],
                      ['fcnn-kmnist.keras', 'lenet5-kmnist.keras'],
                      ['fcnn-emnist.keras', 'lenet5-emnist.keras']]
        dataset_list = ['mnist', 'fmnist', 'kmnist', 'emnist']

        for ds, model_l in zip(dataset_list, model_list):
            d = Dataset(ds)
            dmaacc_run.set_dataset(d)
            for model_n in model_l:
                dmaacc_run.load_model('examples/' + ds + '/' + model_n)
                model_current = dmaacc_run.get_model()
                for num in n_list:
                    dmaacc_run.set_mutation_level('cluster')
                    for i in range(6):
                        dmaacc_run.set_cluster_size(num)
                        if arch_type == 'one_by_one':
                            df_clusters = dmaacc_run.run_neuron_clustering()
                            csvfile = 'exp_neuron_clustering.csv'
                            if os.path.isfile(csvfile):
                                df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
                            else:
                                df_clusters.to_csv(csvfile, mode='w', header=True, index=False)

    # ========================================================================================================
    elif run_type == 'mutant_clustering':
        dmaacc_run = DMAACC()
        dmaacc_run.set_mutation_level("cluster")
        dmaacc_run.set_mutator_list(['CW', 'NAI', 'NEB'])
        PH_thresholds = [n / 100 for n in range(30, 75, 5)]
        model_list = [['fcnn-mnist.keras', 'lenet5-mnist.keras'],
                      ['fcnn-fmnist.keras', 'lenet5-fmnist.keras'],
                      ['fcnn-kmnist.keras', 'lenet5-kmnist.keras'],
                      ['fcnn-emnist.keras', 'lenet5-emnist.keras']]
        dataset_list = ['mnist', 'fmnist', 'kmnist', 'emnist']

        for i in range(6):
            for ds, model_l in zip(dataset_list, model_list):
                d = Dataset(ds)
                dmaacc_run.set_dataset(d)
                for model_n in model_l:
                    dmaacc_run.load_model('examples/' + ds + '/' + model_n)
                    model_current = dmaacc_run.get_model()
                    for threshold in PH_thresholds:
                        dmaacc_run.set_ParHAC_threshold(threshold)
                        dmaacc_run.set_mutation_level(['cluster'])  # , 'neuron'])
                        if arch_type == 'one_by_one':
                            df_clusters = dmaacc_run.run_mutant_clustering()
                            csvfile = 'exp_mutant_clustering.csv'
                            if os.path.isfile(csvfile):
                                df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
                            else:
                                df_clusters.to_csv(csvfile, mode='w', header=True, index=False)

    # ========================================================================================================
    elif run_type == 'random_mutation_selection':
        dmaacc_run = DMAACC()
        dmaacc_run.set_mutation_level("neuron")
        dmaacc_run.set_mutator_list(['CW', 'NAI', 'NEB'])
        selection_fractions = [0.25, 0.5, 0.75]
        model_list = [['fcnn-mnist.keras', 'lenet5-mnist.keras'],
                      ['fcnn-fmnist.keras', 'lenet5-fmnist.keras'],
                      ['fcnn-kmnist.keras', 'lenet5-kmnist.keras'],
                      ['fcnn-emnist.keras', 'lenet5-emnist.keras']]
        dataset_list = ['mnist', 'fmnist', 'kmnist', 'emnist']

        for i in range(6):
            for ds, model_l in zip(dataset_list, model_list):
                d = Dataset(ds)
                dmaacc_run.set_dataset(d)
                for model_n in model_l:
                    dmaacc_run.load_model('examples/' + ds + '/' + model_n)
                    model_current = dmaacc_run.get_model()
                    for fraction in selection_fractions:
                        dmaacc_run.set_selection_fraction(fraction)
                        if arch_type == 'one_by_one':
                            df_clusters = dmaacc_run.run_vanilla() # what changed is that we change the selection fraction
                            csvfile = 'exp_random_mutation_selection.csv'
                            if os.path.isfile(csvfile):
                                df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
                            else:
                                df_clusters.to_csv(csvfile, mode='w', header=True, index=False)

    # ========================================================================================================
    elif run_type == 'boundary_sampling':
        dmaacc_run = DMAACC()
        dmaacc_run.set_mutation_level("neuron")
        dmaacc_run.set_mutator_list(['CW', 'NAI', 'NEB'])
        boundary_sampling_thresholds = [1.5, 2, 5, 8, 10, 12, 15, 20, 100, 1000, 10000]
        model_list = [['fcnn-mnist.keras', 'lenet5-mnist.keras',
                       'alexnet2-mnist.keras', 'vggnet162-mnist.keras'],
                      ['fcnn-fmnist.keras', 'lenet5-fmnist.keras',
                       'alexnet2-fmnist.keras', 'vggnet162-fmnist.keras'],
                      ['fcnn-kmnist.keras', 'lenet5-kmnist.keras',
                       'alexnet2-kmnist.keras', 'vggnet162-kmnist.keras'],
                      ['fcnn-emnist.keras', 'lenet5-emnist.keras',
                       'alexnet2-emnist.keras', 'vggnet162-emnist.keras'],
                      ['alexnet2-cifar10.keras', 'resnet102-cifar10.keras',
                       'vggnet162-cifar10.keras', 'mobilenetv22-cifar10.keras'],
                      ['alexnet2-svhn.keras', 'resnet102-svhn.keras',
                       'vggnet162-svhn.keras', 'mobilenetv22-svhn.keras'],
                      ['alexnet2-caltech-101.keras', 'resnet102-caltech-101.keras',
                       'vggnet162-caltech-101.keras', 'mobilenetv22-caltech-101.keras']]
        dataset_list = ['mnist', 'fmnist', 'kmnist', 'emnist', 'cifar10', 'svhn', 'caltech-101']

        for i in range(6):
            for ds, model_l in zip(dataset_list, model_list):
                d = Dataset(ds)
                dmaacc_run.set_dataset(d)
                for model_n in model_l:
                    dmaacc_run.load_model('examples/' + ds + '/' + model_n)
                    model_current = dmaacc_run.get_model()
                    for threshold in boundary_sampling_thresholds:
                        dmaacc_run.set_boundary_threshold(threshold)
                        if arch_type == 'one_by_one':
                            df_clusters = dmaacc_run.run_boundary_sampling()
                            csvfile = 'exp_boundary_sampling.csv'
                            if os.path.isfile(csvfile):
                                df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
                            else:
                                df_clusters.to_csv(csvfile, mode='w', header=True, index=False)




    # ========================================================================================================
    # ========================================================================================================
    elif run_type == 'once':
        parser = ArgumentParser()
        parser.add_argument('-m',
                            '--model',
                            dest='model_filename',
                            help='Model file in .keras format',
                            required=True)
        parser.add_argument('-ml',
                            '--mutation_level',
                            dest='mutation_level',
                            help='Potential values are \'neuron\' or \'cluster\'',
                            required=True)
        parser.add_argument('-mp',
                            '--mutation-percent',
                            dest='mutation_percent',
                            default='0.1',
                            help='The fraction to add/subtract when mutating the weights (default: 0.1)',
                            required=False)
        parser.add_argument('-c',
                            '--cluster-size',
                            dest='cluster_size',
                            default='5',
                            help='Number of neurons per cluster. Each layer\'s neurons are partioned by this number. Ignored if one-unit-per-cluster activated.',
                            required=False)
        parser.add_argument('-u',
                            '--one-unit-per-cluster',
                            dest='one_unit_per_cluster',
                            default='False',
                            help='One unit per cluster (default: False)',
                            required=False)
        args = parser.parse_args()

        dmaacc_run = DMAACC()

        dmaacc_run.load_model(args.model_filename)
        dmaacc_run.set_mutation_level(args.mutation_level)
        dmaacc_run.set_mutation_percent(float(args.mutation_percent))
        dmaacc_run.set_cluster_size(int(args.cluster_size))
        dmaacc_run.set_one_unit_per_cluster(args.one_unit_per_cluster.lower() == 'true')
        df1 = dmaacc_run.run_approach_1()
        df2, df3 = dmaacc_run.run_approach_2()

    # if arch_type == 'all':
    #     df_clusters = dmaacc_run.run_vanilla()
    #     csvfile = 'vanilla_experiments.csv'
    #     if os.path.isfile(csvfile):
    #         df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
    #     else:
    #         df_clusters.to_csv(csvfile, mode='w', header=True, index=False)
    # el

    # if arch_type == 'all':
    #     df_clusters = dmaacc_run.run_approach_1()
    #     csvfile = 'experiments_approach1.csv'
    #     if os.path.isfile(csvfile):
    #         df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
    #     else:
    #         df_clusters.to_csv(csvfile, mode='w', header=True, index=False)
    # el

    # if arch_type == 'all':
    #     df_clusters = dmaacc_run.run_approach_2()
    #     csvfile = 'experiments_approach2.csv'
    #     if os.path.isfile(csvfile):
    #         df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
    #     else:
    #         df_clusters.to_csv(csvfile, mode='w', header=True, index=False)
    # el
