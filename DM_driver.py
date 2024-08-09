import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from argparse import ArgumentParser
from network import Dataset
from network import Network
from DMAACC import DMAACC
import os.path

if __name__ == "__main__":
    run_type = 'vanilla'
    # run_type = 'approach1'
    # run_type = 'approach2'

    if run_type == 'once':
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

    # ========================================================================================================
    elif run_type == 'vanilla':
        dmaacc_run = DMAACC()
        dmaacc_run.set_mutation_percent(0.1)
        dmaacc_run.set_mutator_list(['CW', 'NAI', 'NEB'])
        model_list = [['rnn-imdb.keras'],
                      ['rnn-reuters.keras'],
                      ['resnet18-cifar10.keras'],
                      ['resnet18-cifar100.keras'],
                      ['resnet18-svhn.keras']]
        # ['fcnn-mnist.keras', 'lenet5-mnist.keras'],
        # ['fcnn-fmnist.keras', 'lenet5-fmnist.keras'],
        # ['fcnn-kmnist.keras', 'lenet5-kmnist.keras'],
        # ['fcnn-emnist.keras', 'lenet5-emnist.keras'],
        dataset_list = ['imdb', 'reuters', 'cifar10', 'cifar100', 'svhn']  # 'mnist', 'fmnist', 'kmnist', 'emnist',

        csvfile = 'vanilla_experiments.csv'
        for ds, model_l in zip(dataset_list, model_list):
            d = Dataset(ds)
            dmaacc_run.set_dataset(d)
            for model_n in model_l:
                dmaacc_run.load_model('examples/' + ds + '/' + model_n)
                model_current = dmaacc_run.get_model()
                for i in range(6):
                    dmaacc_run.set_mutation_level('neuron')
                    df_clusters = dmaacc_run.run_vanilla()
                    if os.path.isfile(csvfile):
                        df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
                    else:
                        df_clusters.to_csv(csvfile, mode='w', header=True, index=False)

    # ========================================================================================================
    elif run_type == 'approach1':
        dmaacc_run = DMAACC()
        # dmaacc_run.set_one_unit_per_cluster(args.one_unit_per_cluster.lower() == 'true')
        dmaacc_run.set_mutation_percent(0.1)
        dmaacc_run.set_mutator_list(['CW', 'NAI', 'NEB'])
        n_list = [2, 4, 6, 8, 10]

        #model_type_list = [method for method in dir(Network) if ('_scratch' in method or '_keras' in method)]
        model_list = [['rnn-imdb.keras'],
                      ['rnn-reuters.keras'],
                      ['resnet18-cifar10.keras'],
                      ['resnet18-cifar100.keras'],
                      ['resnet18-svhn.keras']]
        # ['fcnn-mnist.keras', 'lenet5-mnist.keras'],
        # ['fcnn-fmnist.keras', 'lenet5-fmnist.keras'],
        # ['fcnn-kmnist.keras', 'lenet5-kmnist.keras'],
        # ['fcnn-emnist.keras', 'lenet5-emnist.keras'],
        dataset_list = ['imdb', 'reuters', 'cifar10', 'cifar100', 'svhn']  # 'mnist', 'fmnist', 'kmnist', 'emnist',
        # model_list = []
        # for dataset in dataset_list:
        #     for model in model_type_list:
        #         if os.path.isfile('examples/' + dataset + '/' + model + '-' + dataset + '.keras'):
        #             model_list += model + '-' + dataset + '.keras'
        csvfile = 'experiments_approach1.csv'
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
                        df_clusters = dmaacc_run.run_approach_1()
                        if os.path.isfile('experiments_approach1.csv'):
                            df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
                        else:
                            df_clusters.to_csv(csvfile, mode='w', header=True, index=False)
                # for i in range(6):
                #     dmaacc_run.set_mutation_level('neuron')
                #     df_clusters = dmaacc_run.run_approach_1()
                #     if os.path.isfile('experiments_approach1_' + str(dmaacc_run.get_mutation_level) + '.csv'):
                #         df_clusters.to_csv(csvfile, mode='a', header=False, index=False)
                #     else:
                #         df_clusters.to_csv(csvfile, mode='w', header=True, index=False)

    # ========================================================================================================
    elif run_type == 'approach2':
        dmaacc_run = DMAACC()
        dmaacc_run.set_mutation_level("cluster")
        dmaacc_run.set_mutator_list(['CW', 'NAI', 'NEB'])
        PH_thresholds = [n / 2 for n in range(6, 15)]

        #model_type_list = [method for method in dir(Network) if ('_scratch' in method or '_keras' in method)]
        model_list = [['lenet5-mnist.keras'],
                      ['fcnn-fmnist.keras', 'lenet5-fmnist.keras'],
                      ['fcnn-kmnist.keras', 'lenet5-kmnist.keras'],
                      ['fcnn-emnist.keras', 'lenet5-emnist.keras']] #,
                      # ['resnet18-cifar10.keras'],
                      # ['resnet18-cifar100.keras'],
                      # ['resnet18-svhn.keras'],
                      # ['rnn-imdb.keras'],
                      # ['rnn-reuters.keras']] 'fcnn-mnist.keras',
        dataset_list = ['mnist', 'fmnist', 'kmnist', 'emnist']  # , 'cifar10', 'cifar100', 'svhn', 'imdb', 'reuters']
        # model_list = []
        # temp_list = []
        # for dataset in dataset_list:
        #     for model in model_type_list:
        #         if os.path.isfile('examples/'+dataset+'/'+model+'-'+dataset+'.keras'):
        #             temp_list += [model+'-'+dataset+'.keras']
        #     model_list += [temp_list]

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
                        df_vanilla, df_cluster = dmaacc_run.run_approach_2()
                        # if os.path.isfile('experiments_approach2_vanilla.csv'):
                        #     df_vanilla.to_csv('experiments_approach2_vanilla.csv', mode='a', header=False, index=False)
                        # else:
                        #     df_vanilla.to_csv('experiments_approach2_vanilla.csv', mode='w', header=True, index=False)
                        if os.path.isfile('experiments_approach2.csv'):
                            df_cluster.to_csv('experiments_approach2.csv', mode='a', header=False, index=False)
                        else:
                            df_cluster.to_csv('experiments_approach2.csv', mode='w', header=True, index=False)
