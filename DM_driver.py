from argparse import ArgumentParser
from DMAACC import DMAACC
from network import Dataset
from network import Network
# Model, Mutate, Test
# Model, Cluster, Mutate, Test
import csv
from keras.callbacks import EarlyStopping
if __name__ == "__main__":
    run_type = 'once'

    if run_type == 'once':
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


        dmaacc_run.load_model("examples/mnist/"+args.model_filename)
        dmaacc_run.set_mutation_level(args.mutation_level)
        dmaacc_run.set_mutation_percent(float(args.mutation_percent))
        dmaacc_run.set_cluster_size(int(args.cluster_size))
        dmaacc_run.set_one_unit_per_cluster(args.one_unit_per_cluster.lower() == 'true')
        dmaacc_run.run_approach_2()


    #------------------------------------------------------------------------------------------------------
    elif run_type == 'experiment_approach1':
        dmaacc_run = DMAACC()
        # dmaacc_run.set_one_unit_per_cluster(args.one_unit_per_cluster.lower() == 'true')
        dmaacc_run.set_mutation_percent(0.1)
        dmaacc_run.set_mutator_list(['CW', 'NAI', 'NEB'])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
        dataset_list = ['mnist', 'kmnist', 'fmnist', 'emnist', 'cifar10', 'cifar100']
        model_list = [['fcnn-mnist.h5', 'lenet5-mnist.h5', 'resnet18-mnist.h5', 'alexnet-mnist.h5', 'vggnet16-mnist.h5'],
                      ['fcnn-kmnist.h5', 'lenet5-kmnist.h5', 'resnet18-kmnist.h5', 'alexnet-kmnist.h5', 'vggnet16-kmnist.h5'],
                      ['fcnn-fmnist.h5', 'lenet5-fmnist.h5', 'resnet18-fmnist.h5', 'alexnet-fmnist.h5', 'vggnet16-fmnist.h5'],
                      ['fcnn-emnist.h5', 'lenet5-emnist.h5', 'resnet18-emnist.h5', 'alexnet-emnist.h5', 'vggnet16-emnist.h5'],
                      ['resnet18-cifar10.h5' 'vggnet16-cifar10.h5'],
                      ['resnet18-cifar100.h5' 'vggnet16-cifar100.h5']]
        n_list = [2, 4, 6, 8, 10]
        fields = ['model_name', 'clusters_per_layer', 'mutation_level', 'mutation_percent', 'start_time', 'end_time',
                  'diff', 'mutation_score', 'original_model_acc']
        with open('experiments_approach1.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)


            for ds, model_l in zip(dataset_list, model_list):
                experiment_data = []
                d = Dataset(ds)
                dmaacc_run.set_dataset(d)
                for model_n in model_l:
                    dmaacc_run.load_model('examples/' + ds + '/' + model_n)
                    model_current = dmaacc_run.get_model()
                    for i in range(30):
                        h = model_current.fit(d.get_x_train(), d.get_y_train(), epochs=20,
                                          validation_data=(d.get_x_test(), d.get_y_test()),callbacks=[early_stopping])
                        acc = h.history['accuracy'][-1]
                        dmaacc_run.set_mutation_level('cluster')
                        for num in n_list:
                            dmaacc_run.set_clusters_per_layer(num)
                            start, end, mut_score = dmaacc_run.run()
                            experiment_data += [[model_n.split('.')[0], num, 'cluster', 0.1, start, end, end - start, mut_score, acc]]


                        dmaacc_run.set_mutation_level('neuron')
                        start, end, mut_score = dmaacc_run.run()
                        experiment_data += [[model_n.split('.')[0], 0, 'neuron', 0.1, start, end, end - start, mut_score, acc]]
                csvwriter.writerows(experiment_data)


            # print(experiment_data)
    elif run_type == 'experiment_approach2':

        model_list = [method for method in dir(Network) if ('_scratch' in method or '_keras' in method)]
        dataset_list = ['mnist', 'kmnist', 'fmnist', 'emnist', 'cifar10', 'cifar100']

        dmaacc_run = DMAACC()
        dmaacc_run.set_mutation_level("cluster")
        # dmaacc_run.set_one_unit_per_cluster(args.one_unit_per_cluster.lower() == 'true')
        # dmaacc_run.set_mutation_percent(0.1)
        dmaacc_run.set_mutator_list(['CW', 'NAI', 'NEB'])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
        dataset_list = ['mnist', 'kmnist', 'fmnist', 'emnist', 'cifar10', 'cifar100']
        model_list = [['fcnn-mnist.h5', 'lenet5-mnist.h5', 'resnet18-mnist.h5', 'alexnet-mnist.h5', 'vggnet16-mnist.h5'],
                      ['fcnn-kmnist.h5', 'lenet5-kmnist.h5', 'resnet18-kmnist.h5', 'alexnet-kmnist.h5', 'vggnet16-kmnist.h5'],
                      ['fcnn-fmnist.h5', 'lenet5-fmnist.h5', 'resnet18-fmnist.h5', 'alexnet-fmnist.h5', 'vggnet16-fmnist.h5'],
                      ['fcnn-emnist.h5', 'lenet5-emnist.h5', 'resnet18-emnist.h5', 'alexnet-emnist.h5', 'vggnet16-emnist.h5'],
                      ['resnet18-cifar10.h5', 'vggnet16-cifar10.h5'],
                      ['resnet18-cifar100.h5', 'vggnet16-cifar100.h5']]
        n_list = [2, 4, 6, 8, 10]
        fields = ['model_name', 'clusters_per_layer', 'mutation_level', 'mutation_percent', 'start_time', 'end_time',
                  'diff', 'mutation_score', 'original_model_acc']
        with open('experiments_approach2.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)


            for ds, model_l in zip(dataset_list, model_list):
                experiment_data = []
                d = Dataset(ds)
                dmaacc_run.set_dataset(d)
                for model_n in model_l:
                    dmaacc_run.load_model('examples/' + ds + '/' + model_n)
                    model_current = dmaacc_run.get_model()
                    for i in range(30):
                        h = model_current.fit(d.get_x_train(), d.get_y_train(), epochs=20,
                                          validation_data=(d.get_x_test(), d.get_y_test()),callbacks=[early_stopping])
                        acc = h.history['accuracy'][-1]
                        dmaacc_run.set_mutation_level('cluster')
                        for num in n_list:
                            dmaacc_run.set_clusters_per_layer(num)
                            start, end, mut_score = dmaacc_run.run()
                            experiment_data += [[model_n.split('.')[0], num, 'cluster', 0.1, start, end, end - start, mut_score, acc]]


                        dmaacc_run.set_mutation_level('neuron')
                        start, end, mut_score = dmaacc_run.run()
                        experiment_data += [[model_n.split('.')[0], 0, 'neuron', 0.1, start, end, end - start, mut_score, acc]]
                csvwriter.writerows(experiment_data)


            # print(experiment_data)