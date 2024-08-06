from network import Network
import scipy.io as sio
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import pandas as pd

df_clusters = pd.DataFrame(
            columns=['Model_Type', 'Dataset', 'Mutation_Level', 'Mutate_time',
                     'Number_of_Mutants', 'Mutation_Score', 'MS_time', 'Total_time'])

df_clusters.loc[len(df_clusters.index)] = ['fcnn', 'mnist', 'neuron', 25, 555, 0.647, 36, 61]
df_clusters.loc[len(df_clusters.index)] = ['lenet5', 'mnist', 'neuron', 25, 555, 0.647, 36, 61]
df_clusters.loc[len(df_clusters.index)] = ['resnet', 'mnist', 'neuron', 25, 555, 0.647, 36, 61]
df_clusters.loc[len(df_clusters.index)] = ['rnn', 'mnist', 'neuron', 25, 555, 0.647, 36, 61]

# df_clusters = df_clusters.add(pd.DataFrame())
df_clusters.to_csv('test.csv', mode='w', header=True, index=True)
df_clusters.to_csv('test2.csv', mode='w', header=True, index=False)


print([n/2 for n in range(5, 16)])

train_data = sio.loadmat('examples/svhn/data/train_32x32.mat')
test_data = sio.loadmat('examples/svhn/data/test_32x32.mat')
x_train = train_data['X']
y_train = train_data['y']
x_test = test_data['X']
y_test = test_data['y']

nb_classes = 27
data = sio.loadmat('examples/emnist/data/emnist-letters')['dataset']
x_train = data['train'][0, 0]['images'][0, 0]
y_train = data['train'][0, 0]['labels'][0, 0]
x_test = data['test'][0, 0]['images'][0, 0]
y_test = data['test'][0, 0]['labels'][0, 0]
x_train = x_train.reshape(124800,28,28)
x_test = x_test.reshape(20800,28,28)
x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train)[..., None])
x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test)[..., None])
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
x_train, x_test = x_train / 255., x_test / 255.
y_train, y_test = to_categorical(y_train), to_categorical(y_test)




n = Network()

methods_list = [method for method in dir(Network) if callable(
    getattr(Network, method)) and not method.startswith("__") and ('_scratch' in method or '_keras' in method)]

print("Methods using dir():", methods_list)


getattr(Network, method_name)()

r = getattr(Network, method_name)
r()


from Mutant import Mutant
from keras.models import load_model
import utils
import numpy as np
#
# model = load_model('examples/mnist/fcnn-mnist.h5')
# mu = utils.ModelUtils()
# mutated_model = mu.model_copy(model,'test')
# model.summary()
# weights1 = mutated_model.layers[1].get_weights()
#
# w = weights1[0]
# print(type(w.shape))
#
# temp = np.array(weights1[0][:, 3])
# i = temp.shape
# print(i)
# tempb = weights1[1][3]
# tempbb = float(weights1[1][3])
# weights1[0][:, 3] *= 1.1
# weights1[1][3] *= 1.1
# mutated_model.layers[1].set_weights(weights1)
# list_of_mutants = []
# list_of_mutants.append(Mutant(0, mutated_model, mutation_type='C', layer=1, neuron=3))
# test1 = list_of_mutants[0].get_model().get_weights()[0]
#
# weights1[0][:, 3] = temp
# test2 = mutated_model.layers[1].get_weights()[0][:,3]
#
# temp = weights1[0][:, 5]
# weights1[0][:, 5] *= 0.9
# list_of_mutants.append(Mutant(1, mutated_model, mutation_type='C', layer=1, neuron=5))
# test3 = list_of_mutants[0].get_model().get_weights()[0]
#
# weights1[0][:,5] = temp
# test4 = mutated_model.layers[1].get_weights()[0][:,5]

# import ctypes
#
# #libquickstart = ctypes.CDLL('libquickstart.so') /home/lauren-lyons/Documents/Auburn/DeepMAACC/libquickstart.so
# libquickstart = ctypes.CDLL('/usr/local/lib/libquickstart.so')
#
# do_cluster = libquickstart.do_clustering
# do_cluster.argtypes = []
#
# graph = libquickstart.SimpleUndirectedGraph()
# #graph.AddNode()
# graph.AddEdge(0,1,0.5)  # (from, to, weight)
# graph.AddEdge(1,2,0.2)
# clusterer = libquickstart.ParHacClusterer(graph)
# config = libquickstart.ClustererConfig()
# #config.mutable_parhac_clusterer_config()->set_weight_threshold(0.3);
# config.set_weight_threshold(0.3)
# clusters = clusterer.Cluster(config)
#
#
# for clust in clusters:
#     print(clust)
