# from network import Network, Dataset
import scipy.io as sio
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import pandas as pd


# d = Dataset('mnist')
xtrain = np.load('examples/mnist/data/mnist_train_inputs.npy')
ytrain = np.load('examples/mnist/data/mnist_train_outputs.npy')
xtest = np.load('examples/mnist/data/mnist_test_inputs.npy')
ytest = np.load('examples/mnist/data/mnist_test_outputs.npy')

print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

if xtrain[0].shape != (224, 224, 3):
    print(xtrain[0].shape)
    print(np.squeeze(xtrain[0]).shape)
    print(np.squeeze(xtrain).shape)
    print(np.squeeze(np.squeeze(xtrain)).shape)
    # print('padded_arr')
    # padded_arr = np.pad(np.squeeze(xtrain), ((0, 0), (96, 96), (96, 96), (0, 0)), mode='constant')
    # print('padded_arr')
    # print(padded_arr.shape)

if xtrain[0].shape != (224, 224, 3):
    amt = int((34 - xtrain[0].shape[0]) / 2)
    x_train = np.pad(np.squeeze(xtrain), ((0, 0), (amt, amt), (amt, amt), (0, 0)), mode='constant')
    x_test = np.pad(np.squeeze(xtest), ((0, 0), (amt, amt), (amt, amt), (0, 0)), mode='constant')

print('end')
print(x_train.shape)
print(x_test.shape)

xtrain = np.load('examples/caltech-101/data/caltech-101_train_inputs.npy')
ytrain = np.load('examples/caltech-101/data/caltech-101_train_outputs.npy')
xtest = np.load('examples/caltech-101/data/caltech-101_test_inputs.npy')
ytest = np.load('examples/caltech-101/data/caltech-101_test_outputs.npy')

print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

count = [0] * 101
for i in ytrain:
    










#
# t = (3, 4) + tuple(np.array([3,5,6,5,1,4,8,8,9,5,1,4,5]).flatten(), ) + tuple(np.array([3,5,6,5,1,4,8,8,9,5,1,4,5]).flatten(), )
# print(t)
#
#
# df_clusters = pd.DataFrame(
#             columns=['Model_Type', 'Dataset', 'Mutation_Level', 'Mutate_time',
#                      'Number_of_Mutants', 'Mutation_Score', 'MS_time', 'Total_time'])
#
# df_clusters.loc[len(df_clusters.index)] = ['fcnn', 'mnist', 'neuron', 25, 555, 0.647, 36, 61]
# df_clusters.loc[len(df_clusters.index)] = ['lenet5', 'mnist', 'neuron', 25, 555, 0.647, 36, 61]
# df_clusters.loc[len(df_clusters.index)] = ['resnet', 'mnist', 'neuron', 25, 555, 0.647, 36, 61]
# df_clusters.loc[len(df_clusters.index)] = ['rnn', 'mnist', 'neuron', 25, 555, 0.647, 36, 61]
#
# # df_clusters = df_clusters.add(pd.DataFrame())
# df_clusters.to_csv('test.csv', mode='w', header=True, index=True)
# df_clusters.to_csv('test2.csv', mode='w', header=True, index=False)
#
#
# print([n/2 for n in range(5, 16)])
#
# train_data = sio.loadmat('examples/svhn/data/train_32x32.mat')
# test_data = sio.loadmat('examples/svhn/data/test_32x32.mat')
# x_train = train_data['X']
# y_train = train_data['y']
# x_test = test_data['X']
# y_test = test_data['y']
#
# nb_classes = 27
# data = sio.loadmat('examples/emnist/data/emnist-letters')['dataset']
# x_train = data['train'][0, 0]['images'][0, 0]
# y_train = data['train'][0, 0]['labels'][0, 0]
# x_test = data['test'][0, 0]['images'][0, 0]
# y_test = data['test'][0, 0]['labels'][0, 0]
# x_train = x_train.reshape(124800,28,28)
# x_test = x_test.reshape(20800,28,28)
# x_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train)[..., None])
# x_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test)[..., None])
# x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
# x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
# x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
# x_train, x_test = x_train / 255., x_test / 255.
# y_train, y_test = to_categorical(y_train), to_categorical(y_test)
#
#
#
#
# n = Network()
#
# methods_list = [method for method in dir(Network) if callable(
#     getattr(Network, method)) and not method.startswith("__") and ('_scratch' in method or '_keras' in method)]
#
# print("Methods using dir():", methods_list)
#
#
# getattr(Network, method_name)()
#
# r = getattr(Network, method_name)
# r()
#
#
#
# [  [[[0, 12, 132, 144], [1, 135], [2, 147], [3, 129, 130], [4, 138, 139], [5, 136], [6, 16, 141, 142], [7, 149], [8, 124], [9, 148], [10, 145], [11, 140],
#    [13, 133], [14, 143], [15, 146], [17], [18, 114], [19, 126], [20], [21, 120], [22, 127], [23], [24, 121], [25, 123], [26], [27], [28], [29], [30], [31],
#    [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51, 52], [53], [54, 55], [56], [57, 58],
#    [59], [60, 61], [62], [63, 64], [65], [66], [67], [68], [69], [70], [71], [72, 73], [74], [75, 76], [77], [78, 79], [80], [81], [82], [83], [84], [85], [86],
#    [87], [88], [89], [90, 91], [92], [93, 94], [95], [96], [97], [98], [99], [100], [101], [102, 103], [104], [105], [106], [107], [108], [109], [110], [111], [112],
#    [113], [115], [116], [117], [118], [119], [122], [125], [128], [131], [134], [137]]],
#  [[[0, 146], [1, 144], [2, 148], [3, 149], [4, 134], [5, 147], [6, 140], [7, 138], [8, 135], [9, 141], [10, 139], [11, 142], [12], [13], [14, 143], [15],
#    [16], [17, 145], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40],
#    [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63], [64], [65], [66],
#    [67], [68], [69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79], [80], [81], [82], [83], [84], [85], [86], [87], [88], [89], [90], [91], [92],
#    [93], [94], [95], [96], [97], [98], [99], [100], [101], [102], [103], [104], [105], [106], [107], [108], [109], [110], [111], [112], [113], [114], [115], [116],
#    [117], [118], [119], [120], [121], [122], [123], [124], [125], [126], [127], [128], [129], [130], [131], [132], [133], [136], [137]]],
#  [[[0, 144], [1, 146], [2, 145], [3, 149], [4, 148], [5, 147], [6, 138], [7, 142], [8, 137], [9, 141], [10, 143], [11, 139], [12], [13], [14, 140],
#    [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39],
#    [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63], [64],
#    [65], [66], [67], [68], [69], [70], [71], [72], [73], [74], [75], [76], [77], [78], [79], [80], [81], [82], [83], [84], [85], [86], [87], [88], [89],
#    [90], [91], [92], [93], [94], [95], [96], [97], [98], [99], [100], [101], [102], [103], [104], [105], [106], [107], [108], [109], [110], [111], [112],
#    [113], [114], [115], [116], [117], [118], [119], [120], [121], [122], [123], [124], [125], [126], [127], [128], [129], [130], [131], [132], [133], [134], [135], [136]]]  ]
#
#
#
# [[[0, 6, 7, 138, 139, 148], [1, 149], [2, 132], [3, 4, 135, 142], [5, 126], [8, 140]
# , [9, 129], [10, 147], [11, 127], [12, 13, 16, 133, 144, 145], [14], [15, 136], [17]
# , [18, 137], [19, 114, 115], [20], [21, 22, 141], [23], [24, 123, 124], [25, 130], [
# 26], [27, 143], [28, 146], [29], [30], [31], [32], [33], [34], [35], [36, 120], [37]
# , [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51,
#  52], [53], [54, 55], [56], [57, 58], [59], [60, 61], [62], [63, 64], [65], [66], [6
# 7], [68], [69], [70], [71], [72, 73], [74], [75, 76], [77], [78, 79], [80], [81], [8
# 2], [83], [84], [85], [86], [87], [88], [89], [90, 91], [92], [93, 94], [95], [96],
# [97], [98], [99], [100], [101], [102, 103], [104], [105], [106], [107], [108], [109]
# , [110], [111], [112], [113], [116], [117], [118], [119], [121], [122], [125], [128]
# , [131], [134]], [[0, 144], [1, 146], [2, 147], [3, 134], [4, 149], [5, 148], [6, 13
# 9], [7, 141], [8, 135], [9, 142], [10, 143], [11, 140], [12], [13], [14, 145], [15],
#  [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29],
#  [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43],
#  [44], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56], [57],
#  [58], [59], [60], [61], [62], [63], [64], [65], [66], [67], [68], [69], [70], [71],
#  [72], [73], [74], [75], [76], [77], [78], [79], [80], [81], [82], [83], [84], [85],
#  [86], [87], [88], [89], [90], [91], [92], [93], [94], [95], [96], [97], [98], [99],
#  [100], [101], [102], [103], [104], [105], [106], [107], [108], [109], [110], [111],
#  [112], [113], [114], [115], [116], [117], [118], [119], [120], [121], [122], [123],
#  [124], [125], [126], [127], [128], [129], [130], [131], [132], [133], [136], [137],
#  [138]], [[0, 146], [1, 144], [2, 145], [3, 147], [4, 148], [5, 149], [6, 140], [7,
# 135], [8, 136], [9, 138], [10, 139], [11], [12, 141], [13, 142], [14, 143], [15], [1
# 6], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [3
# 0], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [4
# 4], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [5
# 8], [59], [60], [61], [62], [63], [64], [65], [66], [67], [68], [69], [70], [71], [7
# 2], [73], [74], [75], [76], [77], [78], [79], [80], [81], [82], [83], [84], [85], [8
# 6], [87], [88], [89], [90], [91], [92], [93], [94], [95], [96], [97], [98], [99], [1
# 00], [101], [102], [103], [104], [105], [106], [107], [108], [109], [110], [111], [1
# 12], [113], [114], [115], [116], [117], [118], [119], [120], [121], [122], [123], [124], [125], [126], [127], [128], [129], [130], [131], [132], [133], [134], [137]]]
#
#
#
# from Mutant import Mutant
# from keras.models import load_model
# import utils
# import numpy as np
# #
# # model = load_model('examples/mnist/fcnn-mnist.h5')
# # mu = utils.ModelUtils()
# # mutated_model = mu.model_copy(model,'test')
# # model.summary()
# # weights1 = mutated_model.layers[1].get_weights()
# #
# # w = weights1[0]
# # print(type(w.shape))
# #
# # temp = np.array(weights1[0][:, 3])
# # i = temp.shape
# # print(i)
# # tempb = weights1[1][3]
# # tempbb = float(weights1[1][3])
# # weights1[0][:, 3] *= 1.1
# # weights1[1][3] *= 1.1
# # mutated_model.layers[1].set_weights(weights1)
# # list_of_mutants = []
# # list_of_mutants.append(Mutant(0, mutated_model, mutation_type='C', layer=1, neuron=3))
# # test1 = list_of_mutants[0].get_model().get_weights()[0]
# #
# # weights1[0][:, 3] = temp
# # test2 = mutated_model.layers[1].get_weights()[0][:,3]
# #
# # temp = weights1[0][:, 5]
# # weights1[0][:, 5] *= 0.9
# # list_of_mutants.append(Mutant(1, mutated_model, mutation_type='C', layer=1, neuron=5))
# # test3 = list_of_mutants[0].get_model().get_weights()[0]
# #
# # weights1[0][:,5] = temp
# # test4 = mutated_model.layers[1].get_weights()[0][:,5]
#
# # import ctypes
# #
# # #libquickstart = ctypes.CDLL('libquickstart.so') /home/lauren-lyons/Documents/Auburn/DeepMAACC/libquickstart.so
# # libquickstart = ctypes.CDLL('/usr/local/lib/libquickstart.so')
# #
# # do_cluster = libquickstart.do_clustering
# # do_cluster.argtypes = []
# #
# # graph = libquickstart.SimpleUndirectedGraph()
# # #graph.AddNode()
# # graph.AddEdge(0,1,0.5)  # (from, to, weight)
# # graph.AddEdge(1,2,0.2)
# # clusterer = libquickstart.ParHacClusterer(graph)
# # config = libquickstart.ClustererConfig()
# # #config.mutable_parhac_clusterer_config()->set_weight_threshold(0.3);
# # config.set_weight_threshold(0.3)
# # clusters = clusterer.Cluster(config)
# #
# #
# # for clust in clusters:
# #     print(clust)
