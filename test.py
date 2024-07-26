from Mutant import Mutant
from keras.models import load_model
import utils
import numpy as np

model = load_model('examples/mnist/fcnn-mnist.h5')
mu = utils.ModelUtils()
mutated_model = mu.model_copy(model,'test')
model.summary()
weights1 = mutated_model.layers[1].get_weights()

w = weights1[0]
print(type(w.shape))

temp = np.array(weights1[0][:, 3])
i = temp.shape
print(i)
tempb = weights1[1][3]
tempbb = float(weights1[1][3])
weights1[0][:, 3] *= 1.1
weights1[1][3] *= 1.1
mutated_model.layers[1].set_weights(weights1)
list_of_mutants = []
list_of_mutants.append(Mutant(0, mutated_model, mutation_type='C', layer=1, neuron=3))
test1 = list_of_mutants[0].get_model().get_weights()[0]

weights1[0][:, 3] = temp
test2 = mutated_model.layers[1].get_weights()[0][:,3]

temp = weights1[0][:, 5]
weights1[0][:, 5] *= 0.9
list_of_mutants.append(Mutant(1, mutated_model, mutation_type='C', layer=1, neuron=5))
test3 = list_of_mutants[0].get_model().get_weights()[0]

weights1[0][:,5] = temp
test4 = mutated_model.layers[1].get_weights()[0][:,5]

