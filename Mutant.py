class Mutant:
    def __init__(self, uid, mutant, mutation_type, layer, neuron):
        self._uid = uid
        self._mutant = mutant
        self._mutation_type = mutation_type
        self._layer = layer
        self._neuron = neuron

    def get_uid(self):
        return self._uid
    def get_model(self):
        return self._mutant
    def get_mutation_type(self):
        return self._mutation_type
    def get_layer(self):
        return self._layer
    def get_neuron(self):
        return self._neuron
