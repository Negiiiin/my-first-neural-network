import math
import random
import functools
import numpy as np
from utility import alphabetize, abs_mean
import matplotlib.pyplot as plt
import pandas as pd
import copy

def isclose(a, b, rel_tol=0.00000007, abs_tol=0.0):
    return abs(a-b) <= rel_tol

def findInput(inputs, elem, name):
    for e in inputs:
        if(e.get_name() == 'i0' and ('w' + name) == elem.get_name()):
            return e
        elif(e.get_name()[0] == 'i' and ('w' + e.get_name()[1:] + name) == elem.get_name()):
            return e
        elif(('w' + e.get_name() + name) == elem.get_name()):
            return e

def findNorm(weights):
    result = 0
    for w in weights:
        if('w' in w.get_name):
            result += ((w.get_value())**2)
    return result


class ValuedElement(object):
    def __init__(self,name,val):
        self.my_name = name
        self.my_value = val

    def set_value(self,val):
        self.my_value = val

    def get_value(self):
        return self.my_value

    def get_name(self):
        return self.my_name

    def __repr__(self):
        return "%s(%1.2f)" %(self.my_name, self.my_value)

class DifferentiableElement(object):
    def output(self):
        raise NotImplementedError("This is an abstract method")

    def dOutdX(self, elem):
        raise NotImplementedError("This is an abstract method")

    def clear_cache(self):
        pass

class Input(ValuedElement,DifferentiableElement):
    def __init__(self,name,val):
        ValuedElement.__init__(self,name,val)
        DifferentiableElement.__init__(self)

    def output(self):
        return self.my_value
        raise NotImplementedError("Implement me!")

    def dOutdX(self, elem):
        return 0.0
        raise NotImplementedError("Implement me!")

class Weight(ValuedElement):
    def __init__(self,name,val):
        ValuedElement.__init__(self,name,val)
        self.next_value = None

    def set_next_value(self,val):
        self.next_value = val

    def update(self):
        self.my_value = self.next_value


class Neuron(DifferentiableElement):
    def __init__(self, name, inputs, input_weights, use_cache=True):
        assert len(inputs)==len(input_weights)
        for i in range(len(inputs)):
            assert isinstance(inputs[i],(Neuron,Input))
            assert isinstance(input_weights[i],Weight)
        DifferentiableElement.__init__(self)
        self.my_name = name
        self.my_inputs = inputs
        self.my_weights = input_weights
        self.use_cache = use_cache
        self.clear_cache()
        self.my_descendant_weights = None
        self.my_direct_weights = None

    def get_descendant_weights(self):
        if self.my_descendant_weights is None:
            self.my_descendant_weights = {}
            inputs = self.get_inputs()
            weights = self.get_weights()
            for i in range(len(weights)):
                weight = weights[i]
                weight_name = weight.get_name()
                self.my_descendant_weights[weight_name] = set()
                input = inputs[i]
                if not isinstance(input, Input):
                    descendants = input.get_descendant_weights()
                    for name, s in descendants.items():
                        st = self.my_descendant_weights[weight_name]
                        st = st.union(s)
                        st.add(name)
                        self.my_descendant_weights[weight_name] = st

        return self.my_descendant_weights

    def isa_descendant_weight_of(self, target, weight):
        weights = self.get_descendant_weights()
        if weight.get_name() in weights:
            return target.get_name() in weights[weight.get_name()]
        else:
            raise Exception("weight %s is not connect to this node: %s"
                            %(weight, self))

    def has_weight(self, weight):
        return weight.get_name() in self.get_descendant_weights()

    def get_weight_nodes(self):
        return self.my_weights

    def clear_cache(self):
        self.my_output = None
        self.my_doutdx = {}

    def output(self):
        if self.use_cache:
            if self.my_output is None:
                self.my_output = self.compute_output()
            return self.my_output
        return self.compute_output()

    def compute_output(self):
        net = 0.0
        inputs = self.get_inputs()
        weights = self.get_weights()
        for i in range(0, len(inputs)):
            inp = inputs[i]
            wei = weights[i]
            net += (wei.get_value()*inp.output())
        return 1.0 / (1.0 + math.exp(-net))
        raise NotImplementedError("Implement me!")

    def dOutdX(self, elem):
        if self.use_cache:
            if elem not in self.my_doutdx:
                self.my_doutdx[elem] = self.compute_doutdx(elem)
            return self.my_doutdx[elem]
        return self.compute_doutdx(elem)

    def compute_doutdx(self, elem):
        weights = self.my_weights
        inputs = self.my_inputs
        output = 0.0
        if(self.has_weight(elem)):
            e = findInput(inputs, elem, self.my_name)
            output = (e.output() * self.output() * (1 - self.output()))
        else:
            output = self.output() * (1 - self.output())
            toMult = 0.0
            for w in weights:
                if(self.isa_descendant_weight_of(elem, w)):
                    e = findInput(inputs, w, self.my_name)
                    toMult += (e.dOutdX(elem) * w.get_value())
                else:
                    toMult += 0.0
            output *= toMult
        return output
        raise NotImplementedError("Implement me!")

    def get_weights(self):
        return self.my_weights

    def get_inputs(self):
        return self.my_inputs

    def get_name(self):
        return self.my_name

    def __repr__(self):
        return "Neuron(%s)" %(self.my_name)

class PerformanceElem(DifferentiableElement):
    def __init__(self,input,desired_value):
        assert isinstance(input,(Input,Neuron))
        DifferentiableElement.__init__(self)
        self.my_input = input
        self.my_desired_val = desired_value

    def output(self):
        return -1*((1/2) * ((self.my_desired_val - self.my_input.output())**2))
        raise NotImplementedError("Implement me!")

    def dOutdX(self, elem):
        return (self.my_desired_val - self.my_input.output()) * (self.my_input.dOutdX(elem))
        raise NotImplementedError("Implement me!")

    def set_desired(self,new_desired):
        self.my_desired_val = new_desired

    def get_input(self):
        return self.my_input


class Network(object):
    def __init__(self,performance_node,neurons):
        self.inputs =  []
        self.weights = []
        self.performance = performance_node
        self.output = performance_node.get_input()
        self.neurons = neurons[:]
        self.neurons.sort(key=functools.cmp_to_key(alphabetize))
        for neuron in self.neurons:
            self.weights.extend(neuron.get_weights())
            for i in neuron.get_inputs():
                if isinstance(i,Input) and not ('i0' in i.get_name()) and not i in self.inputs:
                    self.inputs.append(i)
        self.weights.reverse()
        self.weights = []
        for n in self.neurons:
            self.weights += n.get_weight_nodes()

    @classmethod
    def from_layers(self,performance_node,layers):
        neurons = []
        for layer in layers:
            if layer.get_name() != 'l0':
                neurons.extend(layer.get_elements())
        return Network(performance_node, neurons)

    def clear_cache(self):
        for n in self.neurons:
            n.clear_cache()

def seed_random():
    """Seed the random number generator so that random
    numbers are deterministically 'random'"""
    random.seed(200)
    np.random.seed(200)

def random_weight():
    return random.randrange(-1, 2)

def make_neural_net_basic():
    
    i0 = Input('i0', -1.0)
    i1 = Input('i1', 0.0)
    i2 = Input('i2', 0.0)

    w1A = Weight('w1A', 1)
    w2A = Weight('w2A', 1)
    wA  = Weight('wA', 1)

    A = Neuron('A', [i1,i2,i0], [w1A,w2A,wA])
    P = PerformanceElem(A, 0.0)

    net = Network(P,[A])
    return net

def make_neural_net_two_layer():
    seed_random()
    #Neuron A
    w_A = random_weight()
    i0_A = Input('i0', -1.0)
    wA  = Weight('wA', w_A)
    w_1A = random_weight()
    i1 = Input('i1', 0.0)
    w1A = Weight('w1A', w_1A)
    w_2A = random_weight()
    i2 = Input('i2', 0.0)
    w2A = Weight('w2A', w_2A)
    A = Neuron('A', [i1,i2,i0_A], [w1A,w2A,wA])

    #Neuron B
    w_B = random_weight()
    i0_B = Input('i0', -1.0)
    wB  = Weight('wB', w_B)
    w_1B = random_weight()
    w1B = Weight('w1B', w_1B)
    w_2B = random_weight()
    w2B = Weight('w2B', w_2B)
    B = Neuron('B', [i1,i2,i0_B], [w1B,w2B,wB])

    #Neuron C
    w_C = random_weight()
    i0_C = Input('i0', -1.0)
    wC  = Weight('wC', w_C)
    w_AC = random_weight()
    wAC  = Weight('wAC', w_AC)
    w_BC = random_weight()
    wBC  = Weight('wBC', w_BC)
    C = Neuron('C', [A, B, i0_C], [wAC, wBC, wC])

    #Performance Elem
    P = PerformanceElem(C, 0.0)
    net = Network(P,[A, B, C])
    return net
   

def make_neural_net_two_moons():
    #Inputs
    i1 = Input('i1', 0.0)
    i2 = Input('i2', 0.0)
    # Network Not Available
    return

class RegularizedPerformanceElem(PerformanceElem):
    def output(self):
        return -1*((1/2) * ((self.my_desired_val - self.my_input.output())**2)) - (0.00035*self.my_norm)
        raise NotImplementedError("Implement me!")

    def dOutdX(self, elem):
        if('w' in elem.get_name()):
            return (self.my_desired_val - self.my_input.output()) * (self.my_input.dOutdX(elem)) - (0.00035*2*elem.get_value())
        else:
            return (self.my_desired_val - self.my_input.output()) * (self.my_input.dOutdX(elem))
        raise NotImplementedError("Implement me!")
        
    def set_norm(self, norm):
        self.my_norm = norm


def train(network,
          data,
          rate=1.0,
          target_abs_mean_performance=0.0001,
          max_iterations=10000,
          verbose=False):
          
    iteration = 0
    while iteration < max_iterations:
        fully_trained = False
        performances = []
        for datum in data:
            for i in range(len(network.inputs)):
                network.inputs[i].set_value(datum[i])

            network.performance.set_desired(datum[-1])

            network.clear_cache()
            
            '''
            allWeights = network.weights
            norm = findNorm(allWeights)
            network.performance.set_norm(norm)
            #'''

            for w in network.weights:
                w.set_next_value(w.get_value() +
                                 rate * network.performance.dOutdX(w))
            for w in network.weights:
                w.update()
            performances.append(network.performance.output())
            network.clear_cache()
        abs_mean_performance = abs_mean(performances)

        if abs_mean_performance < target_abs_mean_performance:
            if verbose:
                print("iter %d: training complete.\n"\
                      "mean-abs-performance threshold %s reached (%1.6f)"\
                      %(iteration,
                        target_abs_mean_performance,
                        abs_mean_performance))
            break

        iteration += 1
    plot_decision_boundary(network, -1, 1, -1, 1)


def test(network, data, verbose=False):
    correct = 0
    X = []
    Y = []
    for datum in data:

        for i in range(len(network.inputs)):
            network.inputs[i].set_value(datum[i])
            if(i == 1):
                X.append(datum[i])
            else:
                Y.append(datum[i])
        
        network.clear_cache()
        result = network.output.output()
        if(result < 0.5):
            X.pop()
            Y.pop()
        prediction = round(result)

        network.clear_cache()

        if prediction == datum[-1]:
            correct+=1
            if verbose:
                print("test(%s) returned: %s => %s [%s]" %(str(datum),
                                                           str(result),
                                                           datum[-1],
                                                           "correct"))
        else:
            if verbose:
                print("test(%s) returned: %s => %s [%s]" %(str(datum),
                                                           str(result),
                                                           datum[-1],
                                                           "wrong"))
    return float(correct)/len(data)

def plot_decision_boundary(network, xmin, xmax, ymin, ymax):
    data = []
    inData = []
    i = xmin
    j = ymin
    soton = []
    X = []
    Y = []
    for i in range (0, 1000):
        radif = []
        radif.append(random.uniform(xmin,xmax))
        radif.append(random.uniform(ymin,ymax))
        soton.append(radif)

    for datum in soton:
        for i in range(len(network.inputs)):
            network.inputs[i].set_value(datum[i])
            if(i == 0):
                X.append(datum[i])
            else:
                Y.append(datum[i])
        
        network.clear_cache()
        result = network.output.output()
        if(result < 0.5):
            X.pop()
            Y.pop()
    colors = np.random.rand(len(X))
    plt.scatter(X, Y, s = 10, c = colors, alpha = 0.7)
    X = []
    Y = []
    X.append(xmin)
    X.append(xmax)
    Y.append(ymin)
    Y.append(ymax)
    plt.scatter(X, Y, s = 10, c = 'white', alpha = 0.7)
    plt.show()

def finiteDifference(netwo):
    Fx = 0
    li = []
    Fx = netwo.output.output()
    for w in netwo.weights:
        value = copy.deepcopy(w.get_value())
        w.set_value(value+(10**(-8)))
        newOutPut = netwo.output.output()
        li.append((newOutPut - Fx)/(10**(-8)))
        w.set_value(value)
        netwo.clear_cache()
    netwo.clear_cache()
    li2 = []
    for w in netwo.weights:
        li2.append(netwo.output.dOutdX(w))
