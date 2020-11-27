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
    weightNames = ["w1A", "w2A", "w1B", "w2B", "w1C", "w2C", "w1D", "w2D", "w1E", "w2E", "w1F", "w2F", "w1G", "w2G", "w1H", "w2H"
    , "w1I", "w2I", "w1J", "w2J", "w1K", "w2K", "w1L", "w2L", "w1M", "w2M", "w1N", "w2N", "w1O", "w2O", "w1P", "w2P", "w1Q"
    , "w2Q", "w1R", "w2R", "w1S", "w2S", "w1T", "w2T", "w1U", "w2AU", "w1V", "w2V", "w1W", "w2W", "w1X", "w2X", "w1Y", "w2Y"
    , "w1Z", "w2Z", "w1AA", "w2AA", "w1AB", "w2AB", "w1AC", "w2AC", "w1AD", "w2AD", "w1AE", "w2AE", "w1AF", "w2AF", "w1AG"
    , "w2AG", "w1AH", "w2AH", "w1AI", "w2AI", "w1AJ", "w2AJ", "w1AK", "w2AK", "w1AL", "w2AL", "w1AM", "w2AM", "w1AN", "w2AN", 
    "wAA15", "wBA15", "wCA15", "wDA15", "wEA15", "wFA15", "wGA15", "wHA15", "wIA15", "wJA15", "wKA15", "wLA15", "wMA15"
    , "wNA15", "wOA15", "wPA15", "wQA15", "wRA15", "wSA15", "wTA15", "wUA15", "wVA15", "wWA15", "wXA15", "wYA15", "wZA15"
    , "wA01A15", "wA02A15", "wA03A15", "wA04A15", "wA05A15", "wA06A15", "wA07A15", "wA08A15", "wA09A15", "wA10A15", "wA11A15"
    , "wA12A15", "wA13A15", "wA14A15", "wA15"]
    result = 0
    for w in weights:
        if(w.get_name in weightNames):
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

    raise NotImplementedError("Implement me!")


def make_neural_net_challenging():
    
    raise NotImplementedError("Implement me!")
   

def make_neural_net_two_moons():
    #Inputs
    i1 = Input('i1', 0.0)
    i2 = Input('i2', 0.0)
    #Neuron A
    w_A = random_weight()
    i0_A = Input('i0', -1.0)
    wA  = Weight('wA', w_A)
    w_1A = random_weight()
    w1A = Weight('w1A', w_1A)
    w_2A = random_weight()
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
    w_1C = random_weight()
    w1C = Weight('w1C', w_1C)
    w_2C = random_weight()
    w2C = Weight('w2C', w_2C)
    C = Neuron('C', [i1,i2,i0_C], [w1C,w2C,wC])

    #Neuron D
    w_D = random_weight()
    i0_D = Input('i0', -1.0)
    wD  = Weight('wD', w_B)
    w_1D = random_weight()
    w1D = Weight('w1D', w_1D)
    w_2D = random_weight()
    w2D = Weight('w2D', w_2D)
    D = Neuron('D', [i1,i2,i0_D], [w1D,w2D,wD])

    #Neuron E
    w_E = random_weight()
    i0_E = Input('i0', -1.0)
    wE  = Weight('wE', w_E)
    w_1E = random_weight()
    w1E = Weight('w1E', w_1E)
    w_2E = random_weight()
    w2E = Weight('w2E', w_2E)
    E = Neuron('E', [i1,i2,i0_E], [w1E,w2E,wE])

    #Neuron F
    w_F = random_weight()
    i0_F = Input('i0', -1.0)
    wF  = Weight('wF', w_F)
    w_1F = random_weight()
    w1F = Weight('w1F', w_1F)
    w_2F = random_weight()
    w2F = Weight('w2F', w_2F)
    F = Neuron('F', [i1,i2,i0_F], [w1F,w2F,wF])

    #Neuron G
    w_G = random_weight()
    i0_G = Input('i0', -1.0)
    wG  = Weight('wG', w_G)
    w_1G = random_weight()
    w1G = Weight('w1G', w_1G)
    w_2G = random_weight()
    w2G = Weight('w2G', w_2G)
    G = Neuron('G', [i1,i2,i0_G], [w1G,w2G,wG])

    #Neuron H
    w_H = random_weight()
    i0_H = Input('i0', -1.0)
    wH  = Weight('wH', w_H)
    w_1H = random_weight()
    w1H = Weight('w1H', w_1H)
    w_2H = random_weight()
    w2H = Weight('w2H', w_2H)
    H = Neuron('H', [i1,i2,i0_H], [w1H,w2H,wH])

    #Neuron I
    w_I = random_weight()
    i0_I = Input('i0', -1.0)
    wI  = Weight('wI', w_I)
    w_1I = random_weight()
    w1I = Weight('w1I', w_1I)
    w_2I = random_weight()
    w2I = Weight('w2I', w_2I)
    I = Neuron('I', [i1,i2,i0_I], [w1I,w2I,wI])

    #Neuron J
    w_J = random_weight()
    i0_J = Input('i0', -1.0)
    wJ  = Weight('wJ', w_J)
    w_1J = random_weight()
    w1J = Weight('w1J', w_1J)
    w_2J = random_weight()
    w2J = Weight('w2J', w_2J)
    J = Neuron('J', [i1,i2,i0_J], [w1J,w2J,wJ])

    #Neuron K
    w_K = random_weight()
    i0_K = Input('i0', -1.0)
    wK  = Weight('wK', w_K)
    w_1K = random_weight()
    w1K = Weight('w1K', w_1K)
    w_2K = random_weight()
    w2K = Weight('w2K', w_2K)
    K = Neuron('K', [i1,i2,i0_K], [w1K,w2K,wK])

    #Neuron L
    w_L = random_weight()
    i0_L = Input('i0', -1.0)
    wL  = Weight('wL', w_L)
    w_1L = random_weight()
    w1L = Weight('w1L', w_1L)
    w_2L = random_weight()
    w2L = Weight('w2L', w_2L)
    L = Neuron('L', [i1,i2,i0_L], [w1L,w2L,wL])

    #Neuron M
    w_M = random_weight()
    i0_M = Input('i0', -1.0)
    wM  = Weight('wM', w_M)
    w_1M = random_weight()
    w1M = Weight('w1M', w_1M)
    w_2M = random_weight()
    w2M = Weight('w2M', w_2M)
    M = Neuron('M', [i1,i2,i0_M], [w1M,w2M,wM])

    #Neuron N
    w_N = random_weight()
    i0_N = Input('i0', -1.0)
    wN  = Weight('wN', w_N)
    w_1N = random_weight()
    w1N = Weight('w1N', w_1N)
    w_2N = random_weight()
    w2N = Weight('w2N', w_2N)
    N = Neuron('N', [i1,i2,i0_N], [w1N,w2N,wN])

    #Neuron O
    w_O = random_weight()
    i0_O = Input('i0', -1.0)
    wO  = Weight('wO', w_O)
    w_1O = random_weight()
    w1O = Weight('w1O', w_1O)
    w_2O = random_weight()
    w2O = Weight('w2O', w_2O)
    O = Neuron('O', [i1,i2,i0_O], [w1O,w2O,wO])

    #Neuron P
    w_P = random_weight()
    i0_P = Input('i0', -1.0)
    wP  = Weight('wP', w_P)
    w_1P = random_weight()
    w1P = Weight('w1P', w_1P)
    w_2P = random_weight()
    w2P = Weight('w2P', w_2P)
    P = Neuron('P', [i1,i2,i0_P], [w1P,w2P,wP])

    #Neuron Q
    w_Q = random_weight()
    i0_Q = Input('i0', -1.0)
    wQ  = Weight('wQ', w_Q)
    w_1Q = random_weight()
    w1Q = Weight('w1Q', w_1Q)
    w_2Q = random_weight()
    w2Q = Weight('w2Q', w_2Q)
    Q = Neuron('Q', [i1,i2,i0_Q], [w1Q,w2Q,wQ])

    #Neuron R
    w_R = random_weight()
    i0_R = Input('i0', -1.0)
    wR  = Weight('wR', w_R)
    w_1R = random_weight()
    w1R = Weight('w1R', w_1R)
    w_2R = random_weight()
    w2R = Weight('w2R', w_2R)
    R = Neuron('R', [i1,i2,i0_R], [w1R,w2R,wR])

    #Neuron S
    w_S = random_weight()
    i0_S = Input('i0', -1.0)
    wS  = Weight('wS', w_S)
    w_1S = random_weight()
    w1S = Weight('w1S', w_1S)
    w_2S = random_weight()
    w2S = Weight('w2S', w_2S)
    S = Neuron('S', [i1,i2,i0_S], [w1S,w2S,wS])

    #Neuron T
    w_T = random_weight()
    i0_T = Input('i0', -1.0)
    wT  = Weight('wT', w_T)
    w_1T = random_weight()
    w1T = Weight('w1T', w_1T)
    w_2T = random_weight()
    w2T = Weight('w2T', w_2T)
    T = Neuron('T', [i1,i2,i0_T], [w1T,w2T,wT])

    #Neuron U
    w_U = random_weight()
    i0_U = Input('i0', -1.0)
    wU  = Weight('wU', w_H)
    w_1U = random_weight()
    w1U = Weight('w1U', w_1U)
    w_2U = random_weight()
    w2U = Weight('w2U', w_2U)
    U = Neuron('U', [i1,i2,i0_U], [w1U,w2U,wU])

    #Neuron V
    w_V = random_weight()
    i0_V = Input('i0', -1.0)
    wV  = Weight('wV', w_V)
    w_1V = random_weight()
    w1V = Weight('w1V', w_1V)
    w_2V = random_weight()
    w2V = Weight('w2V', w_2V)
    V = Neuron('V', [i1,i2,i0_V], [w1V,w2V,wV])

    #Neuron W
    w_W = random_weight()
    i0_W = Input('i0', -1.0)
    wW  = Weight('wW', w_W)
    w_1W = random_weight()
    w1W = Weight('w1W', w_1W)
    w_2W = random_weight()
    w2W = Weight('w2W', w_2W)
    W = Neuron('W', [i1,i2,i0_W], [w1W,w2W,wW])

    #Neuron X
    w_X = random_weight()
    i0_X = Input('i0', -1.0)
    wX  = Weight('wX', w_X)
    w_1X = random_weight()
    w1X = Weight('w1X', w_1X)
    w_2X = random_weight()
    w2X = Weight('w2X', w_2X)
    X = Neuron('X', [i1,i2,i0_X], [w1X,w2X,wX])

    #Neuron Y
    w_Y = random_weight()
    i0_Y = Input('i0', -1.0)
    wY  = Weight('wY', w_Y)
    w_1Y = random_weight()
    w1Y = Weight('w1Y', w_1Y)
    w_2Y = random_weight()
    w2Y = Weight('w2Y', w_2Y)
    Y = Neuron('Y', [i1,i2,i0_Y], [w1Y,w2Y,wY])

    #Neuron Z
    w_Z = random_weight()
    i0_Z = Input('i0', -1.0)
    wZ  = Weight('wZ', w_Z)
    w_1Z = random_weight()
    w1Z = Weight('w1Z', w_1Z)
    w_2Z = random_weight()
    w2Z = Weight('w2Z', w_2Z)
    Z = Neuron('Z', [i1,i2,i0_Z], [w1Z,w2Z,wZ])

    #Neuron A01
    w_A01 = random_weight()
    i0_A01 = Input('i0', -1.0)
    wA01  = Weight('wA01', w_A01)
    w_1A01 = random_weight()
    w1A01 = Weight('w1A01', w_1A01)
    w_2A01 = random_weight()
    w2A01 = Weight('w2A01', w_2A01)
    A01 = Neuron('A01', [i1,i2,i0_A01], [w1A01,w2A01,wA01])

    #Neuron A02
    w_A02 = random_weight()
    i0_A02 = Input('i0', -1.0)
    wA02  = Weight('wA02', w_A02)
    w_1A02 = random_weight()
    w1A02 = Weight('w1A02', w_1A02)
    w_2A02 = random_weight()
    w2A02 = Weight('w2A02', w_2A02)
    A02 = Neuron('A02', [i1,i2,i0_A02], [w1A02,w2A02,wA02])

    #Neuron A03
    w_A03 = random_weight()
    i0_A03 = Input('i0', -1.0)
    wA03  = Weight('wA03', w_A03)
    w_1A03 = random_weight()
    w1A03 = Weight('w1A03', w_1A03)
    w_2A03 = random_weight()
    w2A03 = Weight('w2A03', w_2A03)
    A03 = Neuron('A03', [i1,i2,i0_A03], [w1A03,w2A03,wA03])

    #Neuron A04
    w_A04 = random_weight()
    i0_A04 = Input('i0', -1.0)
    wA04  = Weight('wA04', w_A04)
    w_1A04 = random_weight()
    w1A04 = Weight('w1A04', w_1A04)
    w_2A04 = random_weight()
    w2A04 = Weight('w2A04', w_2A04)
    A04 = Neuron('A04', [i1,i2,i0_A04], [w1A04,w2A04,wA04])

    #Neuron A05
    w_A05 = random_weight()
    i0_A05 = Input('i0', -1.0)
    wA05  = Weight('wA05', w_A05)
    w_1A05 = random_weight()
    w1A05 = Weight('w1A05', w_1A05)
    w_2A05 = random_weight()
    w2A05 = Weight('w2A05', w_2A05)
    A05 = Neuron('A05', [i1,i2,i0_A05], [w1A05,w2A05,wA05])

    #Neuron A06
    w_A06 = random_weight()
    i0_A06 = Input('i0', -1.0)
    wA06  = Weight('wA06', w_A06)
    w_1A06 = random_weight()
    w1A06 = Weight('w1A06', w_1A06)
    w_2A06 = random_weight()
    w2A06 = Weight('w2A06', w_2A06)
    A06 = Neuron('A06', [i1,i2,i0_A06], [w1A06,w2A06,wA06])

    #Neuron A07
    w_A07 = random_weight()
    i0_A07 = Input('i0', -1.0)
    wA07  = Weight('wA07', w_A07)
    w_1A07 = random_weight()
    w1A07 = Weight('w1A07', w_1A07)
    w_2A07 = random_weight()
    w2A07 = Weight('w2A07', w_2A07)
    A07 = Neuron('A07', [i1,i2,i0_A07], [w1A07,w2A07,wA07])
 
    #Neuron A08
    w_A08 = random_weight()
    i0_A08 = Input('i0', -1.0)
    wA08  = Weight('wA08', w_A08)
    w_1A08 = random_weight()
    w1A08 = Weight('w1A08', w_1A08)
    w_2A08 = random_weight()
    w2A08 = Weight('w2A08', w_2A08)
    A08 = Neuron('A08', [i1,i2,i0_A08], [w1A08,w2A08,wA08])
 
    #Neuron A09
    w_A09 = random_weight()
    i0_A09 = Input('i0', -1.0)
    wA09  = Weight('wA09', w_A09)
    w_1A09 = random_weight()
    w1A09 = Weight('w1A09', w_1A09)
    w_2A09 = random_weight()
    w2A09 = Weight('w2A09', w_2A09)
    A09 = Neuron('A09', [i1,i2,i0_A09], [w1A09,w2A09,wA09])
 
    #Neuron A10
    w_A10 = random_weight()
    i0_A10 = Input('i0', -1.0)
    wA10  = Weight('wA10', w_A10)
    w_1A10 = random_weight()
    w1A10 = Weight('w1A10', w_1A10)
    w_2A10 = random_weight()
    w2A10 = Weight('w2A10', w_2A10)
    A10 = Neuron('A10', [i1,i2,i0_A10], [w1A10,w2A10,wA10])
 
    #Neuron A11
    w_A11 = random_weight()
    i0_A11 = Input('i0', -1.0)
    wA11  = Weight('wA11', w_A11)
    w_1A11 = random_weight()
    w1A11 = Weight('w1A11', w_1A11)
    w_2A11 = random_weight()
    w2A11 = Weight('w2A11', w_2A11)
    A11 = Neuron('A11', [i1,i2,i0_A11], [w1A11,w2A11,wA11])
 
    #Neuron A12
    w_A12 = random_weight()
    i0_A12 = Input('i0', -1.0)
    wA12  = Weight('wA12', w_A12)
    w_1A12 = random_weight()
    w1A12 = Weight('w1A12', w_1A12)
    w_2A12 = random_weight()
    w2A12 = Weight('w2A12', w_2A12)
    A12 = Neuron('A12', [i1,i2,i0_A12], [w1A12,w2A12,wA12])
 
    #Neuron A13
    w_A13 = random_weight()
    i0_A13 = Input('i0', -1.0)
    wA13  = Weight('wA13', w_A13)
    w_1A13 = random_weight()
    w1A13 = Weight('w1A13', w_1A13)
    w_2A13 = random_weight()
    w2A13 = Weight('w2A13', w_2A13)
    A13 = Neuron('A13', [i1,i2,i0_A13], [w1A13,w2A13,wA13])
 
    #Neuron A14
    w_A14 = random_weight()
    i0_A14 = Input('i0', -1.0)
    wA14  = Weight('wA14', w_A14)
    w_1A14 = random_weight()
    w1A14 = Weight('w1A14', w_1A14)
    w_2A14 = random_weight()
    w2A14 = Weight('w2A14', w_2A14)
    A14 = Neuron('A14', [i1,i2,i0_A14], [w1A14,w2A14,wA14])

    #Neuron A15
    w_A15 = random_weight()
    i0_A15 = Input('i0', -1.0)
    wA15  = Weight('wA15', w_C)
    w_AA15 = random_weight()
    wAA15  = Weight('wAA15', w_AA15)
    w_BA15 = random_weight()
    wBA15  = Weight('wBA15', w_BA15)
    w_CA15 = random_weight()
    wCA15  = Weight('wCA15', w_CA15)
    w_DA15 = random_weight()
    wDA15  = Weight('wDA15', w_DA15)
    w_EA15 = random_weight()
    wEA15  = Weight('wEA15', w_EA15)
    w_FA15 = random_weight()
    wFA15  = Weight('wFA15', w_FA15)
    w_GA15 = random_weight()
    wGA15  = Weight('wGA15', w_GA15)
    w_HA15 = random_weight()
    wHA15  = Weight('wHA15', w_HA15)
    w_IA15 = random_weight()
    wIA15  = Weight('wIA15', w_IA15)
    w_JA15 = random_weight()
    wJA15  = Weight('wJA15', w_JA15)
    w_KA15 = random_weight()
    wKA15  = Weight('wKA15', w_KA15)
    w_LA15 = random_weight()
    wLA15  = Weight('wLA15', w_LA15)
    w_MA15 = random_weight()
    wMA15  = Weight('wMA15', w_MA15)
    w_NA15 = random_weight()
    wNA15  = Weight('wNA15', w_NA15)
    w_OA15 = random_weight()
    wOA15  = Weight('wOA15', w_OA15)
    w_PA15 = random_weight()
    wPA15  = Weight('wPA15', w_PA15)
    w_QA15 = random_weight()
    wQA15  = Weight('wQA15', w_QA15)
    w_RA15 = random_weight()
    wRA15  = Weight('wRA15', w_RA15)
    w_SA15 = random_weight()
    wSA15  = Weight('wSA15', w_SA15)
    w_TA15 = random_weight()
    wTA15  = Weight('wTA15', w_TA15)
    w_UA15 = random_weight()
    wUA15  = Weight('wUA15', w_UA15)
    w_VA15 = random_weight()
    wVA15  = Weight('wVA15', w_VA15)
    w_WA15 = random_weight()
    wWA15  = Weight('wWA15', w_WA15)
    w_XA15 = random_weight()
    wXA15  = Weight('wXA15', w_XA15)
    w_YA15 = random_weight()
    wYA15  = Weight('wYA15', w_YA15)
    w_ZA15 = random_weight()
    wZA15  = Weight('wZA15', w_ZA15)
    w_A01A15 = random_weight()
    wA01A15  = Weight('wA01A15', w_A01A15)
    w_A02A15 = random_weight()
    wA02A15  = Weight('wA02A15', w_A02A15)
    w_A03A15 = random_weight()
    wA03A15  = Weight('wA03A15', w_A03A15)
    w_A04A15 = random_weight()
    wA04A15  = Weight('wA04A15', w_A04A15)
    w_A05A15 = random_weight()
    wA05A15  = Weight('wA05A15', w_A05A15)
    w_A06A15 = random_weight()
    wA06A15  = Weight('wA06A15', w_A06A15)
    w_A07A15 = random_weight()
    wA07A15  = Weight('wA07A15', w_A07A15)
    w_A08A15 = random_weight()
    wA08A15  = Weight('wA08A15', w_A08A15)
    w_A09A15 = random_weight()
    wA09A15  = Weight('wA09A15', w_A09A15)
    w_A10A15 = random_weight()
    wA10A15  = Weight('wA10A15', w_A10A15)
    w_A11A15 = random_weight()
    wA11A15  = Weight('wA11A15', w_A11A15)
    w_A12A15 = random_weight()
    wA12A15  = Weight('wA12A15', w_A12A15)
    w_A13A15 = random_weight()
    wA13A15  = Weight('wA13A15', w_A13A15)
    w_A14A15 = random_weight()
    wA14A15  = Weight('wA14A15', w_A14A15)

    A15 = Neuron('A15', [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, A01, A02, A03, A04, A05
    , A06, A07, A08, A09, A10, A11, A12, A13, A14, i0_A15],
     [wAA15, wBA15, wCA15, wDA15, wEA15, wFA15, wGA15, wHA15, wIA15, wJA15, wKA15, wLA15, wMA15, wNA15, wOA15, wPA15
     , wQA15, wRA15, wSA15, wTA15, wUA15, wVA15, wWA15, wXA15, wYA15, wZA15, wA01A15, wA02A15, wA03A15, wA04A15
     , wA05A15, wA06A15, wA07A15, wA08A15, wA09A15, wA10A15, wA11A15, wA12A15, wA13A15, wA14A15, wA15])
    #Performance Elem
    # Per = PerformanceElem(A15, 0.0)
    Per = RegularizedPerformanceElem(A15,0.0)
    net = Network(Per,[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, A01, A02, A03, A04, A05
    , A06, A07, A08, A09, A10, A11, A12, A13, A14, A15])
    return net

    raise NotImplementedError("Implement me!")

class RegularizedPerformanceElem(PerformanceElem):
    def output(self):
        return -1*((1/2) * ((self.my_desired_val - self.my_input.output())**2)) - (0.00035*self.my_norm)
        raise NotImplementedError("Implement me!")

    def dOutdX(self, elem):
        weightNames = ["w1A", "w2A", "w1B", "w2B", "w1C", "w2C", "w1D", "w2D", "w1E", "w2E", "w1F", "w2F", "w1G", "w2G", "w1H", "w2H"
        , "w1I", "w2I", "w1J", "w2J", "w1K", "w2K", "w1L", "w2L", "w1M", "w2M", "w1N", "w2N", "w1O", "w2O", "w1P", "w2P", "w1Q"
        , "w2Q", "w1R", "w2R", "w1S", "w2S", "w1T", "w2T", "w1U", "w2AU", "w1V", "w2V", "w1W", "w2W", "w1X", "w2X", "w1Y", "w2Y"
        , "w1Z", "w2Z", "w1AA", "w2AA", "w1AB", "w2AB", "w1AC", "w2AC", "w1AD", "w2AD", "w1AE", "w2AE", "w1AF", "w2AF", "w1AG"
        , "w2AG", "w1AH", "w2AH", "w1AI", "w2AI", "w1AJ", "w2AJ", "w1AK", "w2AK", "w1AL", "w2AL", "w1AM", "w2AM", "w1AN", "w2AN", 
        "wAA15", "wBA15", "wCA15", "wDA15", "wEA15", "wFA15", "wGA15", "wHA15", "wIA15", "wJA15", "wKA15", "wLA15", "wMA15"
        , "wNA15", "wOA15", "wPA15", "wQA15", "wRA15", "wSA15", "wTA15", "wUA15", "wVA15", "wWA15", "wXA15", "wYA15", "wZA15"
        , "wA01A15", "wA02A15", "wA03A15", "wA04A15", "wA05A15", "wA06A15", "wA07A15", "wA08A15", "wA09A15", "wA10A15", "wA11A15"
        , "wA12A15", "wA13A15", "wA14A15", "wA15"]
        if(elem.get_name() in weightNames):
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
    for i in range(0, len(li)):
        if(isclose(li[i], li2[i])):
            print("TRUEEEEE")
        else:
            print("FALSEEEE")
            print(li[i], li2[i], "\n")