import math

def sigmoid(s, beta):
    return 1/(1+math.e**(-s*beta))

def continuous_sparsification(model, loader):
