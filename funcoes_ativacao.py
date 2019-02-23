import numpy as np
#transfer functions

def stepFunction(soma):
    print(soma)
    if (soma >= 1):
        return 1
    return 0

def sigmoidFunction(soma):
    return 1/(1+np.exp(-soma))

def tahnFunction(soma):
    return (np.exp(soma)-np.exp(-soma))/(np.exp(soma)+np.exp(-soma))

def reluFunction(soma):
    if soma>=0:
        return soma
    return 0

def linearFunction(soma):
    return soma

def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

soma = 5*0.2+2*0.5+1*0.1

teste = stepFunction(soma)
teste1 = sigmoidFunction(soma)
teste2 = tahnFunction(soma)
teste3 = reluFunction(soma)
teste4 = linearFunction(soma)
valores = [5.0, 2.0, 1.3]


print(teste)
print(teste1)
print(teste2)
print(teste3)
print(teste4)
print(softmaxFunction(valores))