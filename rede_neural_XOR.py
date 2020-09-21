import numpy as np
import math

          
def evaluate(inp,Ws):

    n_internals = len(Ws) 
    internals = []

    #Respostas intermediarias + resposta final na ultima posição de Internals
    for i in range(0,n_internals):
        internals.append( [ 0 for i in range(0,len(Ws[i][0]))] )
        
    
    for i in range(0,n_internals):
        if(i == 0):
            data = inp
        else:
            data = internals[i-1]
        for j in range(0,len(internals[i])):
            sum = 0
            for k in range (0,len(data)):
                sum += data[k] * Ws[i][k][j]
            sum += Ws[i][-1][j]
            internals[i][j] = 1.0 / (1.0 + math.exp(-sum))

    return internals

        
def train(inp,internals,Ws,expected):

    n = 0.5
    
    #Calculo da ultima camada
    for i in range(0,len(Ws[-1]) - 1):
        for j in range(0,len(Ws[-1][0])):
            Ws[-1][i][j] += n * (expected[j] - internals[-1][j]) * internals[-1][j] * (1 - internals[-1][j]) * internals[-2][i]
    
    #Bias da ultima camada
    for j in range(0,len(Ws[-1][0])):
        Ws[-1][-1][j] += n * (expected[j] - internals[-1][j]) * internals[-1][j] * (1 - internals[-1][j])
        
    #Calculo das camadas internas
    for i in range (len(internals)-2,-1,-1):
        for j in range(0,len(Ws[i][0])):
            dpjw = 0 
            for k in range(0,len(internals[i+1])):
                dpjw += (expected[k] - internals[i+1][k]) * internals[i+1][k] * ( 1 - internals[i+1][k]) * Ws[i+1][j][k]
            for k in range(0,len(Ws[i]) - 1):
                Ws[i][k][j] += n * dpjw * internals[i][j] * ( 1 - internals[i][j] ) * inp[k]
            Ws[i][-1][j] += n * dpjw * internals[i][j] * ( 1 - internals[i][j] )
            
    #print(internals)
    #print(Ws)

'''
#Teste com XOR#

##Criando matriz W XOR segundo exemplo da sala de aula
Ws = np.array([[[0.4,0.8],[0.5,0.8],[-0.6,-0.2]],[[-0.4],[0.9],[-0.3]]])
train_data = [[1,1],[0,0],[0,1],[1,0]]
expected = [[0],[0],[1],[1]]
err2 = 9999
count = 0
while(err2/4.0 > 0.001):
    err2 = 0
    for i in range(len(train_data)):
        internals = evaluate(train_data[i],Ws)
        train(train_data[i],internals,Ws,expected[i])
        err2 += (internals[-1][0] - expected[i][0]) ** 2
    count += 1   
    
    #Visualizando evolução do erro
    print(err2)
    
    
print("\nTreiando com " + str(count) +" passos")
print("Testando XOR com entrada 0 0")
internals = evaluate([0,0],Ws)
print(internals[-1][0])

print("Testando XOR com entrada 0 1")
internals = evaluate([0,1],Ws)
print(internals[-1][0])

print("Testando XOR com entrada 1 0")
internals = evaluate([1,0],Ws)
print(internals[-1][0])

print("Testando XOR com entrada 1 1")
internals = evaluate([1,1],Ws)
print(internals[-1][0])
'''




##Teste com identidade

size = 15
Ws = []
#Inicializando matriz W com pesos aleatorios
Ws.append(np.random.rand(size + 1,int(math.ceil(math.log2(size))))) # +1 para Bias
Ws.append(np.random.rand(int(math.log2(size)) + 1,size))
train_data = np.identity(size)

err = 99999
step = 100
count = 0
train_length = size
while (err/train_length > 0.01):
    for i in range(train_length):
        internals = evaluate(train_data[i],Ws)
        train(train_data[i],internals,Ws,train_data[i])
    count += 1
    if( count % step == 0):
        err = 0
        for i in range(train_length):
            internals = evaluate(train_data[i],Ws)
            for j in range(len(train_data)):
                err += (internals [-1][j] - train_data[i][j]) ** 2
                
        #visualizando evolução do erro
        print(err/train_length)

print("Treiando com " + str(count) +" passos")
print("Testando resultado")
result = []
intermediario = []
for i in range(len(train_data)):
    internals = evaluate(train_data[i],Ws)
    result.append(internals[-1])
    intermediario.append(internals[-2])
    
print("Identidade obtida:\n")
print(result)
print()
print("Resultado intermediario/encoding:\n")
print(intermediario)
