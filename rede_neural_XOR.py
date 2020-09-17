import numpy as np
import math

'''
size = 8

Ws = []

Ws.append(np.zeros((size + 1,int(math.log2(size))))) # +1 para Bias
Ws.append(np.zeros((int(math.log2(size)) + 1,size)))


input = np.identity(size)
'''

##Criando matriz W XOR segundo exemplo da sala de aula
Ws = np.array([[[0.4,0.8],[0.5,0.8],[-0.6,-0.2]],[[-0.4],[0.9],[-0.3]]])



               
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
            internals[i][j] = 1 / (1 + math.exp(-sum))

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
            
    print(internals)
    print(Ws)


#Teste
n = 0.5
repeat = True
train_data = [[1,1],[0,0],[0,1],[1,0]]
expected = [[0],[0],[1],[1]]
err2 = 0

for i in range(len(train_data)):
    internals = evaluate(train_data[i],Ws)
    train(train_data[i],internals,Ws,expected[i])
    



