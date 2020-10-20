import numpy as np
import math
import random
import pandas as pd
          
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

        
def train(inp,internals,Ws,expected,n,m,delta_Ws):

    
    #Calculo da ultima camada
    for i in range(0,len(Ws[-1]) - 1):
        for j in range(0,len(Ws[-1][0])):
            delta_Ws[-1][i][j] = n * (expected[j] - internals[-1][j]) * internals[-1][j] * (1 - internals[-1][j]) * internals[-2][i] + m * delta_Ws[-1][i][j]
            Ws[-1][i][j] += delta_Ws[-1][i][j]
    
    #Bias da ultima camada
    for j in range(0,len(Ws[-1][0])):
        delta_Ws[-1][-1][j] = n * (expected[j] - internals[-1][j]) * internals[-1][j] * (1 - internals[-1][j]) + m * delta_Ws[-1][-1][j]
        Ws[-1][-1][j] += delta_Ws[-1][-1][j]
        
    #Calculo das camadas internas
    for i in range (len(internals)-2,-1,-1):
        for j in range(0,len(Ws[i][0])):
            dpjw = 0 
            for k in range(0,len(internals[i+1])):
                dpjw += (expected[k] - internals[i+1][k]) * internals[i+1][k] * ( 1 - internals[i+1][k]) * Ws[i+1][j][k]
            for k in range(0,len(Ws[i]) - 1):
                delta_Ws[i][k][j] = n * dpjw * internals[i][j] * ( 1 - internals[i][j] ) * inp[k] + m * delta_Ws[i][k][j]
                Ws[i][k][j] += delta_Ws[i][k][j]
            delta_Ws[i][-1][j] =  n * dpjw * internals[i][j] * ( 1 - internals[i][j] ) + m * delta_Ws[i][-1][j]
            Ws[i][-1][j] += delta_Ws[i][k][j]
            
    #print(internals)



#Teste com tracks#

#Fixando sementes de valores aleatorios
np.random.seed(1)
random.seed(1)

#Estrutura da rede
in_size = 68
middle_size = 5
out_size = 2

##Obtendo dados e normalizando para escala entre 0 e 1
data = pd.read_csv('default_features_1059_tracks.txt',header=None)
for i in range(0,70):
    aux = list(data[i])
    maior = max(aux)
    menor = min(aux)
    aux = [ (aux[k] - menor ) / (maior - menor) for k in range(len(aux))]
    data[i] = aux
    
numeric_data = data.values.tolist()

#Embaralhando dados
random.shuffle(numeric_data)

#Definindo parametros de execução
cicles = [400,800]
aprendizado=[0.5]
momentum = [0.3,0.4]
train_size = [(.80,.20),(.70,.30),(.60,.40)]
data_size = len(numeric_data)

output = open("results.csv",'w')
output.write('cicle;learn_rate;momentum;proportion;accuracy\n')


#Teste com 1 camada intermediaria
for cicle in cicles:
    for n in aprendizado:
        for m in momentum:
            for proportion in train_size:
                
                
                
                #Inicializando matriz W com pesos aleatorios
                Ws_1cam = []
                delta_Ws_1cam = []
                Ws_1cam.append(np.random.rand(in_size + 1,middle_size)) # +1 para Bias
                Ws_1cam.append(np.random.rand(middle_size + 1,out_size))
                delta_Ws_1cam.append(np.zeros((in_size + 1,middle_size))) # +1 para Bias
                delta_Ws_1cam.append(np.zeros((middle_size + 1,out_size)))
                
                
            
                #Separando base de treino e testes
                train_data = numeric_data[0:int(proportion[0]*data_size)]
                test_data = numeric_data[int(proportion[0]*data_size):data_size]
                
                #Iniciando treinamento de N ciclos
                for i in range(cicle):
                    for j in range(len(train_data)):
                        td = train_data[j][0:68]
                        #vi = (vi - vmin)/(vmax-vmin)
                        internals = evaluate(td,Ws_1cam)
                        train(td,internals,Ws_1cam,train_data[j][68:70],n,m,delta_Ws_1cam)

                            
                #Utilizando base de testes
                err = 0
  
                for i in range(len(test_data)):
                    td = test_data[i][0:68]
                    internals = evaluate(td,Ws_1cam)
                    result = internals[-1]
                    print(result,test_data[i][68:70])
                    err += (result[0] - test_data[i][68])**2 + (result[1] - test_data[i][69])**2 
                print(cicle,n,m,proportion,err)        
                output.write(str(cicle)+';'+str(n)+';'+str(m)+';'+str(proportion)+';'+str(err)+'\n')
                
output.close()



'''

##Teste com WINE

#Fixando sementes de valores aleatorios
np.random.seed(1)
random.seed(1)

#Estrutura da rede
in_size = 13
middle_size = 5
out_size = 3

##Obtendo dados e normalizando para escala entre 0 e 1
data = pd.read_csv('wine.data',header=None)
for i in range(1,14):
    aux = list(data[i])
    maior = max(aux)
    menor = min(aux)
    aux = [ (aux[k] - menor ) / (maior - menor) for k in range(len(aux))]
    data[i] = aux
    
numeric_data = data.values.tolist()

#Embaralhando dados
random.shuffle(numeric_data)

#Definindo parametros de execução
cicles = [100,200,400,800]
aprendizado=[0.1,0.2,0.3,0.4,0.5]
momentum = [0.0,0.1,0.2,0.3,0.4]
train_size = [(.80,.20),(.70,.30),(.60,.40)]
data_size = len(numeric_data)

output = open("results.csv",'w')
output.write('cicle;learn_rate;momentum;proportion;accuracy\n')


#Teste com 1 camada intermediaria
for cicle in cicles:
    for n in aprendizado:
        for m in momentum:
            for proportion in train_size:
                
                
                
                #Inicializando matriz W com pesos aleatorios
                Ws_1cam = []
                delta_Ws_1cam = []
                Ws_1cam.append(np.random.rand(in_size + 1,middle_size)) # +1 para Bias
                Ws_1cam.append(np.random.rand(middle_size + 1,out_size))
                delta_Ws_1cam.append(np.zeros((in_size + 1,middle_size))) # +1 para Bias
                delta_Ws_1cam.append(np.zeros((middle_size + 1,out_size)))
                
                
            
                #Separando base de treino e testes
                train_data = numeric_data[0:int(proportion[0]*data_size)]
                test_data = numeric_data[int(proportion[0]*data_size):data_size]
                
                #Iniciando treinamento de N ciclos
                for i in range(cicle):
                    for j in range(len(train_data)):
                        td = train_data[j][1:14]
                        #vi = (vi - vmin)/(vmax-vmin)
                        internals = evaluate(td,Ws_1cam)
                        if(int(train_data[j][0]) == 1):
                            train(td,internals,Ws_1cam,[1,0,0],n,m,delta_Ws_1cam)
                        if(int(train_data[j][0]) == 2):
                            train(td,internals,Ws_1cam,[0,1,0],n,m,delta_Ws_1cam)
                        else:
                            train(td,internals,Ws_1cam,[0,0,1],n,m,delta_Ws_1cam)
                            
                #Utilizando base de testes
                acertos = 0
  
                for i in range(len(test_data)):
                    td = test_data[i][1:14]
                    internals = evaluate(td,Ws_1cam)
                    result = internals[-1]
                    print(result,test_data[i][0])
                    if(result.index(max(result)) + 1 == int(test_data[i][0])):
                        acertos += 1
                        
                print(cicle,n,m,proportion)        
                output.write(str(cicle)+';'+str(n)+';'+str(m)+';'+str(proportion)+';'+str(100 * acertos/len(test_data))+'\n')

output.close()
'''
