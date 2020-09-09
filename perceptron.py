import numpy as np

w = np.zeros( (5, 5) )

baseA_down=[[1,0,0,0,1],
            [1,0,0,0,1],
            [0,1,1,1,0],
            [0,1,0,1,0],
            [0,0,1,0,0]]

baseA_up = [[0,0,1,0,0],
            [0,1,0,1,0],
            [0,1,1,1,0],
            [1,0,0,0,1],
            [1,0,0,0,1]]
            
training_data = [ 
[[0,0,1,0,1],
[0,1,0,1,0],
[0,1,1,1,0],
[1,0,0,0,1],
[1,0,1,0,1]],

[[0,1,1,0,0],
[0,1,0,1,0],
[0,1,1,1,0],
[1,1,0,0,1],
[1,0,0,0,1]],
                
[[0,0,1,0,0],
[0,1,0,1,0],
[0,1,1,1,1],
[1,0,0,0,1],
[1,0,1,0,1]],
            
[[0,0,1,0,0],
[0,1,1,1,0],
[0,1,1,1,0],
[1,0,0,0,1],
[1,0,0,1,1]],

[[1,0,0,0,1],
[1,0,0,0,1],
[0,1,1,1,0],
[0,1,0,1,0],
[1,0,1,0,1]],

[[1,0,1,0,1],
[1,0,0,0,1],
[0,1,1,1,0],
[1,1,0,1,0],
[0,0,1,0,0]],


[[0,0,1,0,0],
[0,1,0,1,0],
[0,1,1,1,0],
[1,0,0,0,1],
[1,0,0,0,1]],

[[1,0,0,0,1],
[1,0,0,0,1],
[0,1,1,1,0],
[0,1,0,1,0],
[0,0,1,0,0]],


[[1,0,1,0,1],
[1,1,0,1,1],
[0,1,1,1,0],
[0,1,0,1,0],
[0,0,1,0,0]],

[[1,0,0,0,1],
[1,0,0,0,1],
[0,1,1,1,0],
[0,1,0,1,0],
[0,1,0,1,0]]

]

#Tipo de resposta: 0 = down, 1 = up 
expected_response = [1,1,1,1,0,0,1,0,0,0]

               
def evaluate(data,w):
    result = 0
    for i in range(0,5):
        for j in range(0,5):
            result += data[i][j] * w[i][j]
    if(result <= 0 ):
        return 0
    else:
        return 1
        
        
def train(data,w,n,r,er):
    for i in range(0,5):
        for j in range(0,5):
            w[i][j] += n*(er - r)*data[i][j]



n = 0.5
repeat = True

## Inicio do treinamento
while(repeat):
    repeat = False
    for i in range (0,len(training_data)):
        response = evaluate(training_data[i],w)
        if( response != expected_response[i] ):
            train(training_data[i],w,n,response,expected_response[i])
            repeat = True
            
## Teste com outro dado
weirdA_up = [[1,0,0,1,0],
            [0,1,0,1,0],
            [0,1,1,0,0],
            [1,0,0,1,1],
            [0,1,0,0,1]]
print(w)
print(evaluate(weirdA_up,w))
