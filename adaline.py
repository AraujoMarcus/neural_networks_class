import numpy as np

test_data = [
# Up 
np.array([[-1,-1,+1,-1,+1],
          [-1,+1,-1,+1,-1],
          [-1,+1,+1,+1,-1],
          [+1,-1,-1,-1,+1],
          [+1,-1,+1,-1,+1]]),
# Up
np.array([[-1,1,1,-1,-1],
          [-1,1,-1,1,-1],
          [-1,1,1,1,-1],
          [1,1,-1,-1,1],
          [1,-1,-1,-1,1]]),
# Up                
np.array([[-1,-1,1,-1,-1],
          [-1,1,-1,1,-1],
          [-1,1,1,1,1],
          [1,-1,-1,-1,1],
          [1,-1,1,-1,1]]),
# Up            
np.array([[-1,-1,1,-1,-1],
          [-1,1,1,1,-1],
          [-1,1,1,1,-1],
          [1,-1,-1,-1,1],
          [1,-1,-1,1,1]]),
# Down
np.array([[1,-1,-1,-1,1],
          [1,-1,-1,-1,1],
          [-1,1,1,1,-1],
          [-1,1,-1,1,-1],
          [1,-1,1,-1,1]]),
# Down
np.array([[1,-1,1,-1,1],
          [1,-1,-1,-1,1],
          [-1,1,1,1,-1],
          [1,1,-1,1,-1],
          [-1,-1,1,-1,-1]]),

# Up
np.array([[-1,-1,1,-1,-1],
          [-1,1,-1,1,-1],
          [-1,1,1,1,-1],
          [1,-1,-1,-1,1],
          [1,-1,-1,-1,1]]),
# Down
np.array([[1,-1,-1,-1,1],
          [1,-1,-1,-1,1],
          [-1,1,1,1,-1],
          [-1,1,-1,1,-1],
          [-1,-1,1,-1,-1]]),

# Down
np.array([[1,-1,1,-1,1],
          [1,1,-1,1,1],
          [-1,1,1,1,-1],
          [-1,1,-1,1,-1],
          [-1,-1,1,-1,-1]]),
# Down
np.array([[1,-1,-1,-1,1],
          [1,-1,-1,-1,1],
          [-1,1,1,1,-1],
          [-1,1,-1,1,-1],
          [-1,1,-1,1,-1]])

]

samples = [
np.array([[+1, -1, -1, -1, +1],
          [+1, -1, -1, -1, +1],
          [-1, +1, +1, +1, -1],
          [-1, +1, -1, +1, -1],
          [-1, -1, +1, -1, -1]]),

np.array([[+1, -1, -1, -1, +1],
          [+1, -1, -1, -1, +1],
          [+1, +1, +1, +1, -1],
          [-1, +1, -1, +1, -1],
          [-1, -1, +1, -1, -1]]),

np.array([[+1, -1, -1, -1, +1],
          [+1, -1, +1, -1, +1],
          [-1, +1, +1, +1, -1],
          [-1, +1, -1, +1, -1],
          [-1, -1, +1, -1, -1]]),

np.array([[+1, +1, -1, +1, +1],
          [+1, -1, -1, -1, +1],
          [-1, +1, +1, +1, -1],
          [-1, +1, -1, +1, -1],
          [-1, -1, +1, -1, -1]]),

np.array([[+1, -1, -1, -1, +1],
          [+1, -1, -1, -1, +1],
          [-1, +1, +1, +1, -1],
          [-1, +1, -1, +1, -1],
          [-1, +1, +1, -1, -1]]),

np.array([[+1, -1, -1, -1, +1],
          [+1, -1, -1, -1, +1],
          [-1, +1, +1, +1, -1],
          [-1, +1, -1, +1, -1],
          [-1, -1, +1, -1, +1]]),

np.array([[-1, -1, +1, -1, -1],
          [-1, +1, -1, +1, -1],
          [-1, +1, +1, +1, -1],
          [+1, -1, -1, -1, +1],
          [+1, -1, -1, -1, +1]]),

np.array([[-1, -1, +1, -1, -1],
          [+1, +1, -1, +1, -1],
          [-1, +1, +1, +1, -1],
          [+1, -1, -1, -1, +1],
          [+1, -1, -1, -1, +1]]),
           
np.array([[-1, -1, +1, -1, -1],
          [-1, +1, -1, +1, -1],
          [-1, +1, +1, +1, -1],
          [+1, -1, -1, -1, +1],
          [+1, -1, +1, -1, +1]]),
           
np.array([[-1, -1, +1, -1, +1],
          [-1, +1, -1, +1, -1],
          [-1, +1, +1, +1, -1],
          [+1, -1, -1, -1, +1],
          [+1, -1, -1, +1, +1]]),
           
np.array([[-1, -1, +1, -1, -1],
          [-1, +1, -1, +1, -1],
          [+1, +1, +1, +1, -1],
          [+1, -1, -1, -1, +1],
          [+1, -1, -1, -1, +1]]),
           
np.array([[-1, -1, +1, -1, -1],
          [-1, +1, -1, +1, -1],
          [-1, +1, +1, +1, -1],
          [+1, -1, -1, -1, +1],
          [+1, +1, +1, -1, +1]])]

targets = [+1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1]

# Com bias
def update_wb(X, y_hat, y, w, learning_rate=0.1):
  errs = y - y_hat
  w[1:] += learning_rate * X.dot(errs)
  w[0] += learning_rate * errs.sum()
  
# Sem bias
def update_w(X, y_hat, y, w, learning_rate=0.1):
  errs = (y - y_hat)
  w += learning_rate * (X * errs)

def fit(X, y, w):
  converging = True
  
  for i in range(10):
    
    # Com bias
    y_hat = X.dot(w[1:]) + w[0]

    # Sem bias
    #y_hat = X.T.dot(w.transpose())
    #print(int(np.sum(y_hat)))
    
    if y != int(np.sum(y_hat)):
      update_wb(X, y_hat, y, w)

def predict(X, w):

  res = -1
  
  #print(w.shape)
  #print(np.sum(X.dot(w[1:]) + w[0]))
  #print(np.sum(X.dot(w)))

  if np.sum(X.dot(w[1:]) + w[0]) <= 0.0:
  #if np.sum(X.dot(w)) <= 0.0:
    res = 1

  return res


size = 5

# Com bias
weights_b = np.zeros(size*size + 1)
weights_b[0] = 1

# Sem bias
#weights = np.zeros(size*size)

#print("Original weights:")
#print(weights_b)

# Etapa de treinamento
for data, target in zip(samples, targets):
  fit(data.flatten(), target, weights_b)

#print("New weights:")
#print(weights_b)

for test in test_data:
  pred = predict(test.flatten(), weights_b)

  if pred == -1:
    print('Up')
  else:
    print('Down')