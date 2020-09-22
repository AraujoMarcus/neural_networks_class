import numpy as np 


def sigmoid (z):
    '''
    Sigmoid function for normalizing results
    '''

    s = 1 / (1 + np.exp(-z))

    return s

def initialize_weights(n_input, n_hidden, n_output):
    '''
    Initialize weights and bias with random values
    '''

    W1 = np.random.rand(n_hidden, n_input + 1)
    W2 = np.random.rand(n_output, n_input + 1)


    return {'W1': W1, 'W2': W2}

def forward(X, params):
    '''
    Forward linear equation calculus propagation
    '''

    W1 = params['W1']
    B1 = W1[:,0].reshape(W1[:,0].shape[0], 1)
    W2 = params['W2']
    B2 = W2[:,0].reshape(W2[:,0].shape[0], 1)

    Z1 = np.dot(W1[:,1:], X) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2[:,1:], X) + B2
    A2 = sigmoid(Z2)

    fw = {'Z1': Z1,
          'A1': A1,
          'A2': A2,
          'Z2': Z2}

    return A2, fw

def cost_function(A2, y, params):
    '''
    Computes the cost of output layer
    '''
    y = y.reshape(y.shape[0], 1)
    m = y.shape[1]

    cost = -(np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2)))/m

    cost = float(np.squeeze(cost))

    return cost

def backward(params, fw, X, y):
    '''
    Applies backward propagation
    '''

    m = X.shape[1]

    #W1 = params['W1']
    W2 = params['W2']

    A1 = fw['A1']
    A2 = fw['A2']

    dZ2 = A2 - y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2[:,1:].T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def predict(params, X):
    '''
    Predictions function
    '''

    A2, fw = forward(X, parameters)
    predictions = (A2 > 0.5)
    
    return predictions


if __name__ == "__main__":
    

    X = np.array([[1, 0, 0, 1],
                  [1, 0, 1, 0]])

    y = np.array([0, 0, 1, 1])

    n_input = X.shape[0]
    n_hidden = 2
    n_output = y.shape[0]


    parameters = initialize_weights(n_input, n_hidden, n_output)

    epochs = 2000
    learning_rate = 0.1

    for epoch in range(epochs):

        A2, fw = forward(X, parameters)

        cost = cost_function(A2, y, parameters)

        grads = backward(parameters, fw, X, y)

        W1 = parameters["W1"][:,1:]
        b1 = parameters["W1"][:,0].reshape(W1.shape[0], 1)
        W2 = parameters["W2"][:,1:]
        b2 = parameters["W2"][:,0].reshape(W2.shape[0], 1)
    
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        parameters["W1"][:,1:] = W1
        parameters["W1"][:,0] = b1.reshape(b1.shape[0])
        parameters["W2"][:,1:] = W2
        parameters["W2"][:,0] = b2.reshape(b2.shape[0])

        print("W1 = " + str(parameters["W1"][:,1:]))
        print("b1 = " + str(parameters["W1"][:,0]))
        print("W2 = " + str(parameters["W2"][:,1:]))
        print("b2 = " + str(parameters["W2"][:,0]))

        predictions = predict(parameters, X)
        print("predictions mean = " + str(np.mean(predictions)))
