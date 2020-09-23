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

    W1 = np.random.rand(n_hidden, n_input)
    B1 = np.random.rand(n_hidden, 1)
    W2 = np.random.rand(n_output, n_hidden)
    B2 = np.random.rand(n_output, 1)


    return {'W1': W1, 'B1': B1, 'W2': W2, 'B2': B2}

def forward(X, params):
    '''
    Forward linear equation calculus propagation
    '''

    W1 = params['W1']
    B1 = params['B1']
    W2 = params['W2']
    B2 = params['B2']

    Z1 = np.dot(W1, X) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + B2
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
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def predict(params, X):
    '''
    Predictions function
    '''

    A2, fw = forward(X, parameters)
    
    return A2


if __name__ == "__main__":
    
    size = 15

    X = np.identity(size)

    #y = np.array([0, 0, 1, 1])
    #y = y.reshape(1, y.shape[0])

    n_input = X.shape[0]**2
    n_hidden = int((X.shape[0]/2)**2)
    n_output = X.shape[0]**2


    parameters = initialize_weights(n_input, n_hidden, n_output)

    epochs = 10
    examples = X.shape[1]
    iterations = 10
    learning_rate = 0.1

    dataset = []

    for i in range(10):
        X_random = np.random.randint(2, size=(size, size))

        dataset.append(X_random.flatten().reshape(n_input, 1))

    X = X.flatten().reshape(n_input, 1)

    for epoch in range(epochs):

        print("Epoch %s:" % str(epoch + 1))

        for inputs in dataset:

            for i in range(iterations):

                inputs = X.flatten().reshape(n_input, 1)

                A2, fw = forward(inputs, parameters)

                cost = cost_function(A2, X, parameters)

                grads = backward(parameters, fw, inputs, X)

                W1 = parameters["W1"]
                B1 = parameters["B1"]
                W2 = parameters["W2"]
                B2 = parameters["B2"]
            
                dW1 = grads["dW1"]
                db1 = grads["db1"]
                dW2 = grads["dW2"]
                db2 = grads["db2"]

                W1 = W1 - learning_rate * dW1
                B1 = B1 - learning_rate * db1
                W2 = W2 - learning_rate * dW2
                B2 = B2 - learning_rate * db2

                parameters["W1"] = W1
                parameters["B1"] = B1
                parameters["W2"] = W2
                parameters["B2"] = B2


            predictions = predict(parameters, inputs)
            print("predictions mean = " + str(np.mean(predictions)))