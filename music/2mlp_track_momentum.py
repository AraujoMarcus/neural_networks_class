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
    W2 = np.random.rand(n_output, n_hidden + 1)
    deltaW1 = np.zeros((n_hidden, n_input + 1))
    deltaW2 = np.zeros((n_output, n_hidden + 1))

    return {'W1': W1, 'W2': W2, 'deltaW1': deltaW1, 'deltaW2': deltaW2}

def forward(X, params):
    '''
    Forward linear equation calculus propagation
    '''
    #X = X.T

    W1 = params['W1'][:,1:]
    B1 = W1[:,0].reshape(W1.shape[0], 1)
    W2 = params['W2'][:,1:]
    B2 = W2[:,0].reshape(W2.shape[0], 1)

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
    W2 = params['W2'][:,1:]

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
    Prediction function
    '''

    A2, fw = forward(X, parameters)
    #predictions = (A2 > 0.5)
    
    return A2


if __name__ == "__main__":
    
    dataset = []
    with open('/home/guilherme/workspace/neural_networks_class/music/default_features_1059_tracks.txt', 'r') as fs:
        for line in fs:
            l = line.split(',')
            dataset.append(l)

    fs.close()

    dataset = np.array(dataset)
    #np.random.shuffle(dataset)

    X = dataset[:,:-2]
    z = []

    for i in dataset[:,-1]: 
        z.append(i.split('\n')[0])

    z = np.array(z)
    dataset[:, -1] = z.T

    y = dataset[:,68:]

    X = np.array(X).astype('float').T

    y = np.array(y).astype('float')
    y = [i.T.reshape(2, 1) for i in y]
    

    n_input = X.shape[0]
    n_hidden = X.shape[0] * 2
    n_output = 2


    parameters = initialize_weights(n_input, n_hidden, n_output)

    epochs = 5
    examples = X.shape[1]
    iterations = 100
    learning_rate = 0.02
    mu = 0.1

    cost = 0

    for epoch in range(epochs):

        print("Epoch %s:" % str(epoch + 1))

        for e in range(examples):

            for i in range(iterations):

                x = X[:,e].reshape(X.shape[0], 1)
                y_true = y[e]

                A2, fw = forward(x, parameters)

                cost = cost_function(A2, y_true, parameters)

                grads = backward(parameters, fw, x, y_true)

                W1 = parameters["W1"][:,1:]
                b1 = parameters["W1"][:,0].reshape(W1.shape[0], 1)
                W2 = parameters["W2"][:,1:]
                b2 = parameters["W2"][:,0].reshape(W2.shape[0], 1)
                deltaW1 = parameters["deltaW1"][:,1:]
                deltab1 = parameters["deltaW1"][:,0].reshape(W1.shape[0], 1)
                deltaW2 = parameters["deltaW2"][:,1:]
                deltab2 = parameters["deltaW2"][:,0].reshape(W2.shape[0], 1)
            
                dW1 = grads["dW1"]
                db1 = grads["db1"]
                dW2 = grads["dW2"]
                db2 = grads["db2"]

                W1 = (learning_rate * dW1) + (mu * deltaW1) 
                b1 = (learning_rate * db1) + (mu * deltab1)
                W2 = (learning_rate * dW2) + (mu * deltaW2)
                b2 = (learning_rate * db2) + (mu * deltab2)

                deltaW1 = dW1
                deltab1 = db1
                deltaW2 = dW2
                deltab2 = db2

                parameters["W1"][:,1:] = W1
                parameters["W1"][:,0] = b1.reshape(b1.shape[0])
                parameters["W2"][:,1:] = W2
                parameters["W2"][:,0] = b2.reshape(b2.shape[0])
                parameters["deltaW1"][:,1:] = deltaW1
                parameters["deltaW1"][:,0] = deltab1.reshape(b1.shape[0])
                parameters["deltaW2"][:,1:] = deltaW2
                parameters["deltaW2"][:,0] = deltab2.reshape(b2.shape[0])
            
        print("Cost: %.4f" % cost)



    # Test fase
    preds = []
    for i in range(100):

        random = np.random.randint(178, size=1)

        x = X[:,random[0]].reshape(X.shape[0], 1)
        y_true = y[random[0]]
        #print("True wine category: " + str(y_true))

        lst = predict(parameters, x)
        print("predictions: " + str(lst))

        predictions = []
        for i in lst:
            if i < 0.500:
                predictions.append([0.000])
            else:
                predictions.append([1.000])

        np.set_printoptions(formatter={'float': lambda predictions: "{0:0.3f}".format(predictions)})
        preds.append(np.array_equal(predictions, y_true))

    total = 0
    for i in preds:
        if i:
            total += 1

    print('Accuracy: %d%%' % total)
