import numpy as np

#
def createData():
    X = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']])
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    return X, y

def train(X, y):
    num = X.shape[0] #Size of data
    pLabel = np.zeros(numLabels) #To store P(y = c_K)
    for i in range(numLabels):
        pLabel[i] = np.sum(y == label[i])
    #print(pLabel)
    p = np.zeros((numClasses, numLabels)) #Conditional Probability
    for i in range(numClasses):
        for j in range(numLabels):
            tmp = np.reshape(X[y == label[j]], (1, -1))
            p[i][j] = np.sum(tmp == classes[i]) / pLabel[j]
    return pLabel, p

def predict(pLabel, p, x):
    px = np.zeros(numLabels)
    dim = x.shape[0]
    #print(classes == x)
    for i in range(numLabels):
        px[i] = pLabel[i] * 1.0 #for I don't know whether it was a array or a pointer
        for j in range(dim):
            px[i] *= p[int(np.argwhere(classes == x[j]))][i]
    return label[np.argmax(px)]

if __name__ == "__main__":
    X, y = createData()
    label = np.unique(y)
    classes = np.unique(X) #The total number of classes
    numLabels = len(label)
    numClasses = len(classes)
    pLabel, p = train(X, y)
    x = np.array([2, 'S'])
    res = predict(pLabel, p, x)
    print(res)
