from prediction.spamfilter import spamfilter
from trainspamfilter import trainspamfilter
from util.valsplit import valsplit

import numpy as np


# load the data:
# load the data:
data = np.load('dataset/data_test_kaggle.npy', allow_pickle=True).item()
X = data['X']
Y = data['Y']

# split the data
# xTr and xVal will be of the shape d x n (num_dimensions x num_datapoints)
xTr,xtest,yTr,ytest = valsplit(X,Y)

#loss_function["hinge, logistic, ridge"] and grad_method selcetion["Fixed_step,newton, Backtracking"] 
loss_function="hinge"
grad_method="Backtracking"

# train spam filter with settings and parameters in trainspamfilter.py
w_trained = trainspamfilter(xTr,yTr,loss_function,grad_method)

# evaluate spam filter on validation set using default threshold
spamfilter(xtest,ytest,w_trained)

# load the data:
data = np.load('dataset/data_train_default.npy', allow_pickle=True).item()
X = data['X']
Y = data['Y']

# split the data
# xTr and xVal will be of the shape d x n (num_dimensions x num_datapoints)
xTr,xtest,yTr,ytest = valsplit(X,Y)

#loss_function["hinge, logistic, ridge"] and grad_method selcetion["Fixed_step,newton, Backtracking"] 
loss_function="hinge"
grad_method="Backtracking"

# train spam filter with settings and parameters in trainspamfilter.py
w_trained = trainspamfilter(xTr,yTr,loss_function,grad_method)

# evaluate spam filter on validation set using default threshold
spamfilter(xtest,ytest,w_trained)


