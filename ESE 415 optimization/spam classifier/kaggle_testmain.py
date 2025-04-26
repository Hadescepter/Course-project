from prediction.spamfilter import spamfilter


import numpy as np


# load the data:
data = np.load('dataset/data_test_kaggle.npy', allow_pickle=True).item()
X = data['X']
Y = data['Y']

w_trained = np.load('w_trained.npy')
# evaluate spam filter on validation set using default threshold
spamfilter(X,Y,w_trained)