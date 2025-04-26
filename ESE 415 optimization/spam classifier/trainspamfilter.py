
import numpy as np
import time
from loss_function.ridge import ridge
from loss_function.hinge import hinge
from loss_function.logistic import logistic
from grdscent import grdescent
from prediction.spamfilter import spamfilter

def trainspamfilter(xTr,yTr,loss_function_name="rideg",grad_method="newton"):
    #
    # INPUT:
    # xTr
    # yTr
    #
    # OUTPUT: w_trained
    #
    # Consider optimizing the input parameters for your loss and GD!


    # (not the most successful) EXAMPLE:
    print("...using ridge")
    '''f = lambda w : ridge(w,xTr,yTr,1)

    w_trained = grdescent(f,np.zeros((xTr.shape[0],1)),1e-05,1000)'''


    #五折验证
    np.random.seed(55)
    n_samples=len(xTr[0])
    
    remaind = n_samples % 5
    if remaind != 0:
        num_suplement = 5 - remaind  
        indices = np.random.choice(n_samples, num_suplement, replace=True)  
        X_add = xTr[:,indices]
        y_add = yTr[:,indices]
        xTr= np.hstack([xTr, X_add])  
        yTr= np.hstack([yTr, y_add])  
    indices = np.arange(len(xTr[0]))
    np.random.shuffle(indices)
    folds_5_X = np.array(np.array_split(xTr[:,indices], 5,axis=1))
    folds_5_y = np.array(np.array_split(yTr[:,indices], 5,axis=1))

    '''# print("...usinge hinge")
    # f = lambda w: hinge(w, xTr, yTr, lambda_reg)

    w_initial = np.zeros((xTr.shape[0], 1))
    w_trained = grdescent(f, w_initial, learning_rate, max_iters, tolerance)'''

   
    hyper_stepsizes=np.logspace(-6,-1,num=6)
    if loss_function_name == "rigde":
        lambd_values = np.linspace(0, 1, 11)
    elif loss_function_name=="hinge":
        lambd_values = np.linspace(0, 1, 11)
    else:    
        lambd_values=[0]
    
    

    # 展平网格为一维数组，生成参数组合
    folds_data = []  #  (X_train, Y_train, X_val, Y_val)

    for i in range(5):
        X_val = np.array(folds_5_X[i])
        Y_val = np.array(folds_5_y[i])

        X_train = np.empty((X_val.shape[0], 0))
        Y_train = np.empty((1, 0))

        for j in range(5):
            if j != i:
                X_train = np.concatenate([X_train, folds_5_X[j]], axis=1)
                Y_train = np.concatenate([Y_train, folds_5_y[j]], axis=1)
        folds_data.append((X_train, Y_train, X_val, Y_val))
    best_score = -np.inf

    start_time=time.time()

    for lambd in lambd_values:
        for hyper_stepsize in hyper_stepsizes:
            score = 0
        
            for (X_train, Y_train, X_val, Y_val) in folds_data:
                if  loss_function_name == "ridge":
                    f = lambda w: ridge(w, X_train, Y_train, lambd, grad_method)
                elif loss_function_name == "hinge":
                    f = lambda w: hinge(w, X_train, Y_train, lambd, grad_method)
                else:
                    f = lambda w: logistic(w, X_train, Y_train,grad_method)
                w_trained = grdescent(f, np.zeros((X_train.shape[0], 1)), hyper_stepsize, 1000, grad_method)
                _, _, auc = spamfilter(X_val, Y_val, w_trained)
                score += auc
        avg_score = score / 5

        if avg_score > best_score:
            best_score = avg_score
            best_stepsize = hyper_stepsize
            best_lambda = lambd
    end_time=time.time()
    print(f"Data training time: {end_time-start_time :.2f} seconds\n")
    print("Best stepsize:", best_stepsize)
    print("Best lambda:", best_lambda)

   # 根据 function_name 选损失函数，同时传入最佳 lambda
    if loss_function_name== "ridge":
        f = lambda w: ridge(w, xTr, yTr, best_lambda,grad_method)
    elif loss_function_name == "hinge":
        f = lambda w: hinge(w, xTr, yTr, best_lambda,grad_method)
    else:
        f = lambda w: logistic(w, xTr, yTr,grad_method)

    # 用最佳 stepsize 重新在全训练集上训练模型
    w_trained = grdescent(f, np.zeros((xTr.shape[0], 1)), best_stepsize, 1000,grad_method)

    # 保存训练好的权重
    np.save('w_trained.npy', w_trained)

    np.save('w_trained.npy', w_trained)
    return w_trained
