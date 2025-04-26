from numpy import maximum
import numpy as np
import math
'''
    INPUT:
    xTr:    dxn matrix - 2d numpy array (each column is an input vector)
    yTr:    1xn vector - 2d numpy array (each entry is a label)
   lambdaa: float (regularization constant)
    w:      dx1 weight vector - 2d numpy array (default w=0)

    OUTPUTS:

    reg_loss:      float (the total regularized loss obtained with w on xTr and yTr)
    gradient:  dx1 vector - 2d numpy array (the gradient at w)

    [d,n]=size(xTr);
'''
def hinge(w,xTr,yTr,lambdaa,grad_method="newton"):
    #计算loss_function中的y*w*X的值
    z=yTr*(w.T@xTr)
    #损失函数由进行正则化
    reg_loss=np.sum(np.maximum(0.0,1.0-z))+lambdaa*((w.T)@w)
    #指示方程（矩阵中未达到目标预测标签的记为1，需要进行梯度下降）
    indicator_function_ywx=(z<1.0).astype(float)
    #梯度下降由损失函数和正则化项构成
    gradient=-xTr@((yTr*indicator_function_ywx).T)+2*lambdaa*np.eye(len(xTr))@w
    #计算hessian矩阵
    if grad_method=="newton":
        D = np.diagflat(indicator_function_ywx.flatten())
        Hf = xTr @ D @ xTr.T + 2 * lambdaa * np.eye(xTr.shape[0])
    else:
        Hf=[]
    return reg_loss,gradient, Hf
