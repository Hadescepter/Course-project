import math
import numpy as np
'''
    INPUT:
    xTr: dxn matrix - 2d numpy array (each column is an input vector)
    yTr: 1xn vector - 2d numpy array (each entry is a label)
    w :  dx1 weight vector - 2d numpy array (default w=0)

    OUTPUTS:

    loss:      float (the total loss obtained with w on xTr and yTr)
    gradient:  dx1 vector - 2d numpy array (the gradient at w)

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr,grad_method="newton"):
    #优先计算指数项exp（y*w*x）
    z= yTr * (w.T@xTr)
    #对损失函数矩阵进行求和
    loss = np.sum(np.log(1 + np.exp(-z)))
    #通过chain rule 法则先对z进行求导
    diff_loss_z= 1 / (1 + np.exp(z)) 
    #梯度函数为loss关于z的导数乘以z关于w的导数
    gradient = -xTr@( (yTr * diff_loss_z).T ) 
    #计算二阶导数
    if grad_method=="newton":
        sigmoid = 1 / (1 + np.exp(-z))
        second_diff_loss_z = sigmoid * (1 - sigmoid)  # elementwise
       # 构建 D 矩阵（对角矩阵）
        D = np.diagflat((second_diff_loss_z).flatten())
       # Hessian = X * D * X^T
        Hf = xTr @ D @ xTr.T
    else:
        Hf=[]
    return loss,gradient,Hf
