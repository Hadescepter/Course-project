import numpy as np
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

def ridge(w,xTr,yTr,lambdaa,grad_method="newton"):
    '''print("w/",w.shape)
    print("xTr/",xTr.shape)
    print("yTr/",yTr.shape)'''
    #回归损失为w.T*x*x.T*w-2w.T*xy+y.T*y+lambda*w.T*w
    reg_loss=w.T@xTr@xTr.T@w-2*w.T@xTr@yTr.T+yTr@yTr.T+lambdaa*w.T@w
    #通过公式推到可以得到梯度函数=2*（xx.T+lambda*I）*w-x*x*y
    gradient= 2*(xTr@xTr.T+lambdaa*np.eye(len(xTr)))@w-2*xTr@yTr.T
        # 计算二阶导数（Hessian）
    if grad_method == "newton":
        Hf = 2 * (xTr @ xTr.T + lambdaa * np.eye(xTr.shape[0]))
    else:
        Hf = []

    return reg_loss,gradient,Hf
