import numpy as np
def grdescent(func,w0,stepsize,maxiter,grad_method="Fixed_step",tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent
    previous_loss_value = float('inf')
    w=w0


    if grad_method=="Backtracking":
        while(maxiter>0):
            maxiter=maxiter-1
            _, grad,_ = func(w)
            actual_grad_value = np.linalg.norm(grad)
            #如果梯度过小可以提前结束循环
            if actual_grad_value < tolerance:
                break

            w_new = w - stepsize * grad
            updated_loss_value, _,_ = func(w_new)
            #将损失函数变大，则需要调整步长防止错过最优点
            if updated_loss_value < previous_loss_value:
                stepsize *= 1.01
                w = w_new
                previous_loss_value = updated_loss_value
            else:
                stepsize *= 0.5
                stepsize = max(stepsize, eps)
    elif grad_method == "Fixed_step":
        while maxiter > 0:
            maxiter -= 1
            _, grad, _ = func(w)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < tolerance:
                break
            w = w - stepsize * grad
    elif grad_method == "newton":
        while maxiter > 0:
            maxiter -= 1
            _, grad, H = func(w)
            grad_norm = np.linalg.norm(grad)

            if grad_norm < tolerance:
                break
            # 加一点正则项防止Hessian奇异（数值稳定）
            H_reg = H + eps * np.eye(H.shape[0])
            # Newton step: w_new = w - H^{-1} * grad
            try:
                delta = np.linalg.solve(H_reg, grad)  # 更稳定比直接inv快
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H_reg) @ grad  # Hessian不可逆时退而求其次
            w = w - delta
    else:
        raise ValueError(f"Unknown grad_method: {grad_method}")
    return w
