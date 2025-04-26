import numpy as np

def spamupdate(w,email,truth):
    email = email.reshape(-1, 1) if email.ndim == 1 else email
    pred = w.T @ email
    if np.sign(pred) != truth:
        w += 0.1 * truth * email

    return w
