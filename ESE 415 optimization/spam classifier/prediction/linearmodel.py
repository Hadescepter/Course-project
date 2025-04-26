def linearmodel(w, xTe):
    preds = w.T @ xTe
# INPUT:
# w weight vector (default w=0)
# xTe dxn matrix (each column is an input vector)
#
# OUTPUTS:
#
# preds predictions
    preds=w.T@xTe

    return preds
