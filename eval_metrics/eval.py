import numpy as np

def mae(y, yhat):
    return (np.abs(y - yhat)).mean()


def mape(pred, actual):
    #     return np.mean(abs((actual - pred) / actual))
    return np.sum(abs(actual - pred)) / np.sum(actual)


def mase(pred, actual, naive, ma):
    # compare mean aboslute error of |actual - pred| with naive random walk or mean average
    index = []
    index += [i for i, x in enumerate(naive) if x == -99999]
    index += [i for i, x in enumerate(ma) if x == -99999]

    if index:
        actual = np.delete(actual, index)
        ma = np.delete(ma, index)
        naive = np.delete(naive, index)
        pred = np.delete(pred, index)

    residual = abs(actual - pred)
    ma_residual = abs(actual - ma)
    naive_residual = abs(actual - naive)
    return np.mean(residual) / np.mean(ma_residual), np.mean(residual) / np.mean(naive_residual)

def mae_scale(y, yhat, naive):
    return np.abs(y - yhat) / np.abs(y - naive)