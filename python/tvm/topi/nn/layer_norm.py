from tvm import auto_scheduler
import re
from tvm import te
import tvm
import tvm.topi as topi

def layer_norm(data, gamma, beta, axis, epsilon, center, scale):
    size = data.shape[axis]
    data_square = topi.multiply(data, data)
    mean = topi.divide(topi.sum(data, axis, True), size)
    mean_squared = topi.multiply(mean, mean)
    squared_mean = topi.divide(topi.sum(data_square, axis, True), size)
    var = topi.subtract(squared_mean, mean_squared)
    denom = topi.sqrt(topi.add(var, epsilon))
    out = topi.divide(topi.subtract(data, mean), denom)

    if scale:
        out = topi.multiply(out, gamma)
    if center:
        out = topi.add(out, beta)

    return out
    