import numpy as np


# Conval
def conv2(X, k):
    x_row, x_col = X.shape
    k_row, k_col = k.shape
    ret_row, ret_col = x_row - k_row + 1, x_col - k_col + 1
    
    ret = np.empty((ret_row, ret_col))
    for y in range(ret_row):
        for x in range(ret_col):
            sub = X[y: y + k_row, x: x + k_col]
            ret[y, x] = np.sum(k * sub)            
    return ret
	

def conv3(X, k):
    x_row, x_col, x_ch = X.shape
    k_row, k_col = k.shape
    ret_row, ret_col, ret_ch = x_row - k_row + 1, x_col - k_col + 1, x_ch
    
    ret = np.empty((ret_row, ret_col, ret_ch))
    for c in range(ret_ch):
        for y in range(ret_row):
            for x in range(ret_col):
                sub = X[y: y + k_row, x: x + k_col, c: c+1]
                ret[y, x, c] = np.sum(k * sub.reshape(k_row, k_col))          
    return ret
	
	
	
# MAX & MEAN Pooling
def pooling_2d(X, mode=np.max, size=2):
    x_row, x_col = X.shape
    ret_row, ret_col = x_row // size, x_col // size + 1
    
    ret = np.empty((ret_row, ret_col))
    for i1, y in enumerate(range(0, x_row, size)):
        for i2, x in enumerate(range(0, x_col, size)):
            sub = X[y: y + size, x: x + size]
            ret[i1, i2] = mode(sub)
            
    return ret


def pooling_3d(X, mode=np.max, size=2):
    x_row, x_col, x_ch = X.shape
    ret_row, ret_col = x_row // size, x_col // size + 1
    
    ret = np.empty((ret_row, ret_col, x_ch))
    for c in range(x_ch):
        for i1, y in enumerate(range(0, x_row, size)):
            for i2, x in enumerate(range(0, x_col, size)):
                sub = X[y: y + size, x: x + size, c: c + 1]
                ret[i1, i2, c] = mode(sub)
                
    return ret
	
	
	
	
# add Padding
def padding_2d(X, k_size=3):
    x_row, x_col = X.shape
    pad_size = (k_size - 1) // 2
    
    ret = np.zeros((x_row + pad_size*2, x_col+ pad_size*2))
    ret[pad_size: x_row + pad_size, pad_size: x_col + pad_size] = X[:, :]   

    return ret
	
	
def padding_3d(X, k_size=3):
    x_row, x_col, x_ch = X.shape
    pad_size = (k_size - 1) // 2
    
    ret = np.zeros((x_row + pad_size*2, x_col+ pad_size*2, x_ch))
    ret[pad_size: x_row + pad_size, pad_size: x_col + pad_size, :] = X[:, :, :]   

    return ret