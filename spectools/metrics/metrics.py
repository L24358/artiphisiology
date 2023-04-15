"""
References:
    - NMSE: https://math.stackexchange.com/questions/488964/the-definition-of-nmse-normalized-mean-square-error
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import kurtosis

def MSE(x, y): return np.mean((x-y)**2)

def R2(x, y): return 1 - MSE(x, y)/np.std(x)/np.std(y)

def IGV(data): # (mean) In-Group Variance, data.shape = (scale, images)
    return np.mean([np.var(row) for row in data.T]) # TODO: normalize to data variance

def NMSE(x, y):
    proj = (x+y)/2 # closest point on the x=y line to the data points
    err = [np.linalg.norm([x[i]-proj[i], y[i]-proj[i]]) for i in range(len(x))] # distance (2-norm)
    normalization = (np.std(x)*np.std(y))**0.5 # for scale invariance (tested)
    return np.mean(err)/normalization

def double_exp(ksts, coefs):
    mu_c, sig_c, mu_a, sig_a, k = coefs

    res = []
    for x in ksts:
        x_c, x_a = x.T
        prob = np.exp(-(x_c - mu_c)**2/(2*sig_c**2)) * np.exp(-(x_a - mu_a)**2/(2*sig_a**2))
        res.append(prob)
    
    return k*max(res)

def double_exp_wrap(xbundle, coefs):
    return np.array([double_exp(ksts, coefs) for ksts in xbundle])

def apc_obj_func(coefs, xbundle, Rs):
    R_preds = double_exp_wrap(xbundle, coefs)
    return sum((R_preds - Rs)**2)

def cubic_spline(invec, ival=1/200): # invec.shape = (-1, 2)
    cs = CubicSpline(np.linspace(0, 1, len(invec)), invec)
    xs = np.arange(0, 1, ival)
    return cs(xs), cs(xs, 1), cs(xs, 2)

def curvature(dp, ddp):
    numerator = np.linalg.norm(np.cross(dp, ddp))
    denominator = np.linalg.norm(dp)**3
    return numerator/denominator

def angle(p): return np.arctan2(*p[::-1])

def responsive(Rs, thre):
    res, idx = [], []
    for i, R in enumerate(Rs):
        if np.mean(R) >= thre:
            res.append(R)
            idx.append(i)
    return np.vstack(res), idx

def response_sparsity(Rs):
    res, idx = [], []
    for i, R in enumerate(Rs):
        k = kurtosis(R)
        if (k >= 2.9) and (k <= 42):
            res.append(R)
            idx.append(i)
    return np.vstack(res), idx

def fvmax(invec):
    """By Taekjun Kim, Anitha Pasupathy, Wyeth Bair, April 23, 2020."""
    num = np.shape(invec)[0]
    inshft = np.vstack((invec[num-2,:],invec,invec[1,:]))
    ip = np.arange(0,1,1/50)
    
    vtx = np.empty((1,num*50-49))
    vty = np.empty((1,num*50-49))
    dvtx = np.empty((1,num*50-49))
    dvty = np.empty((1,num*50-49))
    
    for i in range(0,num-1):
        bufvrt = inshft[i:i+4,:]
        
        # Spline equation
        incr = np.empty((4,len(ip)))
        incr[0,:] =   -ip*ip*ip + 3*ip*ip -3*ip + 1
        incr[1,:] =  3*ip*ip*ip - 6*ip*ip +4
        incr[2,:] = -3*ip*ip*ip + 3*ip*ip +3*ip + 1
        incr[3,:] =    ip*ip*ip
        
        dincr = np.empty((4,len(ip)));
        dincr[0,:] = -3*ip*ip +  6*ip - 3
        dincr[1,:] =  9*ip*ip - 12*ip
        dincr[2,:] = -9*ip*ip +  6*ip + 3
        dincr[3,:] =  3*ip*ip
        
        vtx[0,i*50:(i+1)*50] = np.sum(np.tile(bufvrt[:,0].reshape(4,1),(1,50))*incr,axis=0)/6.0
        vty[0,i*50:(i+1)*50] = np.sum(np.tile(bufvrt[:,1].reshape(4,1),(1,50))*incr,axis=0)/6.0
        
        dvtx[0,i*50:(i+1)*50] = np.sum(np.tile(bufvrt[:,0].reshape(4,1),(1,50))*dincr,axis=0)/6.0
        dvty[0,i*50:(i+1)*50] = np.sum(np.tile(bufvrt[:,1].reshape(4,1),(1,50))*dincr,axis=0)/6.0
    
    # periodic boundaries
    vtx[0,(num-1)*50] = vtx[0,0]
    vty[0,(num-1)*50] = vty[0,0]
    dvtx[0,(num-1)*50] = dvtx[0,0]
    dvty[0,(num-1)*50] = dvty[0,0]
    
    outvec = np.transpose(np.vstack((vtx, vty)))
    doutvec = np.transpose(np.vstack((dvtx, dvty)))
    return outvec, doutvec


if __name__ == "__main__":
    x = np.array([5, 8, 6])
    y = np.array([7, 7, 6])
    nmse = NMSE(x, y) # check scale invariance