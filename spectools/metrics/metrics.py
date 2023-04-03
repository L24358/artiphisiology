import numpy as np
from scipy.interpolate import CubicSpline

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

def cubic_spline(invec, ival=1/200): # invec.shape = (-1, 2)
    cs = CubicSpline(np.linspace(0, 1, len(invec)), invec)
    xs = np.arange(0, 1, ival)
    return cs(xs), cs(xs, 1), cs(xs, 2)

def curvature(dp, ddp):
    numerator = np.linalg.norm(np.cross(dp, ddp))
    denominator = np.linalg.norm(dp)**3
    return numerator/denominator

def angle(p): return np.arctan2(*p[::-1])

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


