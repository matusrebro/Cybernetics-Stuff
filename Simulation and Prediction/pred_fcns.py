"""
Here are functions of prediction models and their
parameters estimators
"""

import numpy as np

# ARX prediction N-steps ahead
def arx_pred(yp,up,theta_AR,theta_X, N):
    na=len(yp)
    nb=len(up)
    
    if na is not len(theta_AR) or nb is not len(theta_X):
        raise ValueError("Different lengths of parameter and regressor arrays")
    else:
        pass
    
    if N>0:
        if N==1:
            y=-np.dot(yp,theta_AR) + np.dot(up,theta_X)
        else:
            y=np.zeros(N)
            
            for k in range(N):
                if k>0:
                    yp=np.roll(yp,1)
                    yp[0]=y[k-1]
                    up=np.roll(up,1)
                    up[0]=0
                y[k]=-np.dot(yp,theta_AR) + np.dot(up,theta_X)
    else:
        raise ValueError("Prediction horizon N must be greater than zero")
    
    return y

# ARX model simulation for input u
def arx_sim(u,y0,theta_AR,theta_X):
    y=np.zeros_like(u)
    y[0]=y0
    yp=np.zeros_like(theta_AR)
    yp[0]=y0
    up=np.zeros_like(theta_X)
    for k in range(1,len(u)):
        y[k]=-np.dot(yp,theta_AR) + np.dot(up,theta_X)
        yp=np.roll(yp,1)
        yp[0]=y[k]
        up=np.roll(up,1)
        up[0]=u[k]        
    return y

# ARX model parameter estimation via Least-Squares method
def arx_par_est(na,nb,y,u):
    H=np.zeros([len(y),na+nb])
    for k in range(H.shape[1]):
        if k<na:
            H[:,k]=-np.roll(y,k+1)
            H[0:k+1,k] = np.zeros(k+1)
        else:
            H[:,k]=np.roll(u,k+1-na)
            H[0:k+1-na,k] = np.zeros(k+1-na)
    return np.linalg.lstsq(H, y, rcond=None)[0]



    
