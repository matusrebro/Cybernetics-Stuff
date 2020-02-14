"""
Here are functions of prediction models and their
parameters estimators
"""

import numpy as np



class Prediction:
    
    # ARX prediction N-steps ahead
    def arx_pred(ny, nu, h, theta, N):
        
        
        """
        
        Model structure example for 2x2 MIMO system:
            
        y1[k] = [ -y1[k-1]  -y2[k-1] + u1[k-1] + u2[k-1]] * Theta1
        y2[k] = [ -y1[k-1]  -y2[k-1] + u1[k-1] + u2[k-1]] * Theta2
        
        and thus:
            
        [y1[k] y2[k]] = [ -y1[k-1]  -y2[k-1] + u1[k-1] + u2[k-1]] [Theta1 Theta2]
        
        ny : orders of all outputs to model, eg. ny = [1, 1]
        nu : orders of all inputs to model, eg. nu = [1, 1]
        h : regressor vector which eg. h = [ -y1[k-1]  -y2[k-1] + u1[k-1] + u2[k-1]]
        
        """
        
        # check for theta vector shape
        if len(theta.shape) == 1:
            outputCount = 1
        elif len(theta.shape) > 1:
            outputCount = theta.shape[1]
        else:
            raise ValueError("Invalid theta shape")
        
        
        if len(ny) != outputCount:
            raise ValueError("Invalid ny vector length")
            
        if np.sum(ny) + np.sum(nu) != len(h):
            raise ValueError("Length of h vector doesnt equal to sum of orders of inputs and outputs")
        
        
        nyu = np.concatenate((ny, nu))
        
        if N>0:
            if N==1:
                y = np.dot(h,theta)
            else:
                y = np.zeros([N,outputCount])
                
                for k in range(N):
                    if k > 0:
                        
                        for n in range(0,len(nyu)-1):
                            
                            hp = np.roll( h[ n*nyu[n] :n*nyu[n]+1  ] )
                        
                        
                        yp=np.roll(yp,1)
                        yp[0]=y[k-1]
                        up=np.roll(up,1)
                        up[0]=0
                    y[k]=-np.dot(yp,theta_AR) + np.dot(up,theta_X)
        else:
            raise ValueError("Prediction horizon N must be greater than zero")
        
        return y


class Simulation:

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

class ParameterEstimation:
    
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



    
