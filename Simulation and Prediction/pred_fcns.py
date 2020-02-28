"""
Here are functions of prediction models and their
parameters estimators
"""

import numpy as np



"""

Model structure: example for 2x2 MIMO system:
    
y1[k] = [ -y1[k-1]  -y2[k-1] + u1[k-1] + u2[k-1]] * Theta1
y2[k] = [ -y1[k-1]  -y2[k-1] + u1[k-1] + u2[k-1]] * Theta2

and thus:
    
[y1[k] y2[k]] = [ -y1[k-1]  -y2[k-1]  u1[k-1]  u2[k-1]] [Theta1 Theta2]

ny : orders of all outputs to model, eg. ny = [1, 1]
nu : orders of all inputs to model, eg. nu = [1, 1]
h : regressor vector eg. h = [ -y1[k-1]  -y2[k-1]  u1[k-1]  u2[k-1] ]

"""


# ARX prediction N-steps ahead
def arx_pred(ny, nu, h, theta, N):
    
    

    
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
    
    
    nyu = np.concatenate((ny, nu)) # orders of both inputs and outputs of model
    
    if N>0:
        if N==1:
            y = np.dot(h,theta) # case of one step ahead prediction
        else:
            y = np.zeros([N,outputCount]) # case of multiple steps ahead prediction
            
            for k in range(N):
                # here happens all of recursion stuff for prediction
                if k > 0:
                    
                    index = 0 # starting index by which we locate which part of regressor vector we can shift
                    n_index = 0 # keeping count of the inner-most loop iteration
                    for n in nyu:
                        # here happens shifting in discrete time
                        h[index : n+index] = np.roll( h[index : n+index], 1)
                        # differentiation between inputs and outputs
                        if n_index < outputCount:
                            h[index] = y[k-1, n_index] # putting last predicted value to k-1 place in regressor
                        else:
                            h[index] = 0 # since model input are not predicted we put zero value here
                        
                        index = index + n
                        n_index = n_index +1

                y[k,:] = np.dot(h, theta) # one step ahead prediction
    else:
        raise ValueError("Prediction horizon N must be greater than zero")
    
    return y


# ARX model simulation for input u
def arx_sim(ny, nu, u, y0, theta):
    
    # check for theta vector shape
    if len(theta.shape) == 1:
        outputCount = 1
    elif len(theta.shape) > 1:
        outputCount = theta.shape[1]
    else:
        raise ValueError("Invalid theta shape")
    
    if len(u.shape) == 1:
        inputCount = 1
    elif len(u.shape) > 1:
        inputCount = u.shape[1]
    else:
        raise ValueError("Invalid input array shape")     
    
    nyu = np.concatenate((ny, nu))
    
    y=np.zeros([u.shape[0], outputCount])
    y[0, :]=y0

    h = np.zeros(np.sum(nyu))
    
    # putting initial condition to regressor vector
    # index = 0 
    # n_index = 0
    # for n in nyu:
    #     if n_index < outputCount:
    #         h[0,index] = y[0, n_index] 
    #     index = index + n
    #     n_index = n_index +1
    
    for k in range(1,u.shape[0]):
        

        index = 0 # starting index by which we locate which part of regressor vector we can shift
        n_index = 0 # keeping count of the inner-most loop iteration
        for n in nyu:
            # here happens shifting in discrete time
            h[index : n+index] = np.roll( h[index : n+index], 1)
            # differentiation between inputs and outputs
            if n_index < outputCount:
                h[index] = y[k-1, n_index] # putting last predicted value to k-1 place in regressor
            else:
                h[index] = u[k-1, n_index - outputCount] # since, model input are not predicted we put zero value here
            
            index = index + n
            n_index = n_index +1

        y[k,:] = np.dot(h, theta) # one step ahead prediction
        
    return y




# ARX model parameter estimation via Least-Squares method
def arx_par_est(ny, nu, y, u):
    
    if y.shape[0] != u.shape[0]:
        raise ValueError("Input and output array length must match")
    
    if len(y.shape) == 1:
        outputCount = 1
        y.shape = (y.shape[0], 1)
    elif len(y.shape) > 1:
        outputCount = y.shape[1]
    else:
        raise ValueError("Invalid output array shape")
    
    if len(u.shape) == 1:
        inputCount = 1
        u.shape = (u.shape[0], 1)
    elif len(u.shape) > 1:
        inputCount = u.shape[1]
    else:
        raise ValueError("Invalid input array shape")        
    
    if len(ny) != outputCount or len(nu) != inputCount:
        raise ValueError("Input or output dimensions do not match the sizes of ny/nu arrays")
    
    
    nyu = np.concatenate((ny, nu))
    
    # regressor matrix initialization
    H=np.zeros([y.shape[0],np.sum(nyu)])
    
    # filling up the matrix with output and input signals

    index = 0 # starting index by which we locate which part of regressor vector we can shift
    n_index = 0 # keeping count of the inner-most loop iteration
    for n in nyu:
        for k in range(n):
            if n_index < outputCount:
                H[:,index+k] = np.roll(y[:,n_index],k+1)
                H[0:k+1,index+k] = np.zeros(k+1)
            else:
                H[:,index+k] = np.roll(u[:,n_index - outputCount],k+1)
                H[0:k+1,index+k] = np.zeros(k+1)
                
        index = index + n
        n_index = n_index + 1
        
    return np.linalg.lstsq(H, y, rcond=None)[0]



    
