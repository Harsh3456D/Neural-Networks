import numpy as np

# Single Layer Network
class SingleLayer:
    def layer():
        x = np.array([[0,0,1],
                     [0,1,1],
                     [1,0,1],
                     [1,1,1]])
        
        y = np.array([[0,1,1,0]]).T
        
        syn0 = 2*np.random.random((3,4)) - 1
        syn1 = 2*np.random.random((4,1)) - 1
        
        for j in range(60000):
            l1 = 1/(1+np.exp(-(np.dot(x,syn0))))
            l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
            l2_delta = (y - l2)*(l2*(1-l2))
            l1_delta = l2_delta.dot(syn1.T) * (l1 * (1 - l1))
            syn1 += l1.T.dot(l2_delta)
            syn0 += x.T.dot(l1_delta)
            
# Double layer Network
class DoubleLayer():
    def nonlin(x, deriv = False):
        if deriv == True:
            return x*(1-x)
        return 1/(1+np.exp(-x))
    
    def Layer():
        X = np.array([[0,0,1],
                      [0,1,1],
                      [1,0,1],
                      [1,1,1]])
        
        y = np.array([[0,0,1,1]]).T
        np.random.seed(1)
        
        syn0 = 2*np.random.random((3,1)) - 1
        
        for i in range(10000):
            l0 = X
            l1 = DoubleLayer.nonlin(np.dot(l0, syn0))
            l1_error = y - l1
            l1_delta = l1_error * DoubleLayer.nonlin(l1, True)
            syn0 += np.dot(l0.T, l1_delta)
        
        print('Output:')
        print(l1)
        
        # Output:
        # [[0.00966449]
        #  [0.00786506]
        #  [0.00786506]
        #  [0.99358898]
        #  [0.99211957]]

# Triple Layer Network
class TripleLayer():
    def nonlin(x , deriv = False):
        if deriv == True:
            return x*(1-x)
        return 1/(1+np.exp(-x))
    
    def Layer():
        
        X = np.array([[0,0,1],
                      [0,1,1],
                      [1,0,1],
                      [1,1,1]])
        
        y = np.array([[0,0,1,1]]).T
        
        np.random.seed(1)
        
        syn0 = 2*np.random.random((3,4)) - 1
        syn1 = 2*np.random.random((4,1)) - 1
        
        for i in range(60000):
            
            l0 = X
            l1 = TripleLayer.nonlin(np.dot(l0, syn0))
            l2 = TripleLayer.nonlin(np.dot(l1, syn1))
            
            l2_error = y - l2
            
            if (i%10000) == 0:
                print("Error:" + str(np.mean(np.abs(l2_error))))
            
            l2_delta = l2_error*TripleLayer.nonlin(l2,deriv=True)
            
            l1_error = l2_delta.dot(syn1.T)
            l1_delta = l1_error * TripleLayer.nonlin(l1, deriv=True)
            
            syn1 += l1.T.dot(l2_delta)
            syn0 += l0.T.dot(l1_delta)
            
            # Output:
            # Error:0.4685343254580603
            # Error:0.005002426725395315
            # Error:0.0034544054615330507
            # Error:0.0027865570196723556
            # Error:0.002394115505520921
            # Error:0.0021288852682254146
    
TripleLayer.Layer()
