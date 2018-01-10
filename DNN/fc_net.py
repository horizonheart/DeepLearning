#coding=utf-8
from layer_utils import *
import numpy as np
class TwoLayerNet(object):   

    #构造函数，初始化神经网络模型
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,           
                              weight_scale=1e-3, reg=0.0):    
        """    
        Initialize a new network.   
        Inputs:    
        - input_dim: An integer giving the size of the input    
        - hidden_dim: An integer giving the size of the hidden layer    
        - num_classes: An integer giving the number of classes to classify    
        - dropout: Scalar between 0 and 1 giving dropout strength.    
        - weight_scale: Scalar giving the standard deviation for random 
                        initialization of the weights.    
        - reg: Scalar giving L2 regularization strength.    
        """    
        self.params = {}    
        self.reg = reg   #正则化因子
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)     
        self.params['b1'] = np.zeros((1, hidden_dim))    
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)  
        self.params['b2'] = np.zeros((1, num_classes))
       #计算该模型的损失函数
    def loss(self, X, y=None):    
        """   
        Compute loss and gradient for a minibatch of data.    
        Inputs:    
        - X: Array of input data of shape (N, d_1, ..., d_k)    
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].  
        Returns:   
        If y is None, then run a test-time forward pass of the model and return:    
        - scores: Array of shape (N, C) giving classification scores, where              
                  scores[i, c] is the classification score for X[i] and class c. 
        If y is not None, then run a training-time forward and backward pass and    
        return a tuple of:    
        - loss: Scalar value giving the loss   
        - grads: Dictionary with the same keys as self.params, mapping parameter             
                 names to gradients of the loss with respect to those parameters.    
        """
        scores = None
        N = X.shape[0]
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        # 第一次的前向传播 ,h1为神经网络第一层的输出值  cache1=((x, w, b), a) x为原始输入，w,b为第一层网络的参数，a为通过激活函数之前的值
        h1, cache1 = affine_relu_forward(X, W1, b1)
        #out为最后一次神经网络的输出，cache2=(h1, W2, b2)
        out, cache2 = affine_forward(h1, W2, b2)
        scores = out              # (N,C)
        # If y is None then we are in test mode so just return scores
        #如果y是空的，则说明是预测模型
        if y is None:   
            return scores

        loss, grads = 0, {}
        #将最后一层神经网络的输出进行softmax变换 data_loss代表的是损失函数  dscores代表的是损失函数关于softmax输出的偏导数
        data_loss, dscores = softmax_loss(scores, y)
        #正则项损失
        reg_loss = 0.5 * self.reg * np.sum(W1*W1) + 0.5 * self.reg * np.sum(W2*W2)
        #总的损失函数
        loss = data_loss + reg_loss

        # Backward pass: compute gradients
        #反向传播计算梯度
        # dh1, dW2, db2 损失函数关于当前神经的输出，权重的偏导数
        dh1, dW2, db2 = affine_backward(dscores, cache2)
        #cache1=((x, w, b), a) x为原始输入，w,b为第一层网络的参数，a为通过激活函数之前的值
        # dX, dW1, db1  损失函数关于当前神经的输出，权重的偏导数
        dX, dW1, db1 = affine_relu_backward(dh1, cache1)
        # Add the regularization gradient contribution
        dW2 += self.reg * W2  #W2=100*10
        dW1 += self.reg * W1  #W1=3072*100
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads