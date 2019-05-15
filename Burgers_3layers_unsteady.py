# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:27:23 2019

@author: HB65402
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 08:14:11 2017

@author: hb65402
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 08:34:07 2017

@author: hb65402
"""

import tensorflow as tf
import numpy as np
np.random.seed(0)
tf_random_seed=0

def Burgers():
    global np
    tf.reset_default_graph()
    #data = np.genfromtxt('Burgers.csv', delimiter=',', names=['Seq','X1','t1','X2','t2','X3','t3','X4','t4','Y'])
    data = np.genfromtxt('Outlier.csv', delimiter=',', names=['Seq','X1','t1','X3','t3','X4','t4','X2','t2','Y','YNorm','A','Asignal'])
    # Forming numpy arrays
    # Note: At Seq 23037, the JobTime is reset
    #t1 = np.array(data['JobTime']).astype(np.float64)[1:37525]
    #t2 = np.array(data['JobTime']).astype(np.float64)[37525:]
    #t2 = t2 + t1[t1.shape[0] - 1]
    #t = np.concatenate((t1, t2))
    seq = np.array(data['Seq']).astype(np.float64)[1:]
    x1 = np.asarray(data['X1']).astype(np.float)[1:]
    t1 = np.asarray(data['t1']).astype(np.float)[1:]
    x2 = np.asarray(data['X2']).astype(np.float)[1:]
    t2 = np.asarray(data['t2']).astype(np.float)[1:]       
    x3 = np.asarray(data['X3']).astype(np.float)[1:]
    t3 = np.asarray(data['t3']).astype(np.float)[1:]  
    x4 = np.asarray(data['X4']).astype(np.float)[1:]
    t4 = np.asarray(data['t4']).astype(np.float)[1:]                  
    a = np.asarray(data['A']).astype(np.float)[1:] 
    Y = np.asarray(data['Y']).astype(np.float)[1:]
    asignal = np.asarray(data['Asignal']).astype(np.float)[1:] 
    train_X = (np.c_[x1, t1])
    
    #train_X = (np.c_[x1, t1])
    train_Xbc2 = (np.c_[x2, t2])
    train_Xic =  (np.c_[x3, t3])
    train_Xbc1 = (np.c_[x4, t4])
    
    
    def getRangeNorm(v):
        vMax = np.max(v)
        vMin = np.min(v)
        vRange = vMax - vMin
        vNorm = (v - vMin) / vRange
        return vMax, vMin, vRange, vNorm  
    
    def getNormToOriginal(vNorm, vMin, vRange):
        return (vNorm * vRange + vMin)
    asmax, asmin, asrange, asnorm = getRangeNorm(asignal)   
    ymax, ymin, yrange, ynorm = getRangeNorm(Y)    
    train_Y = np.c_[ynorm]
    amax, amin, arange, anorm = getRangeNorm(a) 
    mean1 = np.mean(anorm)
    var1 = np.std(anorm)
    anorm_BN = (anorm- mean1) /np.sqrt(var1 + 1.e-8)
    train_a = np.c_[anorm]
    #train_X = np.asarray([[0.25,0.4],[0.25,0.6],[0.25,0.8],[0.25,1]])
    #train_Y = np.asarray([[0.01357],[0.00189],[0.00026],[0.00004]])
    #train_Xic = np.asarray([[0.25,0.],[0.4,0.],[0.6,0.],[0.8,0.]])
    #train_Xbc = numpy.asarray([[1,0.],[1,0.],[1,0.],[1,0]])
    nnodes1 = 50#55
    nnodes2 = 40#11
    beta = 0.0
    x = tf.placeholder("float",[None, 2])
    yac = tf.placeholder("float",[None, 1])
    a1 = tf.placeholder("float",[None, 1])
    W = tf.Variable(tf.zeros([2, 1]))
    b = tf.Variable(tf.zeros([1, 1]))
    
    W1 = tf.Variable(tf.zeros([2,nnodes1]))
    
    b1 = tf.Variable(tf.zeros([nnodes1]))
    
    W2 = tf.Variable(tf.zeros([nnodes1,nnodes2]))
    b2 = tf.Variable(tf.zeros([nnodes2]))
    
    W3 = tf.Variable(tf.zeros([nnodes2,1]))
    b3 = tf.Variable(tf.zeros([1]))
    
    
    #init = tf.global_variables_initializer()
    
    layer1 = tf.nn.sigmoid(tf.matmul(x,W1) + b1)
    layer2 = tf.nn.sigmoid(tf.matmul(layer1,W2) + b2)
    y = tf.matmul(layer2,W3) + b3
    
    layerdert = tf.matmul(layer1*(1 - layer1)*W1[1],W2)
    layer = layerdert*layer2*(1 - layer2)
    yt = tf.matmul(layer,W3)  
    #
    layerderx = tf.matmul(layer1*(1 - layer1)*W1[0],W2)
    layer = layerderx*layer2*(1 - layer2)
    yx = tf.matmul(layer,W3) 
    #
    layer = tf.matmul(layer1*(1 - layer1)*layer1*(1-layer1)*W1[0]*W1[0],W2*W2)
    layerderxx1 = layer * (layer2*(1 - layer2) - 2*layer2*layer2*(1 - layer2))
    layer = tf.matmul((layer1*(1 - layer1) - 2*layer1*layer1*(1 - layer1))*W1[0]*W1[0],W2)
    layerderxx2 = layer*layer2*(1 - layer2)
    layerderxx = layerderxx1 + layerderxx2
    yxx = tf.matmul(layerderxx,W3) 
    #
    #
    xic = tf.placeholder("float",[None, 2])
    layer1ic = tf.nn.sigmoid(tf.matmul(xic,W1) + b1)
    layer2ic = tf.nn.sigmoid(tf.matmul(layer1ic,W2) + b2)
    yic = tf.matmul(layer2ic,W3) + b3 
    #
    xbc1 = tf.placeholder("float",[None, 2])
    layer1bc = tf.nn.sigmoid(tf.matmul(xbc1,W1) + b1)
    layer2bc = tf.nn.sigmoid(tf.matmul(layer1bc,W2) + b2)
    ybc1 = tf.matmul(layer2bc,W3) + b3
    #
    xbc2 = tf.placeholder("float",[None, 2])
    layer1bc = tf.nn.sigmoid(tf.matmul(xbc2,W1) + b1)
    layer2bc = tf.nn.sigmoid(tf.matmul(layer1bc,W2) + b2)
    ybc2 = tf.matmul(layer2bc,W3) + b3
    
    
    #x1 = 1+tf.exp(-(W3[0]*x[0]+b3))
    
    #x2 = 1/x1-tf.sin(3.14*x[0])
    #x3 = 1+tf.exp(-(W3[1]*x[1]+b))
    #x4 = 1+tf.exp(-(W3[0]+b+W3[1]*x[1]))
    #activation = tf.nn.sigmoid(tf.matmul(x, W)+b)
    #cost = (tf.square(activation *(1-activation)*-W[1]+activation*activation *(1-activation)*-W[0]
                                   #+1.0*W[0]*(1-2*activation)))+x2*x2+1/(x3*x3)+1/(x4*x4)
    #cost = tf.square(yt-y*yx-yxx)+tf.square(yic)+tf.square(ybc1)+tf.square(ybc2)
    regularizers = tf.nn.l2_loss(W1)
    cost = tf.reduce_mean(tf.square(yt+y*yx-1*yxx)+tf.square(yic-yac)+tf.square(y-a1))
    #cost = tf.reduce_mean(tf.square(yt+y*yx-1*yxx)+tf.square(yic-yac))
    #cost = tf.reduce_mean(tf.square(y-a1))
    #cost = tf.reduce_mean(cost + beta * regularizers)
    #cost_old = tf.Variable(0.,trainable=False)
    
    
    #cost = tf.reduce_mean(tf.square(yx)+tf.square(ybc1-1)+tf.square(ybc2-1))
    #+(1./(1.+ tf.exp(-(W[0]*x[0]+b)))-tf.sin(3.14*x[0]))^2
    #+tf.square(tf.nn.sigmoid(W[0]*x[0]+b[0]))+
    #tf.square(1./(1.+ exp(-(W[0]*x[0]+b[0]))
    step = tf.Variable(0,trainable=False)
    rate = tf.train.exponential_decay(0.02, step, 200, 0.9999,staircase=True)
    #rate = tf.Variable(0.01, trainable=False)
    
    #if tf.greater(1, 0) is not None:
    #   rate = rate*1.09 
    #else:
    #    rate= rate*0.99
    #    
    #cost_old = cost
    #rate = tf.cond(tf.greater(cost_old, cost), rate*1.09, rate*0.99)
    #tf.summary.scalar('learning_rate', rate)
    #optimizer = tf.train.AdamOptimizer(rate,epsilon=1.e-5).minimize(cost, global_step = step)
    
    opt_func=tf.train.MomentumOptimizer(rate,0.99,use_nesterov=True)
    tvars = tf.trainable_variables()
    #grads,_=tf.clip_by_global_norm(tf.gradients(cost,tvars),2)
    grads = opt_func.compute_gradients(cost,tvars)
    #clipgradsold= tf.gradients(cost,tvars)[0]
    noise = tf.random_normal(shape = tf.shape(W1),mean=0,stddev=0.01)
    def NoiseIfNotnone(grad):
            if grad is None:
                return grad
            else:
                return tf.transpose(grad[0]) + noise
    #clipgrads = [(NoiseIfNotnone(g),v) for g,v in grads]
    #grads = tf.add_scaled_noise_to_gradients(grads,0.001)
    def ClipIfNotnone(grad):
            if grad is None:
                return grad
            else:
                return tf.clip_by_value(grad,-1,1.)
    clipgrads = [(ClipIfNotnone(g),v) for g,v in grads]
    
    def MultIfNotnone(grad,clip):
            if clip is None:
                return grad            
            else:
                if tf.greater(grad*clip,0) is not None:
                    return grad*1000
                else:
                    return grad*0.1
                    
    #if tf.greater(step,0)  is not None:          
        #clipgrads = [(MultIfNotnone(g,clipgradsold),v) for g,v in grads]
        #optimizer = opt_func.apply_gradients(zip(tf.clip_by_globalnorm(grads,2),global_step = step))
    optimizer = opt_func.apply_gradients(clipgrads,global_step = step)
    
    #rate = (tvarsnew-tvarsold)/(gradnew-gradold)
    #optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
    #optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)
    init = tf.initialize_all_variables()
    errors = []
    with tf.Session() as sess:
        sess.run(init)
        batch_size = train_X.shape[0]
        epoch_size = 10000
    
        for i in range(epoch_size):
            for k in range(0,train_X.shape[0],batch_size): 
               if k % batch_size == 0: 
    
            #train_data = sess.run(optimizer, feed_dict={x: train_X, xbc2:train_Xbc2, xic:train_Xic, xbc1:train_Xbc1})
                 _, error_value = sess.run([optimizer, cost], feed_dict={x: train_X[k:(k + batch_size)], yac:train_Y[k:(k + batch_size)], xic:train_Xic[k:(k + batch_size)], xbc1:train_Xbc1[k:(k + batch_size)], xbc2:train_Xbc2[k:(k + batch_size)], a1:train_a[k:(k + batch_size)]})
                 errors.append(error_value)
                 result = sess.run(y, feed_dict={x:train_X})
             
             #weights = sess.run(tf.transpose(W1))
                 resultn = getNormToOriginal(np.array(result), ymin, yrange)
             #np.savetxt("result.csv",np.array(result))
        np.savetxt("cost.csv",np.array(errors))
    import numpy as np
    from math import exp as exp
    import matplotlib.pyplot as plt
    u_analytical = np.zeros(len(result))
    NU = 0.01
    plt.figure()
    ax=plt.subplot(111)
       # Analytical Solution
    for n in range(0,len(result)):
           
               phi = exp( -(x1[n]-4*t1[n])**2/(4*NU*(t1[n]+1)) ) + exp( -(x1[n]-4*t1[n]-2*3.14)**2/(4*NU*(t1[n]+1)) )
    
               dphi = ( -0.5*(x1[n]-4*t1[n])/(NU*(t1[n]+1))*exp( -(x1[n]-4*t1[n])**2/(4*NU*(t1[n]+1)) )
                   -0.5*(x1[n]-4*t1[n]-2*3.14)/(NU*(t1[n]+1))*exp( -(x1[n]-4*t1[n]-2*3.14)**2/(4*NU*(t1[n]+1)) ) )
    
               u_analytical[n] = -2*NU*(dphi/phi) + 4
               u_analytical[n] = 2*x1[n]/(1+2*t1[n])
               #umin, umax, urange, unorm = getRangeNorm(u_analytical)
    ax.plot(x1[len(result)-4:len(result)],resultn[len(result)-4:len(result)],'ko', markerfacecolor='none', alpha=0.5, label=' numerical')
    ax.plot(x1[len(result)-4:len(result)],u_analytical[len(result)-4:len(result)],linestyle='-',label=' analytical')
    ax.legend( bbox_to_anchor=(1.02,1), loc=2)
       
    plt.xlabel('x ')
    plt.ylabel('u ')
    plt.ylim([0.0,2.0])
    plt.xlim([0,1.0])
    plt.show()
    np.savetxt("result.csv",np.c_[anorm,np.array(result),asnorm],fmt=['%0.5f','%0.5f','%0.5f'],delimiter=',',header="Actual,Predictions,Signal",comments='')
    #weights = np.zeros((2,len(result)))
    
    #weights = tf.get_variable("W1",shape = [2,len(result)],resuse=True)
    #np.savetxt("weights.csv",weights)
    del seq, x1, t1, x2, t2, x3, t3, x4, t4, Y
    sess.close()