#!/usr/bin/env python
# coding: utf-8

# # Coregionalized Regression Model (vector-valued regression)

# ### updated: 17th June 2015

# ### by Ricardo Andrade-Pacheco

# This tutorial will focus on the use and kernel selection of the $\color{firebrick}{\textbf{coregionalized regression}}$ model in GPy.

# ## Setup

# The first thing to do is to set the plots to be interactive and to import GPy.

# In[1]:


import pylab as pb
import numpy as np
import GPy

#This functions generate data corresponding to two outputs
f_output1 = lambda x: 1. * np.sign(np.cos(x/5.)) + np.random.rand(x.size)[:,None] * 0.1
f_output2 = lambda x: -0.1 + np.random.rand(x.size)[:,None] * 0.01


#{X,Y} training set for each output
X1 = np.random.rand(100)[:,None]; X1=X1*75
X2 = np.random.rand(100)[:,None]; X2=X2*70 + 30
Y1 = f_output1(X1)
Y2 = f_output2(X2)
#{X,Y} test set for each output
Xt1 = np.random.rand(100)[:,None]*100
Xt2 = np.random.rand(100)[:,None]*100
Yt1 = f_output1(Xt1)
Yt2 = f_output2(Xt2)


# We will also define a function that will be used later for plotting our results.
def plot_2outputs(m,xlim,ylim):
    fig = pb.figure(figsize=(12,8))
    #Output 1
    ax1 = fig.add_subplot(211)
    ax1.set_xlim(xlim)
    ax1.set_title('Output 1')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,0)],which_data_rows=slice(0,100),ax=ax1)
    ax1.plot(Xt1[:,:1],Yt1,'rx',mew=1.5)
    #Output 2
    ax2 = fig.add_subplot(212)
    ax2.set_xlim(xlim)
    ax2.set_title('Output 2')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,1)],which_data_rows=slice(100,200),ax=ax2)
    ax2.plot(Xt2[:,:1],Yt2,'rx',mew=1.5)

# Once we have defined an appropiate kernel for our model, its use is straightforward. In the next example we will use a

K = GPy.kern.RBF(input_dim=1) + GPy.kern.Brownian(input_dim=1)
icm = GPy.util.multioutput.ICM(input_dim=1,num_outputs=2,kernel=K)

m = GPy.models.GPCoregionalizedRegression([X1,X2],[Y1,Y2],kernel=icm)
m.optimize()
print m
plot_2outputs(m,xlim=(0,100),ylim=(-1,1))

pb.show()

# predicting future values
newX = np.arange(100,101)[:,None]
newX = np.hstack([newX,np.ones_like(newX)])
noise_dict = {'output_index':newX[:,1:].astype(int)}

print(m.predict(newX, Y_metadata=noise_dict, full_cov=True))
