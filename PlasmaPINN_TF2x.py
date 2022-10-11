# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 2022

@author: vgopakum, mathewsa

Rewriting the Plasma PINN code in Tensorflow 2.x 
"""

# %%
#Import functions 
import h5py
import time 
import os
import time
from tqdm import tqdm 
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import math

from pyDOE import lhs
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers, activations
from tensorflow.keras import backend as K 
import tensorflow_probability as tfp

save_directory = os.getcwd()

# %% 


#.Size of data set
nx0G = 256
ny0G = 128
nz0G = 32


#.Domain size
a0 = 0.22             #. minor radius (m)
R0 = 0.68 + a0        #. major radius (m)
Lx = 0.35             #. Radial size of simulation domain (normalized by a0)
Ly = 0.25             #. Vertical size of simulation domain (normalized by a0)
Lz = 20.0             #. Connection length (normalized by R0)

dt        = 0.000005  #.time step. (normalized)
nts       = 16000     #.number of time steps per frame.
preFrames = 0         #.time frames in previous job this one restarted from.
iFrame  = [0]         #.First frame to output (zero indexed).
fFrame  = [398]       #.Last frame to output.
t_start = 0.0 

len_frames = fFrame[0] - iFrame[0] + 1
t_end = t_start + nts*dt*(len_frames - 1)
x_t = np.linspace(t_start,t_end,len_frames) #dimensionless time
tmp = Lx
Lx = np.array([tmp, Ly, Lz, t_end])
tmp = nx0G
nx0G = np.array([tmp, ny0G, nz0G, len_frames])

#.Cell spacing
dx = np.array([Lx[0]/nx0G[0], Lx[1]/nx0G[1], Lx[2]/nx0G[2], dt]) 

x = [ (np.arange(0,nx0G[0])-nx0G[0]/2.0)*dx[0]+dx[0]/2.0, \
      (np.arange(0,nx0G[1])-nx0G[1]/2.0)*dx[1]+dx[1]/2.0, \
      (np.arange(0,nx0G[2])-nx0G[2]/2.0)*dx[2]+dx[2]/2.0, \
      x_t]

#normalized diffusion coefficients applied in the code
#note: implicit diffusion applies an additional dt factor
DiffX = 2.*np.pi/(dx[0]*3.)
DiffY = 2.*np.pi/(dx[1]*3.)
DiffZ = 2.*np.pi/(dx[2]*3.)

DiffX_norm = DiffX**2.
DiffY_norm = DiffY**2.
DiffZ_norm = DiffZ**2.

data_file = str(save_directory)+'/PlasmaPINN_data_inputs_paper.h5' 
h5f = h5py.File(data_file, "r")
x_x = np.asarray(h5f['x_x']) # x inputs
x_y = np.asarray(h5f['x_y']) # y inputs 
x_z = np.asarray(h5f['x_z'])  # z inputs
x_t = np.asarray(h5f['x_t'])  # t inputs
y_ne = np.asarray(h5f['y_den'])  #
y_Te = np.asarray(h5f['y_Te'])  #

init_weight_den = (1./np.median(np.abs(y_ne))) #??
init_weight_Te = (1./np.median(np.abs(y_Te)))   #??

frac_train = 1.0
N_train = int(frac_train*len(y_ne))
idx = np.random.choice(len(y_ne), N_train, replace=False)

x_train = x_x[idx,:]
y_train = x_y[idx,:]
z_train = x_z[idx,:]
t_train = x_t[idx,:]
v1_train = y_ne[idx,:] 
v5_train = y_Te[idx,:] 

sample_batch_size = int(5000)

# %%

class MLP(tf.keras.Model):

  def __init__(self):
    super().__init__()
    initializer = tf.keras.initializers.GlorotUniform()
    self.dense1 = tf.keras.layers.Dense(50, activation=tf.nn.tanh, kernel_initializer=initializer)
    self.dense2 = tf.keras.layers.Dense(50, activation=tf.nn.tanh, kernel_initializer=initializer)
    self.dense3 = tf.keras.layers.Dense(50, activation=tf.nn.tanh, kernel_initializer=initializer)
    self.dense4 = tf.keras.layers.Dense(50, activation=tf.nn.tanh, kernel_initializer=initializer)
    self.dense2 = tf.keras.layers.Dense(50, activation=tf.nn.tanh, kernel_initializer=initializer)
    self.output_layer = tf.keras.layers.Dense(1, kernel_initializer=initializer)

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    x = self.dense3(x)
    x = self.dense4(x)
    x = self.output_layer(x)

    return x


#Setting up the 5 Individual networks
nn_1, nn_2, nn_3, nn_4, nn_5 = MLP(), MLP(), MLP(), MLP(), MLP()

nn_1.build(input_shape=(None,3))
nn_2.build(input_shape=(None,3))
nn_3.build(input_shape=(None,3))
nn_4.build(input_shape=(None,3))
nn_5.build(input_shape=(None,3))


# %% 
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
z_train = tf.convert_to_tensor(z_train, dtype=tf.float32)
t_train = tf.convert_to_tensor(t_train, dtype=tf.float32)
v1_train = tf.convert_to_tensor(v1_train, dtype=tf.float32)
v5_train = tf.convert_to_tensor(v5_train, dtype=tf.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, t_train, v1_train, v5_train))
train_dataset = train_dataset.batch(batch_size=sample_batch_size)

X = np.concatenate([x_train, y_train, t_train], 1) 
lb = X.min(0)
ub = X.max(0)
# %%
#Loss Functions 

def reconstruction(model, X, Y):
    u = model(X)
    recon_loss = u-Y
    return tf.reduce_mean(tf.square(recon_loss))

@tf.function
def pde_loss(x,y,t):

        sample_size = len(x)
        mi_me = 3672.3036
        eta = 63.6094
        nSrcA = 20.0
        enerSrceA = 0.001 
        xSrc = -0.15
        sigSrc = 0.01
        B = (0.22+0.68)/(0.68 + 0.22 + a0*x)
        eps_R = 0.4889
        eps_v = 0.3496
        alpha_d = 0.0012
        tau_T = 1.0
        kappa_e = 7.6771
        kappa_i = 0.2184
        eps_G = 0.0550
        eps_Ge = 0.0005 

        v1 = nn_1(tf.concat([x,y,t], 1))
        v2 = nn_2(tf.concat([x,y,t], 1))
        v3 = nn_3(tf.concat([x,y,t], 1))
        v4 = nn_4(tf.concat([x,y,t], 1))
        v5 = nn_5(tf.concat([x,y,t], 1))

        PINN_v2 = v2
        PINN_v3 = v3
        PINN_v4 = v4 

        v1_t = tf.gradients(v1, t)[0]
        v1_x = tf.gradients(v1, x)[0]
        v1_y = tf.gradients(v1, y)[0]
        v2_x = tf.gradients(PINN_v2, x)[0]
        v2_y = tf.gradients(PINN_v2, y)[0]
        v5_t = tf.gradients(v5, t)[0]
        v5_x = tf.gradients(v5, x)[0]
        v5_y = tf.gradients(v5, y)[0] 
        pe = v1*v5
        pe_y = tf.gradients(pe, y)[0]
        jp = v1*((tau_T**0.5)*v4 - v3) 
        lnn = tf.math.log(v1)
        lnn_x = tf.gradients(lnn, x)[0]
        lnn_xx = tf.gradients(lnn_x, x)[0]
        lnn_xxx = tf.gradients(lnn_xx, x)[0]
        lnn_xxxx = tf.gradients(lnn_xxx, x)[0] 
        lnn_y = tf.gradients(lnn, y)[0]
        lnn_yy = tf.gradients(lnn_y, y)[0]
        lnn_yyy = tf.gradients(lnn_yy, y)[0]
        lnn_yyyy = tf.gradients(lnn_yyy, y)[0] 
        Dx_lnn = -((50./DiffX_norm)**2.)*lnn_xxxx
        Dy_lnn = -((50./DiffY_norm)**2.)*lnn_yyyy
        D_lnn = (Dx_lnn + Dy_lnn) 
        lnTe = tf.math.log(v5)
        lnTe_x = tf.gradients(lnTe, x)[0]
        lnTe_xx = tf.gradients(lnTe_x, x)[0]
        lnTe_xxx = tf.gradients(lnTe_xx, x)[0]
        lnTe_xxxx = tf.gradients(lnTe_xxx, x)[0]
        lnTe_y = tf.gradients(lnTe, y)[0]
        lnTe_yy = tf.gradients(lnTe_y, y)[0]
        lnTe_yyy = tf.gradients(lnTe_yy, y)[0]
        lnTe_yyyy = tf.gradients(lnTe_yyy, y)[0]
        Dx_lnTe = -((50./DiffX_norm)**2.)*lnTe_xxxx
        Dy_lnTe = -((50./DiffY_norm)**2.)*lnTe_yyyy
        D_lnTe = (Dx_lnTe + Dy_lnTe) 
        S_n = nSrcA*tf.exp(-(x - xSrc)*(x - xSrc)/(2.*sigSrc*sigSrc))
        S_Ee = enerSrceA*tf.exp(-(x - xSrc)*(x - xSrc)/(2.*sigSrc*sigSrc)) 
        cond1Sn = tf.greater(S_n[:,0], 0.01*tf.ones(sample_size)) 
        S_n = tf.where(cond1Sn, S_n[:,0], 0.001*tf.ones(sample_size))
        cond1SEe = tf.greater(S_Ee[:,0], 0.01*tf.ones(sample_size))
        S_Ee = tf.where(cond1SEe, S_Ee[:,0], 0.001*tf.ones(sample_size))
        cond2Sn = tf.greater(x, xSrc*tf.ones(sample_size))
        S_n = tf.where(cond2Sn[:,0], S_n, 0.5*tf.ones(sample_size))
        cond2SEe = tf.greater(x, xSrc*tf.ones(sample_size))
        S_Ee = tf.where(cond2SEe[:,0], S_Ee, 0.5*tf.ones(sample_size))
        cond4Sn = tf.greater(v1[:,0], 5.0*tf.ones(sample_size))
        S_n = tf.where(cond4Sn, 0.0*tf.ones(sample_size), S_n)
        cond4SEe = tf.greater(v5[:,0], 1.0*tf.ones(sample_size))
        S_Ee = tf.where(cond4SEe, 0.0*tf.ones(sample_size), S_Ee)
        f_v1 = v1_t + (1./B)*(v2_y*v1_x - v2_x*v1_y) - (-eps_R*(v1*v2_y - alpha_d*pe_y) + S_n + v1*D_lnn)
        f_v5 = v5_t + (1./B)*(v2_y*v5_x - v2_x*v5_y) - v5*(5.*eps_R*alpha_d*v5_y/3. +\
                (2./3.)*(-eps_R*(v2_y - alpha_d*pe_y/v1) +\
                (1./v1)*(0.71*eps_v*(0.0) + eta*jp*jp/(v5*mi_me))) +\
                (2./(3.*pe))*(S_Ee) + D_lnTe) 

        return v1, v5, PINN_v2, PINN_v3, PINN_v4, f_v1, f_v5

def mother_loss(x, y, t, v1_data, v5_data):
    v1, v5, PINN_v2, PINN_v3, PINN_v4, f_v1, f_v5 = pde_loss(x, y, t)

    loss1 = tf.reduce_mean(1.0*init_weight_den*tf.square(v1_data - v1))
    loss5 = tf.reduce_mean(1.0*init_weight_Te*tf.square(v5_data - v5))
    lossf1 = tf.reduce_mean(1.0*init_weight_den*tf.square(f_v1))
    lossf5 = tf.reduce_mean(1.0*init_weight_Te*tf.square(f_v5))

    return loss1, loss5, lossf1, lossf5


# %%
#Training Loop
optimizer_adam_v1 = tf.keras.optimizers.Adam(learning_rate = 1e-3, beta_1=.90)
optimizer_adam_v5 = tf.keras.optimizers.Adam(learning_rate = 1e-3, beta_1=.90)
optimizer_adam_f = tf.keras.optimizers.Adam(learning_rate = 1e-3, beta_1=.90)

it=0
epochs = 100 #Number of Epochs for the training loop. 
loss_list = []

start_time = time.time()
while it < epochs :
    for xx, yy, tt, ne, te in train_dataset:

        # Using Adam 
        with tf.GradientTape() as tape:
            loss1, loss5, lossf1, lossf5 = mother_loss(xx, yy, tt, ne, te)
            grads1 = tape.gradient(loss1, nn_1.trainable_variables)
            optimizer_adam_v1.apply_gradients(zip(grads1, nn_1.trainable_variables))

        with tf.GradientTape() as tape:
            loss1, loss5, lossf1, lossf5 = mother_loss(xx, yy, tt, ne, te)
            grads5 = tape.gradient(loss5, nn_5.trainable_variables)
            optimizer_adam_v5.apply_gradients(zip(grads1, nn_5.trainable_variables))

        with tf.GradientTape() as tape:
            loss1, loss5, lossf1, lossf5 = mother_loss(xx, yy, tt, ne, te)
            gradsf = tape.gradient(lossf1+lossf5, nn_2.trainable_variables)
            optimizer_adam_f.apply_gradients(zip(gradsf, nn_2.trainable_variables+nn_3.trainable_variables+nn_4.trainable_variables))    


        #Using L-FBGS
        # loss1, loss5, lossf1, lossf5 = mother_loss(xx, yy, tt, ne, te)
        # tfp.optimizer.lbfgs_minimize(loss1, nn_1.trainable_variables)
        # tfp.optimizer.lbfgs_minimize(loss5, nn_5.trainable_variables)
        # tfp.optimizer.lbfgs_minimize(lossf1+lossf5, nn_2.trainable_variables+nn_3.trainable_variables+nn_4.trainable_variables)

    it += 1

    print('It: %d, Loss1: %.3e, Loss5: %.3e, Loss_f1: %.3e, Loss_f5: %.3e' % (it, loss1, loss5, lossf1, lossf5))
    loss_list.append(loss1 + loss5 + lossf1 + lossf5)

train_time = time.time() - start_time
plt.plot(loss_list)
plt.xlabel('Iterations')
plt.ylabel('L2 Error')


# %%
