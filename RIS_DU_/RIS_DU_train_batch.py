# -*- coding: utf-8 -*-
"""
Created on Fri Jun. 9 01:30:58 2023

@author: Benson
"""

import pickle
import math as m
import numpy as np
np.object = object
np.int = int
np.bool = bool
import scipy as sci
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda, Conv2D, ReLU, Flatten, LeakyReLU, Layer
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau,EarlyStopping,TensorBoard
from functools import partial
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model 
import os
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

for gpu in gpus:

    tf.config.experimental.set_memory_growth(gpu, True)
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   



# outer(Io) / Inner(In) iteration
Io = 6
In = 4

# parameter
ULA = 'ULA' 
USPA = 'USPA'

# ==================== Tuning Parameter ====================

Ns = 2
Nrf_t = 4
Nrf_r = 4 # original_Nrf_r = 4 
Nt = 32
Nr = 32
N_phi = 64 # the number of RIS (RIS_dim = N_phi*N_phi)
Nk = 16 # number of subcarrier

learning_rate = 0.001 
batch_size = 32 # 32
epochs = 30 # 30

# =========================================================


Mt = int(Nt/Nrf_t)
Mr = int(Nr/Nrf_r)
n_power = 0.1 

# The dimension of parameter per data
Frf_shape = (Nt,Nrf_t)
Wrf_shape = (Nr,Nrf_r)
Wbb_shape = (Nrf_r,Ns,Nk)
Lam_shape = (Ns,Ns,Nk)
H1_shape = (N_phi,Nt,Nk)
H2_shape = (Nr,N_phi,Nk)
np_shape = (1,)
Phi_shape = (N_phi,N_phi)


validation_splite = 0.1


A = np.ones((Mt,1))
B = np.ones((Mr,1))
P_F = np.zeros((Nt,Nrf_t))
P_W = np.zeros((Nr,Nrf_r))
P_RIS = np.zeros((N_phi,N_phi))
P_RIS = np.eye(N_phi)

for i in range(Nrf_t):
    P_F[i*Mt:(i+1)*Mt,i] = A[:,0]
    
for i in range(Nrf_r):
    P_W[i*Mr:(i+1)*Mr,i] = B[:,0]
    
# ---------------------------------------------------------------------------
#   load training data
# ---------------------------------------------------------------------------
def load_data(filename):
    
    # ---------------------------------------------------------------------------
    #   load training data
    # ---------------------------------------------------------------------------
    
    def load_mat_file(filename):
        
        training_data = sio.loadmat('.\\sparse_SV_channel_RIS\\training_data_s\\'+(filename.numpy()).decode())
      
        
        training_H1 = training_data['H1']
        training_H2 = training_data['H2']
        training_Frf = np.exp(1j*np.random.uniform(0,2*np.pi,(Nt,Nrf_t)))*P_F
        training_Wrf = np.exp(1j*np.random.uniform(0,2*np.pi,(Nr,Nrf_r)))*P_W
        training_Wbb = np.random.randn(Nrf_r,Ns,Nk) + 1j*np.random.randn(Nrf_r,Ns,Nk)
        training_Lam = np.zeros((Ns,Ns,Nk))
        training_n_power = np.ones((1,))*n_power
        
        phi = np.exp(1j*np.random.uniform(0,2*np.pi,(N_phi))) # phi is the vector / Phi is the matrix
        training_Phi = np.diag(phi)
        
        for i in range(Nk):
            training_Lam[:,:,i] = np.eye(Ns)
        
        
        r = np.zeros((1,))
        training_H1 = training_H1.astype('complex64')
        training_H2 = training_H2.astype('complex64')
        training_Frf = training_Frf.astype('complex64')
        training_Wrf = training_Wrf.astype('complex64')
        training_Wbb = training_Wbb.astype('complex64')
        training_Phi = training_Phi.astype('complex64')
        training_Lam = training_Lam.astype('float32')
        training_n_power = training_n_power.astype('float32')

        return training_Frf,training_Wrf,training_Wbb,training_Lam,training_H1,training_H2,training_Phi,training_n_power,r
    
    training_Frf,training_Wrf,training_Wbb,training_Lam,training_H1,training_H2,training_Phi,training_n_power,train_output = tf.py_function(load_mat_file,[filename],(tf.complex64,tf.complex64,tf.complex64,tf.complex64,tf.complex64,tf.complex64,tf.complex64,tf.float32,tf.float32))
    
    training_Frf.set_shape([Nt,Nrf_t])
    training_Wrf.set_shape([Nr,Nrf_r])
    training_Wbb.set_shape([Nrf_r,Ns,Nk])
    training_Lam.set_shape([Ns,Ns,Nk])
    training_H1.set_shape([N_phi,Nt,Nk])
    training_H2.set_shape([Nr,N_phi,Nk])
    training_Phi.set_shape([N_phi,N_phi])
    training_n_power.set_shape([1,])

    train_output.set_shape([1,])
    return {'Frf_init':training_Frf,'Wrf_init':training_Wrf,'Wbb_init':training_Wbb,'Lambda_init':training_Lam,'channel_1_matrix':training_H1,'channel_2_matrix':training_H2,'Phi_init':training_Phi,'noise_power':training_n_power},{'loss':train_output}
  
AUTOTUNE = tf.data.experimental.AUTOTUNE

filenames = os.listdir('.\\sparse_SV_channel_RIS\\training_data_s\\') # data path list

# number of training & validation data

N_train = int(len(filenames)*(1-validation_splite))
N_val = len(filenames)-N_train

# splite the data into training & testing
train_filenames = filenames[:N_train]
val_filenames = filenames[N_train:N_train+N_val]


# training dataset
train_path_dataset = tf.data.Dataset.from_tensor_slices(train_filenames) 
train_dataset = train_path_dataset.map(load_data,AUTOTUNE)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(AUTOTUNE)

# testing dataset
val_path_dataset = tf.data.Dataset.from_tensor_slices(val_filenames)
val_dataset = val_path_dataset.map(load_data,AUTOTUNE)
val_dataset = val_dataset.cache()
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(AUTOTUNE)


print('data loaded') 

    
# utility function
def loss_function(temp):

    Frf,Fbb,Wrf,Wbb,H1,H2,Phi,n_power,Lam_stack=temp
    SE_temp = []

    
    for k in range(Nk):
        
        Ft = Frf@Fbb[:,k,:,:] 
        Ft_h = tf.transpose(Ft,perm=[0,2,1],conjugate=True)
        
        Wt = Wrf@Wbb[:,k,:,:]
        Wt_h = tf.transpose(Wt,perm=[0,2,1],conjugate=True)
        
        h1 = H1[:,:,:,k]                                          # H1(realization,N_phi,Nt,Nk)     
        h2 = H2[:,:,:,k]                                          # H2(realization,Nr,N_phi,Nk)
        
        H_eff_k = h2@Phi@h1                                       # H_eff(realization,Nr,Nt)   
        H_eff_k_h = tf.transpose(H_eff_k,perm=[0,2,1],conjugate=True)                                     # H_eff(realization,Nr,Nt)   
        
        
        re_Wt = tf.math.real(Wt)
        im_wt = tf.math.imag(Wt)
        
        r0  = tf.linalg.pinv(re_Wt) @ im_wt
        y11 = tf.linalg.pinv(im_wt @ r0 + re_Wt)
        y10 = -r0 @ y11
        
        pinv_Wt = tf.cast(tf.complex(y11,y10), dtype = Wt.dtype)
        temp = tf.linalg.det(tf.eye(Ns,dtype=tf.complex64) + pinv_Wt@H_eff_k@Ft@Ft_h@H_eff_k_h@Wt/n_power)
 
        
        SE_k = tf.math.log(temp)/tf.cast(tf.math.log(2.0),dtype = tf.complex64)
        SE_temp.append(SE_k)
        
    SE_stack = tf.stack(SE_temp)
    SE = tf.reduce_sum(SE_stack,axis=0)
    SE = SE / Nk
    loss = tf.cast(-tf.math.real(SE),dtype = tf.float32)
    
    return loss

# subclass layer
# ------------------------------ MO (precoder) ------------------------------- #

class DUP_block(Layer):
    def __init__(self):
        super(DUP_block,self).__init__()
        self.bn = BatchNormalization()
        self.den = Dense(1,activation=partial(tf.nn.leaky_relu,alpha=0.1))
    def call(self,inputs):
        Frf,Beta,G_tilde_h,Lam = inputs
        
        selection_matrix = tf.constant(np.ones((Nrf_t,1)),dtype=tf.complex64)
        
        Frf_p = Frf@selection_matrix
        Frf_p_vec = tf.reshape(Frf_p,(tf.shape(Frf_p)[0],Nt))  # get the value of the matrix (vector form)
        Frf_p_vec_I = tf.math.real(Frf_p_vec)
        Frf_p_vec_Q = tf.math.imag(Frf_p_vec)
        
        Frf_p_vec_IQ = tf.concat([Frf_p_vec_I,Frf_p_vec_Q],axis=1)     
        temp = self.bn(Frf_p_vec_IQ)  
        step_size = self.den(temp)
       
        ##================
        # Frf grad update
        #=================
        
        Frf_h = tf.transpose(Frf,perm=[0,2,1],conjugate=True)
    
        P_mask = tf.constant(P_F,dtype=tf.complex64)
        
        # calculate the euclidean gradient
    
        eg_temp = []
        for k in range(Nk):
            Lam_k = Lam[k]
            G_tilde_k_h = G_tilde_h[k]
            G_tilde_k = tf.transpose(G_tilde_k_h,perm=[0,2,1],conjugate=True)
            
            Beta_k = Beta[k] # shape = (?,1,1)
            
            M = tf.linalg.inv(Lam_k) + G_tilde_k@Frf@Frf_h@G_tilde_k_h/Beta_k       
            M_inv = tf.linalg.inv(M)
            eg_temp.append(G_tilde_k_h@M_inv@M_inv@G_tilde_k@Frf/Beta_k)
            
        eg_stack = tf.stack(eg_temp)     
        eg = tf.reduce_sum(eg_stack,axis=0)     
        eg = -eg/Nk*P_mask
        
        # calculate the Riemannian gradient
        projection_len = tf.cast(tf.math.real(eg*tf.math.conj(Frf)),dtype = tf.complex64)
        rg = eg - projection_len*Frf
        
        # update the point
        step_size = tf.cast(step_size[:,:,tf.newaxis],tf.complex64)
        norm_rg = tf.norm(rg,ord='fro',axis=[1,2])
        norm_rg = norm_rg[:,tf.newaxis,tf.newaxis]
        Frf_new = Frf - step_size*rg/norm_rg
        
        # retraction
        Frf_new_abs = tf.abs(Frf_new)
        Frf_constrained = Frf_new/tf.cast((tf.nn.relu(Frf_new_abs-1)+1),dtype = tf.complex64)
             
        return Frf_constrained
    
class MO_P(Layer):
    def __init__(self):
        super(MO_P, self).__init__()
        self.DUP_1 = DUP_block()
        self.DUP_2 = DUP_block()
        self.DUP_3 = DUP_block()
        self.DUP_4 = DUP_block()
        # self.DUP_5 = DUP_block()
        
        # self.DUP_6 = DUP_block()
        # self.DUP_7 = DUP_block()
        # self.DUP_8 = DUP_block()
        # self.DUP_9 = DUP_block()
        # self.DUP_10 = DUP_block()
        
    def call(self,inputs):
        Frf,Wrf,Wbb,Lam,H1,H2,Phi,n_power = inputs
        
        Beta = []
        G_tilde_h = []
         
        for k in range(Nk):
            Lam_k = Lam[k]
            Wbb_k = Wbb[k]
            Wbb_k_h = tf.transpose(Wbb_k,perm=[0,2,1],conjugate=True)
            
            Beta_k = n_power[:,0,0]*Nt*Nr*tf.linalg.trace(Lam_k@Wbb_k_h@Wbb_k)/Nrf_t/Nrf_r
            Beta_k = Beta_k[:,tf.newaxis,tf.newaxis] # shape = (?,1,1)
                       
            h1 = H1[:,:,:,k]                                          # H1(realization,N_phi,Nt,Nk)            
            h2 = H2[:,:,:,k]                                          # H2(realization,Nr,N_phi,Nk)        
            
            H_eff_k = h2@Phi@h1 # H_eff(realization,Nr,Nt)
                       
            H_eff_k_h = tf.transpose(H_eff_k,perm=[0,2,1],conjugate=True) # H_eff_h(realization,Nt,Nr)
            
            G_tilde_k_h = H_eff_k_h@Wrf@Wbb_k
            
            Beta.append(Beta_k)
            G_tilde_h.append(G_tilde_k_h)
            
        ## MO 
        Frf = self.DUP_1([Frf,Beta,G_tilde_h,Lam[:]])
        Frf = self.DUP_2([Frf,Beta,G_tilde_h,Lam[:]])
        Frf = self.DUP_3([Frf,Beta,G_tilde_h,Lam[:]])
        Frf = self.DUP_4([Frf,Beta,G_tilde_h,Lam[:]])
        # Frf = self.DUP_5([Frf,Beta,G_tilde_h,Lam[:]])
        
        # Frf = self.DUP_6([Frf,Beta,G_tilde_h,Lam[:]])
        # Frf = self.DUP_7([Frf,Beta,G_tilde_h,Lam[:]])
        # Frf = self.DUP_8([Frf,Beta,G_tilde_h,Lam[:]])
        # Frf = self.DUP_9([Frf,Beta,G_tilde_h,Lam[:]])
        # Frf = self.DUP_10([Frf,Beta,G_tilde_h,Lam[:]])
        
        # Frf = FRF 
        
        return Frf,Beta,G_tilde_h
    
# ----------------------------- WMMSE (Precoder) ----------------------------- #    

class BB_P(Layer):
        
    def call(self,inputs):
        Frf,Beta,G_tilde_h,Lam = inputs
       
        Frf_h = tf.transpose(Frf,perm=[0,2,1],conjugate=True)
        xi = []
        Fbb = []
        for k in range(Nk):
            Lam_k = Lam[k]
            Beta_k = Beta[k] # shape = (?,1,1)
           
            G_tilde_k_h =  G_tilde_h[k]
            G_tilde_k = tf.transpose(G_tilde_k_h,perm=[0,2,1],conjugate=True)
            
            F_tilde_k = Frf_h@G_tilde_k_h@Lam_k@G_tilde_k@Frf + Beta_k*tf.eye(Nrf_t,dtype=tf.complex64)
            xi_k = 1/np.sqrt(Nt/Nrf_t/Ns)/tf.norm(tf.linalg.inv(F_tilde_k)@Frf_h@G_tilde_k_h@Lam_k,ord='fro',axis=[1,2]) # shape = (?,)
            xi_k = xi_k[:,tf.newaxis,tf.newaxis] # shape = (?,1,1)
            
            Fbb_k = xi_k*tf.linalg.inv(F_tilde_k)@Frf_h@G_tilde_k_h@Lam_k
            
            xi.append(xi_k)
            Fbb.append(Fbb_k)
        return Fbb,xi
    
# -------------------------------- MO (RIS) ---------------------------------- # 

class DURIS_block(Layer):
    def __init__(self):
        super(DURIS_block,self).__init__()
        self.bn = BatchNormalization()
        self.den = Dense(1,activation=partial(tf.nn.leaky_relu,alpha=0.1))
    def call(self,inputs):
        Phi,Beta,G_hat_h,Lam,C = inputs
        
        selection_matrix = tf.constant(np.ones((N_phi,1)),dtype=tf.complex64)
        
        Phi_p = Phi@selection_matrix
        Phi_p_vec = tf.reshape(Phi_p,(tf.shape(Phi_p)[0],N_phi))     
        Phi_p_vec_I = tf.math.real(Phi_p_vec)
        Phi_p_vec_Q = tf.math.imag(Phi_p_vec)
      
        Phi_p_vec_IQ = tf.concat([Phi_p_vec_I,Phi_p_vec_Q],axis=1)
                
        temp = self.bn(Phi_p_vec_IQ)
        step_size = self.den(temp)
      
        ##================
        # Frf grad update
        #=================
        Phi_h = tf.transpose(Phi,perm=[0,2,1],conjugate=True)
    
        P_mask = tf.constant(P_RIS,dtype=tf.complex64)
        
        # calculate the euclidean gradient
    
        eg_temp = []
        
        for k in range(Nk):
            Lam_k = Lam[k]
            G_hat_k_h = G_hat_h[k]
            G_hat_k = tf.transpose(G_hat_k_h,perm=[0,2,1],conjugate=True)
            C_k = C[k]
            Beta_k = Beta[k] # shape = (?,1,1)
          
            Q = tf.linalg.inv(Lam_k) + G_hat_k@Phi@C_k@Phi_h@G_hat_k_h/Beta_k 
            Q_inv = tf.linalg.inv(Q)
            eg_temp.append(G_hat_k_h@Q_inv@Q_inv@G_hat_k@Phi@C_k/Beta_k)
        eg_stack = tf.stack(eg_temp)     
        eg = tf.reduce_sum(eg_stack,axis=0)       
        eg = -eg/Nk*P_mask
        
        # calculate the Riemannian gradient
        projection_len = tf.cast(tf.math.real(eg*tf.math.conj(Phi)),dtype = tf.complex64)
        rg = eg - projection_len*Phi
        
        # update the point
        step_size = tf.cast(step_size[:,:,tf.newaxis],tf.complex64)
        norm_rg = tf.norm(rg,ord='fro',axis=[1,2])
        norm_rg = norm_rg[:,tf.newaxis,tf.newaxis]
        Phi_new = Phi - step_size*rg/norm_rg
        
        # retraction
        Phi_new_abs = tf.abs(Phi_new)
        Phi_constrained = Phi_new/tf.cast((tf.nn.relu(Phi_new_abs-1)+1),dtype = tf.complex64)
 
        return Phi_constrained
    
class MO_RIS(Layer):
    def __init__(self):
        super(MO_RIS, self).__init__()
        self.DURIS_1 = DURIS_block()
        self.DURIS_2 = DURIS_block()
        self.DURIS_3 = DURIS_block()
        self.DURIS_4 = DURIS_block()
        # self.DURIS_5 = DURIS_block()
        
        # self.DURIS_6 = DURIS_block()
        # self.DURIS_7 = DURIS_block()
        # self.DURIS_8 = DURIS_block()
        # self.DURIS_9 = DURIS_block()
        # self.DURIS_10 = DURIS_block()
    def call(self,inputs):
        Frf,Wrf,Wbb,Lam,H1,H2,Phi,Beta = inputs
        
        # PHI = Phi
        
        G_hat_h = []
        C = []       # C(N_phi,N_phi,Nk)
        Frf_h = tf.transpose(Frf,perm=[0,2,1],conjugate=True)
        Wrf_h = tf.transpose(Wrf,perm=[0,2,1],conjugate=True)
        
        for k in range(Nk):
           
            Wbb_k = Wbb[k]
            
            h1 = H1[:,:,:,k] # H1(realization,N_phi,Nt,Nk)        
            h1_h = tf.transpose(h1,perm=[0,2,1],conjugate=True)
            h2 = H2[:,:,:,k] # H2(realization,Nr,N_phi,Nk)
            h2_h = tf.transpose(h2,perm=[0,2,1],conjugate=True)
            
            
            # The first dimension is N_realization 
            G_hat_k_h = h2_h@Wrf@Wbb_k
            G_hat_h.append(G_hat_k_h)
            ## C !!!
            C_k = h1@Frf@Frf_h@h1_h
            C.append(C_k)    
            
                      
        ## MO 
        Phi = self.DURIS_1([Phi,Beta,G_hat_h,Lam[:],C])
        Phi = self.DURIS_2([Phi,Beta,G_hat_h,Lam[:],C])
        Phi = self.DURIS_3([Phi,Beta,G_hat_h,Lam[:],C])
        Phi = self.DURIS_4([Phi,Beta,G_hat_h,Lam[:],C])
        # Phi = self.DURIS_5([Phi,Beta,G_hat_h,Lam[:],C])
        
        # Phi = self.DURIS_6([Phi,Beta,G_hat_h,Lam[:],C])
        # Phi = self.DURIS_7([Phi,Beta,G_hat_h,Lam[:],C])
        # Phi = self.DURIS_8([Phi,Beta,G_hat_h,Lam[:],C])
        # Phi = self.DURIS_9([Phi,Beta,G_hat_h,Lam[:],C])
        # Phi = self.DURIS_10([Phi,Beta,G_hat_h,Lam[:],C])
        
        # Phi = PHI
        
        return Phi

# ------------------------------ MO (Combiner) ------------------------------- #        
class DUC_block(Layer):
    def __init__(self):
        super(DUC_block,self).__init__()
        self.bn = BatchNormalization()
        self.den = Dense(1,activation=partial(tf.nn.leaky_relu,alpha=0.1))
    def call(self,inputs):
        Wrf,Alpha,G,Lam = inputs
        
        selection_matrix = tf.constant(np.ones((Nrf_r,1)),dtype=tf.complex64)
        Wrf_p = Wrf@selection_matrix
        Wrf_p_vec = tf.reshape(Wrf_p,(tf.shape(Wrf_p)[0],Nr))     
        Wrf_p_vec_I = tf.math.real(Wrf_p_vec)
        Wrf_p_vec_Q = tf.math.imag(Wrf_p_vec)
        # theta = tf.atan(Wrf_p_vec_Q/Wrf_p_vec_I)
        Wrf_p_vec_IQ = tf.concat([Wrf_p_vec_I,Wrf_p_vec_Q],axis=1)
                
        temp = self.bn(Wrf_p_vec_IQ)
        step_size = self.den(temp)
        # print('[Side Effect] Retracing graph Wrf')        
        Wrf_h = tf.transpose(Wrf,perm=[0,2,1],conjugate=True)
        
        P_mask = tf.constant(P_W,dtype=tf.complex64)
        
        # calculate the euclidean gradient
        
        eg_temp = []
        for k in range(Nk):
            Lam_k = Lam[k]
            G_k = G[k]
            G_k_h = tf.transpose(G_k,perm=[0,2,1],conjugate=True)
            
            Alpha_k = Alpha[k]
            N = tf.linalg.inv(Lam_k) + tf.linalg.inv(Lam_k)@G_k_h@Wrf@Wrf_h@G_k/Alpha_k
            
            # Test determine (W_rf)
            # det_Lam_k = tf.linalg.det(Lam_k)
            # print('-------------------------------------------------------')
            # print('Wrf_lam_det=',det_Lam_k)
            # print('-------------------------------------------------------')
            # print('Wrf_inv_lam=',tf.linalg.inv(Lam_k))
            # print('-------------------------------------------------------')
            # N_det = tf.linalg.det(N)
            # print('Wrf_N_det =',tf.math.abs(N_det))

            N_inv = tf.linalg.inv(N)
            eg_temp.append(G_k @N_inv@N_inv@tf.linalg.inv(Lam_k)@G_k_h@Wrf/Alpha_k)
        
        eg_stack = tf.stack(eg_temp)     
        eg = tf.reduce_sum(eg_stack,axis=0)      
        eg = -eg/Nk*P_mask
        
        # calculate the Riemannian gradient
        projection_len = tf.cast(tf.math.real(eg*tf.math.conj(Wrf)),dtype = tf.complex64)
        rg = eg - projection_len*Wrf
        
        # update the point
        step_size = tf.cast(step_size[:,:,tf.newaxis],tf.complex64)
        norm_rg = tf.norm(rg,ord='fro',axis=[1,2])
        norm_rg = norm_rg[:,tf.newaxis,tf.newaxis]
        Wrf_new = Wrf - step_size*rg/norm_rg
        
        # retraction
        Wrf_new_abs = tf.abs(Wrf_new)
        Wrf_constrained = Wrf_new/tf.cast((tf.nn.relu(Wrf_new_abs-1)+1),dtype = tf.complex64)
        
        # # Test matrix (W_rf)
        # print('-------------------------------------------------------')
        # print('Wrf_step_size =', step_size)
        # print('W_rf =', Wrf_constrained)
        # print('abs(W_rf) =', tf.math.abs(Wrf_constrained))
        # print('-------------------------------------------------------')
        return Wrf_constrained
     
class MO_C(Layer):
    def __init__(self):
        super(MO_C, self).__init__()
        self.DUC_1 = DUC_block()
        self.DUC_2 = DUC_block()
        self.DUC_3 = DUC_block()
        self.DUC_4 = DUC_block()
        # self.DUC_5 = DUC_block()
        
        # self.DUC_6 = DUC_block()
        # self.DUC_7 = DUC_block()
        # self.DUC_8 = DUC_block()
        # self.DUC_9 = DUC_block()
        # self.DUC_10 = DUC_block()
        
    def call(self,inputs):
        Frf,Fbb,Wrf,xi,Lam,H1,H2,Phi,n_power = inputs
           
        Alpha = []
        G = []

        for k in range(Nk):
            xi_k = xi[k]
            Alpha_k = n_power*Nr/Nrf_r/xi_k/xi_k
            
            h1 = H1[:,:,:,k] # H1(realization,N_phi,Nt,Nk)        
            h2 = H2[:,:,:,k] # H2(realization,Nr,N_phi,Nk)
            
            H_eff_k = h2@Phi@h1 # H_eff(realization,Nr,Nt)
            
            Fbb_k = Fbb[k]
            
            G_k = H_eff_k@Frf@Fbb_k/xi_k
            
            Alpha.append(Alpha_k)
            G.append(G_k)
        
        # MO
        Wrf = self.DUC_1([Wrf,Alpha,G,Lam[:]])
        Wrf = self.DUC_2([Wrf,Alpha,G,Lam[:]])
        Wrf = self.DUC_3([Wrf,Alpha,G,Lam[:]])
        Wrf = self.DUC_4([Wrf,Alpha,G,Lam[:]])
        # Wrf = self.DUC_5([Wrf,Alpha,G,Lam[:]])
        
        # Wrf = self.DUC_6([Wrf,Alpha,G,Lam[:]])
        # Wrf = self.DUC_7([Wrf,Alpha,G,Lam[:]])
        # Wrf = self.DUC_8([Wrf,Alpha,G,Lam[:]])
        # Wrf = self.DUC_9([Wrf,Alpha,G,Lam[:]])
        # Wrf = self.DUC_10([Wrf,Alpha,G,Lam[:]])
       
        # Wrf = WRF
        
        return Wrf,Alpha,G
    
# ---------------------------- WMMSE (Combiner) ------------------------------ #      
class BB_C_Lam(Layer):
    def call(self,inputs):
        Wrf,Alpha,G = inputs
        
        Wrf_h = tf.transpose(Wrf,perm=[0,2,1],conjugate=True)
        Wbb = []
        Lam = []
        for k in range(Nk):
            Alpha_k = Alpha[k]
            G_k =  G[k]
            G_k_h = tf.transpose(G_k,perm=[0,2,1],conjugate=True)
                   
            Wbb_k = tf.linalg.inv(Wrf_h@G_k@G_k_h@Wrf + Alpha_k*tf.eye(Nrf_r,dtype=tf.complex64))@Wrf_h@G_k
            Lam_k = tf.eye(Ns,dtype=tf.complex64) + G_k_h@Wrf@Wrf_h@G_k/Alpha_k
            
            Wbb.append(Wbb_k)
            Lam.append(Lam_k)
        return Wbb,Lam



# The architecture of WMMSE-MO algorithm
class WMMSE_block(Layer):
    def __init__(self):
        super(WMMSE_block,self).__init__()
        self.mo_p = MO_P()
        self.bb_p = BB_P()
        self.mo_ris = MO_RIS()
        self.mo_c = MO_C()
        self.bb_c_lam = BB_C_Lam()
        
    def call(self,inputs):
        Frf,Wrf,Wbb,Lam,H1,H2,Phi,n_power = inputs
        
        Frf,Beta,G_tilde_h = self.mo_p([Frf,Wrf,Wbb,Lam,H1,H2,Phi,n_power])
        Fbb,xi = self.bb_p([Frf,Beta,G_tilde_h,Lam])
        Phi = self.mo_ris([Frf,Wrf,Wbb,Lam,H1,H2,Phi,Beta])
        Wrf,Alpha,G = self.mo_c([Frf,Fbb,Wrf,xi,Lam,H1,H2,Phi,n_power])
        Wbb,Lam = self.bb_c_lam([Wrf,Alpha,G])
        
        return Frf,Fbb,Wrf,Wbb,Lam,Phi 

          
# model
def DU_model():
    
    # Input
    Frf_init = Input(name ='Frf_init',shape=Frf_shape,dtype=tf.complex64)
    Wrf_init = Input(name ='Wrf_init',shape=Wrf_shape,dtype=tf.complex64)
    Wbb_init = Input(name ='Wbb_init',shape=Wbb_shape,dtype=tf.complex64)
    Lam_init = Input(name ='Lambda_init',shape=Lam_shape,dtype=tf.float32)
    Lam_init_complex = tf.cast(Lam_init,dtype=tf.complex64)
    H1 = Input(name='channel_1_matrix',shape=H1_shape,dtype=tf.complex64)
    H2 = Input(name='channel_2_matrix',shape=H2_shape,dtype=tf.complex64)
    Phi_init = Input(name='Phi_init',shape=Phi_shape,dtype=tf.complex64)
    
    noise_power = Input(name='noise_power',shape=np_shape,dtype=tf.float32)
    n_power = tf.cast(noise_power,dtype=tf.complex64)
    n_power = n_power[:,:,tf.newaxis] # shape=(?,1,1)
    
    
    Frf = Frf_init
    Wrf = Wrf_init
    Phi = Phi_init
    
    
    Wbb = []
    Lam = []
    
    for k in range(Nk):
        Wbb.append(Wbb_init[:,:,:,k])
        Lam.append(Lam_init_complex[:,:,:,k])
    
    WMMSE_1 = WMMSE_block()    
    Frf,Fbb,Wrf,Wbb,Lam,Phi = WMMSE_1([Frf,Wrf,Wbb,Lam,H1,H2,Phi,n_power])
    
    WMMSE_2 = WMMSE_block()
    Frf,Fbb,Wrf,Wbb,Lam,Phi = WMMSE_2([Frf,Wrf,Wbb,Lam,H1,H2,Phi,n_power])
    
    WMMSE_3 = WMMSE_block()
    Frf,Fbb,Wrf,Wbb,Lam,Phi = WMMSE_3([Frf,Wrf,Wbb,Lam,H1,H2,Phi,n_power])
    
    WMMSE_4 = WMMSE_block()
    Frf,Fbb,Wrf,Wbb,Lam,Phi = WMMSE_4([Frf,Wrf,Wbb,Lam,H1,H2,Phi,n_power])
    
    WMMSE_5 = WMMSE_block()
    Frf,Fbb,Wrf,Wbb,Lam,Phi = WMMSE_5([Frf,Wrf,Wbb,Lam,H1,H2,Phi,n_power])
    
    WMMSE_6 = WMMSE_block()
    Frf,Fbb,Wrf,Wbb,Lam,Phi = WMMSE_6([Frf,Wrf,Wbb,Lam,H1,H2,Phi,n_power])
    
    
   

    
    Frf = Lambda(lambda x:x,name='Frf')(Frf)
    Wrf = Lambda(lambda x:x,name='Wrf')(Wrf)
    Fbb_stack = Lambda(lambda x:tf.stack(x,axis = 1),name='Fbb')(Fbb)
    Wbb_stack = Lambda(lambda x:tf.stack(x,axis = 1),name='Wbb')(Wbb)
    Lam_stack = Lambda(lambda x:tf.stack(x,axis = 1),name='Lam_out')(Lam)
    
    Phi = Lambda(lambda x:x,name='Phi')(Phi)
    
    
    loss = Lambda(loss_function,dtype=tf.float32,output_shape=(1,),name='loss')([Frf,Fbb_stack,Wrf,Wbb_stack,H1,H2,Phi,n_power,Lam_stack])
       
    model = tf.keras.Model(
        inputs=[Frf_init,
                Wrf_init,
                Wbb_init,
                Lam_init,
                H1,
                H2,
                Phi_init,
                noise_power],
        outputs=[loss]
    )
    
    return model

print('create model')
model = DU_model()



adam = tf.keras.optimizers.Adam(learning_rate = learning_rate, clipvalue = 1) # test 0617_learning_rate = 0.01,
model.compile(optimizer=adam,loss={'loss':lambda y_true,y_pred:y_pred},run_eagerly=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience = 3, min_lr = 0.00001) # test 0428
earlystopping = EarlyStopping(monitor='val_loss', min_delta = 0.01, patience = 4)


print('begin training')
history = model.fit(
    x = train_dataset.repeat(),
    validation_data = val_dataset.repeat(),
    steps_per_epoch = N_train//batch_size,
    validation_steps = N_val//batch_size,
    epochs=epochs,
    verbose=1,
    # callbacks = [reduce_lr,earlystopping]
    callbacks = [reduce_lr]
   )  


#---------------------------------------------------------------------------
#   training process visualization
#---------------------------------------------------------------------------
training_loss=history.history['loss']
val_loss=history.history['val_loss']
plt.figure(figsize=(6,4))
plt.plot(training_loss,label="training_loss")
plt.plot(val_loss,label="validation_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
title = "Loss_Curve"
plt.title(title)  
plt.legend(loc="best")
plt.show(block=False)

file_path = f"./RIS_DU/Result_RIS_DU/"
from datetime import datetime
today = datetime.today().strftime(f'%Y-%m-%d')
plt.savefig(file_path + f"Loss_curve/{title}_{Io}_{In}_bs_{batch_size}_lr_{learning_rate}-{today}.png")


model_new = tf.keras.Model(inputs=model.input,outputs=
                            [model.get_layer('Frf').output,model.get_layer('Wrf').output,model.get_layer('Fbb').output,model.get_layer('Wbb').output,model.get_layer('Phi').output,model.get_layer('Lam_out').output,model.get_layer('loss').output])


weights_path = '.\\RIS_DU\\Result_RIS_DU\\model_weights_%d_%d_RIS_Nt_%d_N_Phi_%d_Nr_%d_Ns_%d_lr_%f.h5'%(Io,In,Nt,N_phi,Nr,Ns,learning_rate)

model_new.save_weights(weights_path)


# pickle file
pickle_model_new = tf.keras.Model(inputs=model.input,outputs=
                            [model.get_layer('Frf').output,model.get_layer('Wrf').output,model.get_layer('Fbb').output,model.get_layer('Wbb').output,model.get_layer('Phi').output,model.get_layer('Lam_out').output])

ww = pickle_model_new.get_weights()
print('packle weight')
weight_name = '.\\RIS_DU\\Result_RIS_DU\\DU_RIS_%d_%d_lr_%f'%(Io,In,learning_rate)
with open(weight_name, 'wb') as f:
    pickle.dump(ww, f)


# ------------------------- test ------------------------- #


# testing_data = sio.loadmat('.\\sparse_SV_channel_RIS\\testing_data\\H1_H2_Nt_%d_N_phi_%d_Nr_%d_Ns_%d_Training_data_H.mat'%(Nt,N_phi,Nr,Ns)) # data_100

# testing_H1 = np.transpose(testing_data['H1'],(3,0,1,2))
# testing_H2 = np.transpose(testing_data['H2'],(3,0,1,2))
# Num = testing_H1.shape[0]


# SNR_dB = np.array(range(-10,25,5))   
# # SNR_dB = np.array(range(-10,25,5))     
# SNR_lin = 10**(SNR_dB/10)    
# SNR_len =len(SNR_lin)

# # Output 
# Frf_test = np.zeros((SNR_len,Num,Nt,Nrf_t), dtype = 'complex64')  
# Wrf_test = np.zeros((SNR_len,Num,Nr,Nrf_r), dtype = 'complex64') 
# Fbb_test = np.zeros((SNR_len,Num,Nk,Nrf_t,Ns), dtype = 'complex64')   
# Wbb_test = np.zeros((SNR_len,Num,Nk,Nrf_r,Ns), dtype = 'complex64')   
# Phi_test = np.zeros((SNR_len,Num,N_phi,N_phi), dtype = 'complex64')   
# Lam_out_test = np.zeros((SNR_len,Num,Nk,Ns,Ns), dtype = 'complex64')   
# SE_test = []
# SE_test_store = np.zeros((SNR_len,Num))

# del_NaN_SE_test = []


# # SE5 = []

# for s in range(SNR_len):
#     print('\rSNR=%d '%(SNR_dB[s]) , end='',flush=True)
#     snr = SNR_lin[s]
#     n_power = 1/snr
    
#     testing_Frf = np.exp(1j*np.random.uniform(0,2*np.pi,(Num,Nt,Nrf_t)))*P_F
#     testing_Wrf = np.exp(1j*np.random.uniform(0,2*np.pi,(Num,Nr,Nrf_r)))*P_W
#     testing_Phi = np.zeros((Num,N_phi,N_phi))
#     testing_Wbb = np.random.randn(Num,Nrf_r,Ns,Nk) + 1j*np.random.randn(Num,Nrf_r,Ns,Nk)
#     testing_Lam = np.zeros((Num,Ns,Ns,Nk))
#     testing_n_power = np.ones((Num,1,))*n_power
#     r = np.ones((Num,1))

#     for j in range(Num):
#         phi11 = np.exp(1j*np.random.uniform(0,2*np.pi,(N_phi))) # phi is the vector / Phi is the matrix
#         testing_Phi[j,:,:] = np.diag(phi11)
#         for i in range(Nk):
#             testing_Lam[j,:,:,i] = np.eye(Ns)
   
    
#     output1 = model_new.predict(x=[testing_Frf,testing_Wrf,testing_Wbb,testing_Lam,testing_H1,testing_H2,testing_Phi,testing_n_power])
#     # testing_Wbb = np.transpose(output1[3],(0,2,3,1))
#     # testing_Lam = np.transpose(output1[4],(0,2,3,1))
    
#     # output = model_new.predict(x=[output1[0],output1[1],testing_Wbb,testing_Lam,testing_H,testing_n_power])
#     # testing_Wbb = np.transpose(output[3],(0,2,3,1))
#     # testing_Lam = np.transpose(output[4],(0,2,3,1))
    
#     # output = model_new.predict(x=[output[0],output[1],testing_Wbb,testing_Lam,testing_H,testing_n_power])
#     # testing_Wbb = np.transpose(output[3],(0,2,3,1))
#     # testing_Lam = np.transpose(output[4],(0,2,3,1))
    
#     # output = model_new.predict(x=[output[0],output[1],testing_Wbb,testing_Lam,testing_H,testing_n_power])
#     # testing_Wbb = np.transpose(output[3],(0,2,3,1))
#     # testing_Lam = np.transpose(output[4],(0,2,3,1))
    
#     # output = model_new.predict(x=[output[0],output[1],testing_Wbb,testing_Lam,testing_H,testing_n_power])
   
#     # ????
#     Frf_test[s,:,:,:] = output1[0]
#     Wrf_test[s,:,:,:] = output1[1]
#     Fbb_test[s,:,:,:,:] = output1[2]
#     Wbb_test[s,:,:,:,:] = output1[3]
#     Phi_test[s,:,:,:] = output1[4]
#     Lam_out_test[s,:,:,:,:]= output1[5]
#     SE_test_store[s,:] = output1[6]

#     SE_test.append(np.sum(output1[6])/Num)
    
#     # Solve NaN problem
#     del_SE = []
#     NaN_position = np.where(np.isnan(output1[6]))
#     del_SE = np.delete(output1[6],NaN_position)
#     NaN_num = len(del_SE) # SE not contain NaN
#     del_NaN_SE_test.append(np.sum(del_SE)/(NaN_num))
    
#     # SE1.append(-np.sum(output1[5])/Num)
#     # SE5.append(np.sum(output[5])/Num)
    
    
# # plot SE (real-time)    
# abs_SE_test = abs(np.array(SE_test))
# plt.figure(figsize=(10,10)) 
# plt.plot(SNR_dB,abs_SE_test,'r')       
# plt.show()

# # plot del_NaN_SE (real-time)    
# abs_del_NaN_SE_test = abs(np.array(del_NaN_SE_test))
# plt.figure(figsize=(10,10)) 
# plt.plot(SNR_dB,abs_del_NaN_SE_test,'r')       
# plt.show()



