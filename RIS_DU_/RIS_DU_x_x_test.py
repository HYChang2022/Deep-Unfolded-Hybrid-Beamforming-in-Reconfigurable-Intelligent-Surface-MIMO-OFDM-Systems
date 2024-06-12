# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 01:30:58 2023

@author: Benson
"""

import pickle
import time
import math as m
import numpy as np
import scipy as sci
import scipy.io as sio
import matplotlib.pyplot as plt
import os


#---------------------------------------------------------------------------
#   setting the parameters 
#---------------------------------------------------------------------------
# outer(Io) / Inner(In) iteration
Io = 6
In = 4
# parameter
ULA = 'ULA' 
USPA = 'USPA'

Ns = 2
Nt = 32
Nr = 32
N_phi = 64 # the number of RIS (RIS_dim = N_phi*N_phi)
Nrf_t = 4
Nrf_r = 4 # original_Nrf_r = 4
learning_rate = 0.001
Nk = 16 # number of subcarrier;


Mt = int(Nt/Nrf_t)
Mr = int(Nr/Nrf_r)

ratio = 0.1


file_path = f".\\RIS_DU\\Result_RIS_DU\\"
weights_name = f"DU_RIS_{Io}_{In}_lr_{learning_rate}"

# load weight
with open(file_path + weights_name, 'rb') as f:
    weights = pickle.load(f)

weight_f = np.zeros((Io,In,Nt*2))
bias_f = np.zeros((Io,In,1))

weight_w = np.zeros((Io,In,Nr*2))
bias_w = np.zeros((Io,In,1))

weight_p = np.zeros((Io,In,N_phi*2))
bias_p = np.zeros((Io,In,1))


gamma_f =np.zeros((Io,In,Nt*2))
beta_f = np.zeros((Io,In,Nt*2))

gamma_w =np.zeros((Io,In,Nr*2))
beta_w = np.zeros((Io,In,Nr*2))

gamma_p =np.zeros((Io,In,N_phi*2))
beta_p = np.zeros((Io,In,N_phi*2))

mean_f = np.zeros((Io,In,Nt*2))
std_f = np.zeros((Io,In,Nt*2))

mean_w = np.zeros((Io,In,Nr*2))
std_w = np.zeros((Io,In,Nr*2))

mean_p = np.zeros((Io,In,N_phi*2))
std_p = np.zeros((Io,In,N_phi*2))

for i in range(Io):
    for j in range(In):
        
        gamma_f[i,j,:] = weights[i*(3*In)*6+j*4]
        beta_f[i,j,:] = weights[i*(3*In)*6+j*4 +1]
        
        weight_f[i,j,:] = weights[i*(3*In)*6+j*4 +2][:,0]
        bias_f[i,j,:] = weights[i*(3*In)*6+j*4 +3]
        
        gamma_p[i,j,:] = weights[i*(3*In)*6+j*4 + In*4]
        beta_p[i,j,:] = weights[i*(3*In)*6+j*4 +1 + In*4]
        
        weight_p[i,j,:] = weights[i*(3*In)*6+j*4 +2 + In*4][:,0]
        bias_p[i,j,:] = weights[i*(3*In)*6+j*4 +3 + In*4]
        
#--------------------------------------------------------------------- #        
        gamma_w[i,j,:] = weights[i*(3*In)*6+j*4 + In*4*2]
        beta_w[i,j,:] = weights[i*(3*In)*6+j*4 +1 + In*4*2]
        
        weight_w[i,j,:] = weights[i*(3*In)*6+j*4 +2 + In*4*2][:,0]
        bias_w[i,j,:] = weights[i*(3*In)*6+j*4 +3 + In*4*2]
        
#--------------------------------------------------------------------- #          
        mean_f[i,j,:] = weights[i*(3*In)*6+j*2 + In*4*3]
        std_f[i,j:] = weights[i*(3*In)*6+j*2 +1 + In*4*3]
        
        mean_p[i,j,:] = weights[i*(3*In)*6+j*2 +In*4*3 + In*2]
        std_p[i,j,:] = weights[i*(3*In)*6+j*2 +1 +In*4*3 + In*2]
        
#--------------------------------------------------------------------- #          
        mean_w[i,j,:] = weights[i*(3*In)*6+j*2 +In*4*3 + In*2*2]
        std_w[i,j,:] = weights[i*(3*In)*6+j*2 +1 +In*4*3 + In*2*2]
        
#        
selection_matrix_t = np.ones((Nrf_t,1))
selection_matrix_p = np.ones((N_phi,1))
selection_matrix_r = np.ones((Nrf_r,1))

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
    
def DU_Frf(Frf,weight_f,bias_f,gamma_f,beta_f,mean_f,std_f,Beta,G_tilde_h,Lam):
    
    Frf_p = Frf@selection_matrix_t
    Frf_p_vec = np.reshape(Frf_p,(Nt,1))     
    Frf_p_vec_I = Frf_p_vec.real
    Frf_p_vec_Q = Frf_p_vec.imag
    Frf_p_vec_IQ =np.concatenate((Frf_p_vec_I,Frf_p_vec_Q),axis=0)
    
   
    # BatchNormalization
    temp = (Frf_p_vec_IQ-mean_f)/(std_f+0.001)*gamma_f+beta_f
    # print('Frf_old_temp =',temp.shape)
    # Dense 
    temp = weight_f@temp + bias_f
    # print('weight_f =',weight_f.shape)
    
    
    if(temp<0): # PReLU
        temp = ratio*temp
    
    # MO update
    step_size = temp
    Frf_h = Frf.conj().T

    # calculate the euclidean gradient
    eg_temp = 0
    for k in range(Nk):
        Lam_k = Lam[:,:,k]
        G_tilde_k_h = G_tilde_h[:,:,k]
        G_tilde_k = G_tilde_k_h.conj().T
        
        Beta_k = Beta[0,k] 
      
        M =  np.linalg.inv(Lam_k) + G_tilde_k@Frf@Frf_h@G_tilde_k_h/Beta_k
        
        eg_temp = eg_temp + G_tilde_k_h @ np.linalg.inv(M)@np.linalg.inv(M)@G_tilde_k@Frf/Beta_k      
    eg = -eg_temp/Nk*P_F
    
    # calculate the Riemannian gradient
    projection_len = np.real(eg*Frf.conj())
    rg = eg - projection_len*Frf
    
    # update the point
    norm_rg = np.linalg.norm(rg,ord='fro',axis=(0,1))
    Frf_new = Frf - step_size*rg/norm_rg
    
    # retraction
    Frf_new_abs = np.abs(Frf_new)
    Frf_constrained = Frf_new/Frf_new_abs
    Frf_constrained[np.isnan(Frf_constrained)] = 0
    
    # return Frf_constrained,step_size
    return Frf_constrained



def DU_RIS(Phi,weight_p,bias_p,gamma_p,beta_p,mean_p,std_p,Beta,G_hat_h,Lam,C):
    
    Phi_p = Phi @ selection_matrix_p
    Phi_p_vec = np.reshape(Phi_p,(N_phi,1))     
    Phi_p_vec_I = Phi_p_vec.real
    Phi_p_vec_Q = Phi_p_vec.imag
    Phi_p_vec_IQ =np.concatenate((Phi_p_vec_I,Phi_p_vec_Q),axis=0)
    
   
    # BatchNormalization
    temp = (Phi_p_vec_IQ-mean_p)/(std_p+0.001)*gamma_p+beta_p
    # print('Phi_old_temp =',temp.shape)
    
    # Dense 
    temp = weight_p @ temp + bias_p
    
    if(temp<0): # PReLU
        temp = ratio*temp
    
    # MO update
    step_size = temp
    Phi_h = Phi.conj().T

    # calculate the euclidean gradient
    eg_temp = 0
    
    for k in range(Nk):
        Lam_k = Lam[:,:,k]
        G_hat_k_h = G_hat_h[:,:,k]
        G_hat_k = G_hat_k_h.conj().T
        C_k = C[:,:,k]
        Beta_k = Beta[0,k] 
      
        Q =  np.linalg.inv(Lam_k) + G_hat_k@Phi@C_k@Phi_h@G_hat_k_h/Beta_k
        
        eg_temp = eg_temp + G_hat_k_h @ np.linalg.inv(Q)@np.linalg.inv(Q)@G_hat_k@Phi@C_k/Beta_k    
   
    eg = -eg_temp/Nk*P_RIS
    
    # calculate the Riemannian gradient
    projection_len = np.real(eg*Phi.conj())
    rg = eg - projection_len*Phi
    
    # update the point
    norm_rg = np.linalg.norm(rg,ord='fro',axis=(0,1))
    Phi_new = Phi - step_size*rg/norm_rg
    
    # retraction
    Phi_new_abs = np.abs(Phi_new)
    Phi_constrained = Phi_new/Phi_new_abs
    Phi_constrained[np.isnan(Phi_constrained)] = 0
    
    # return Frf_constrained,step_size
    return Phi_constrained



def DU_Wrf(Wrf,weight_w,bias_w,gamma_w,beta_w,mean_w,std_w,Alpha,G,Lam):
    
    Wrf_p = Wrf@selection_matrix_r
    Wrf_p_vec = np.reshape(Wrf_p,(Nr,1))     
    Wrf_p_vec_I = Wrf_p_vec.real
    Wrf_p_vec_Q = Wrf_p_vec.imag
    Wrf_p_vec_IQ =np.concatenate((Wrf_p_vec_I,Wrf_p_vec_Q),axis=0)
    
   
    # BatchNormalization
    temp = (Wrf_p_vec_IQ-mean_w)/(std_w+0.001)*gamma_w +beta_w
    # Dense 
    temp = weight_w@temp + bias_w
    if(temp<0): # PReLU
        temp = ratio*temp
    
    # MO update
    step_size = temp
    
    Wrf_h = Wrf.conj().T
    
    # calculate the euclidean gradient
    eg_temp = 0
    for k in range(Nk):
        Lam_k = Lam[:,:,k]
        G_k = G[:,:,k]
        G_k_h = G_k.conj().T
        
        Alpha_k = Alpha[0,k]
        N = np.linalg.inv(Lam_k) + np.linalg.inv(Lam_k)@G_k_h@Wrf@Wrf_h@G_k/Alpha_k
        
        eg_temp = eg_temp + G_k @ np.linalg.inv(N)@np.linalg.inv(N)@np.linalg.inv(Lam_k)@G_k_h@Wrf/Alpha_k    
    eg = -eg_temp/Nk*P_W

    # calculate the Riemannian gradient
    projection_len = np.real(eg*Wrf.conj())
    rg = eg - projection_len*Wrf
    
    # update the point
    norm_rg = np.linalg.norm(rg,ord='fro',axis=(0,1))
    Wrf_new = Wrf - step_size*rg/norm_rg
    
    # retraction
    Wrf_new_abs = np.abs(Wrf_new)
    Wrf_constrained = Wrf_new/Wrf_new_abs
    Wrf_constrained[np.isnan(Wrf_constrained)] = 0
    
    # return Wrf_constrained,step_size
    return Wrf_constrained

#---------------------------------------------------------------------------
#  WMMSE_MO_iterative
#--------------------------------------------------------------------------- 


def WMMSE_MO(para):
    H1,H2,n_power = para
    # step_size_f = np.zeros((Io,In))
    # step_size_w = np.zeros((Io,In))
    
    Alpha = np.zeros((1,Nk))
    Beta = np.zeros((1,Nk))
    xi = np.zeros((1,Nk))
    G_tilde_h = np.zeros((Nt,Ns,Nk),dtype=np.complex128)
    G = np.zeros((Nr,Ns,Nk),dtype=np.complex128)
    G_hat_h = np.zeros((N_phi,Ns,Nk),dtype=np.complex128)
    H_eff = np.zeros((Nr,Nt,Nk),dtype=np.complex128)
    Phi = np.zeros((Nr,Nt),dtype=np.complex128)
    

    A = np.ones((Mt,1))
    B = np.ones((Mr,1))
    F_mask = np.zeros((Nt,Nrf_t))
    W_mask = np.zeros((Nr,Nrf_r))
    
    C = np.zeros((N_phi,N_phi,Nk))
    
        
    Frf = np.exp(1j*np.random.uniform(0,2*np.pi,(Nt,Nrf_t)))*P_F
    Wrf = np.exp(1j*np.random.uniform(0,2*np.pi,(Nr,Nrf_r)))*P_W
    phi = np.exp(1j*np.random.uniform(0,2*np.pi,(N_phi))) # phi is the vector / Phi is the matrix
    Phi = np.diag(phi)
    
    
    Fbb = np.random.randn(Nrf_t,Ns,Nk) + 1j*np.random.rand(Nrf_t,Ns,Nk)
    Wbb = np.random.randn(Nrf_r,Ns,Nk) + 1j*np.random.rand(Nrf_r,Ns,Nk)
    
    Lam = np.zeros((Ns,Ns,Nk),dtype=np.complex128) # weighted matrix
    
    for i in range(Nk):
        Lam[:,:,i] = np.eye(Ns,dtype=np.complex128)
    
    for i in range(Io):
        
        # Frf base on MO =============================================================================================================
        for k in range(Nk):
            Beta[0,k] = n_power*Nt*Nr/Nrf_t/Nrf_r*np.trace(Lam[:,:,k]@Wbb[:,:,k].conj().T@Wbb[:,:,k])            
            H_eff[:,:,k] = H2[:,:,k]@Phi@H1[:,:,k]
            G_tilde_h[:,:,k] = H_eff[:,:,k].conj().T@Wrf@Wbb[:,:,k]
        
        for j in range(In):
            w_f = weight_f[i,j,:]
            bi_f = bias_f[i,j,:]
            g_f = gamma_f[i,j,:]
            be_f = beta_f[i,j,:]
            mu_f = mean_f[i,j,:]
            st_f = std_f[i,j,:]
            
            w_f = w_f[np.newaxis,:]
            bi_f = bi_f[np.newaxis,:]
            g_f = g_f[:,np.newaxis]
            be_f = be_f[:,np.newaxis]
            mu_f = mu_f[:,np.newaxis]
            st_f = st_f[:,np.newaxis]
            
            Frf = DU_Frf(Frf,w_f,bi_f,g_f,be_f,mu_f,st_f,Beta,G_tilde_h,Lam)
            
        # Fbb_k ======================================================================================================================
        for k in range(Nk):
            Beta_k = Beta[0,k]  
            G_tilde_k_h = G_tilde_h[:,:,k]
            F_tilde = Frf.conj().T@G_tilde_k_h@Lam[:,:,k]@G_tilde_k_h.conj().T@Frf + Beta_k*np.eye(Nrf_t)
            xi_k = 1/np.sqrt(Nt/Nrf_t/Ns*(np.linalg.norm(np.linalg.inv(F_tilde)@Frf.conj().T@G_tilde_k_h@Lam[:,:,k],ord='fro')**2))
            
            Fbb[:,:,k] = xi_k*np.linalg.inv(F_tilde)@Frf.conj().T@G_tilde_k_h@Lam[:,:,k]
            xi[0,k] = xi_k
        
        # RIS base on MO =============================================================================================================
        for k in range(Nk):
            G_hat_h[:,:,k] = H2[:,:,k].conj().T@Wrf@Wbb[:,:,k]
            C[:,:,k] = H1[:,:,k]@Frf@Frf.conj().T@H1[:,:,k].conj().T
           

        for j in range(In):     
            w_p = weight_p[i,j,:]
            bi_p = bias_p[i,j,:]
            g_p = gamma_p[i,j,:]
            be_p = beta_p[i,j,:]
            mu_p = mean_p[i,j,:]
            st_p = std_p[i,j,:]
            
            
            w_p = w_p[np.newaxis,:]
            bi_p = bi_p[np.newaxis,:]
            g_p = g_p[:,np.newaxis]
            be_p = be_p[:,np.newaxis]
            mu_p = mu_p[:,np.newaxis]
            st_p = st_p[:,np.newaxis]
            
            Phi = DU_RIS(Phi,w_p,bi_p,g_p,be_p,mu_p,st_p,Beta,G_hat_h,Lam,C)
                         
        # Wrf base on MO =============================================================================================================
        for k in range(Nk):
            xi_k = xi[0,k]
            Alpha[0,k] = n_power*Nr*Nrf_r/xi_k/xi_k
            H_eff[:,:,k] = H2[:,:,k]@Phi@H1[:,:,k]
            G[:,:,k] = H_eff[:,:,k]@Frf@Fbb[:,:,k]/xi_k
            
        for j in range(In):
            w_w = weight_w[i,j,:]
            bi_w = bias_w[i,j,:]
            g_w = gamma_w[i,j,:]
            be_w = beta_w[i,j,:]
            mu_w = mean_w[i,j,:]
            st_w = std_w[i,j,:]
            
            w_w = w_w[np.newaxis,:]
            bi_w = bi_w[np.newaxis,:]
            g_w = g_w[:,np.newaxis]
            be_w = be_w[:,np.newaxis]
            mu_w = mu_w[:,np.newaxis]
            st_w = st_w[:,np.newaxis]
            
            Wrf = DU_Wrf(Wrf,w_w,bi_w,g_w,be_w,mu_w,st_w,Alpha,G,Lam)
          
        # Wbb_k & Lambda_k ===========================================================================================================
        for k in range(Nk):
            Alpha_k = Alpha[0,k]
            G_k = G[:,:,k]
            
            Wbb[:,:,k] = np.linalg.inv(Wrf.conj().T@G_k@G_k.conj().T@Wrf + Alpha_k*np.eye(Nrf_r))@Wrf.conj().T@G_k
            
            Lam[:,:,k] = np.eye(Ns) + G_k.conj().T@Wrf@Wrf.conj().T@G_k/Alpha_k
        
    return Frf,Fbb,Phi,Wrf,Wbb

if __name__ == '__main__' :
    
    testing_data = sio.loadmat('.\\sparse_SV_channel_RIS\\testing_data\\H1_H2_Nt_%d_N_phi_%d_Nr_%d_Ns_%d_Testing_data_H.mat'%(Nt,N_phi,Nr,Ns)) # data_100
 
    testing_H1 = testing_data['H1']
    testing_H2 = testing_data['H2']
    realization = testing_H1.shape[3]
    
    H_eff = np.zeros((Nr,Nt,Nk),dtype=np.complex128)
       
    # fix SNR = 10 dB
    SNR_dB = np.array(range(10,15,5)) 
    
    
    # SNR_dB = np.array(range(-10,25,5)) 
    
    SNR_lin = 10**(SNR_dB/10)    
    SNR_len =len(SNR_lin)

    R = np.zeros([SNR_len,realization])
    exe_time_avg = np.zeros((SNR_len))
    
    print('start testing')
    start = time.perf_counter()
    for s in range(SNR_len):
        start = time.perf_counter()
        snr = SNR_lin[s]
        n_power = 1/snr

        for i in range(realization):
            print('\rSNR=%d No. %d '%(SNR_dB[s],i) , end='',flush=True)
            # H = channel[:,:,:,i]
            H1 = testing_H1[:,:,:,i]
            H2 = testing_H2[:,:,:,i]
            
            para_WMMSE_MO = (H1,H2,n_power)
            # Frf,Fbb,Wrf,Wbb,sf,sw = WMMSE_MO(para_WMMSE_MO)
            Frf,Fbb,Phi,Wrf,Wbb = WMMSE_MO(para_WMMSE_MO)
            
            for k in range(Nk):
                H_eff[:,:,k] = H2[:,:,k]@Phi@H1[:,:,k] 
                Ft = Frf@Fbb[:,:,k]
                Ft_h = Ft.conj().T
                
                Wt = Wrf@Wbb[:,:,k]
                Wt_h = Wt.conj().T
                
                pinv_Wt = np.linalg.inv(Wt_h@Wt+ 1e-10)@Wt_h     
                R[s,i] =  R[s,i] + np.log2(np.linalg.det(np.eye(Ns)+snr*pinv_Wt@H_eff[:,:,k]@Ft@Ft_h@H_eff[:,:,k].conj().T@Wt))
                
            R[s,i] = R[s,i]/Nk 
            
            
        elapsed = time.perf_counter() - start            
        exe_time_avg[s] =  elapsed/realization
        print('time = %f'%(exe_time_avg[s]))
       
    SE = np.sum(R,axis=1)/realization

    plt.figure(figsize=(10,10)) 
    plt.plot(SNR_dB,SE,'r')
    plt.show()
    
