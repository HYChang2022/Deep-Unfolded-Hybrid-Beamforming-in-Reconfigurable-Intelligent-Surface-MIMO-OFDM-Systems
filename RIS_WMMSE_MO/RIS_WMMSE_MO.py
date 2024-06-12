# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 13:20:43 2023

@author: Benson
"""

import time
import math
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
np.object = object
np.int = int
np.bool = bool

from scipy.linalg import sqrtm
from mypymanopt.manifolds import ComplexCircle
from mypymanopt import Problem
from mypymanopt.solvers import SteepestDescent,ConjugateGradient


#---------------------------------------------------------------------------
#   setting the parameters 
#---------------------------------------------------------------------------

# antenna_array

ULA = 'ULA' 
USPA = 'USPA'
N_phi = 64 # the number of RIS (RIS_dim = N_phi*N_phi)

Nt = 32
Nr = 32

Ns = 2
Nrf_t = 4
Nrf_r = 4   
Mt = int(Nt/Nrf_t)
Mr = int(Nr/Nrf_r)
Nk = 16
Iter = 2 # original Iter = 20 (outer iteration)


# ------------------------------------ The Frf(MO) ------------------------------------ #

def Frf_MO(Frf,Beta,G_tilde_h,Lam):
    
    A = np.ones((Mt,1))
    F_mask = np.zeros((Nt,Nrf_t))

    for i in range(Nrf_t):
        F_mask[i*Mt:(i+1)*Mt,i] = A[:,0] # consider the partially-connected structure
        
    manifold = ComplexCircle(Nt*Nrf_t)
    
    def mycost(x):
        X = np.reshape(x,(Nt,Nrf_t)) # order need more check
        # X[F_mask==0] = 0
        
        f = 0
        for k in range(Nk): # deal with subcarriers separately
            Lam_k = Lam[:,:,k]
            G_tilde_k_h = G_tilde_h[:,:,k]
            Beta_k = Beta[0,k]
            f = f + np.trace(np.linalg.inv(np.linalg.inv(Lam_k)+G_tilde_k_h.conj().T@X@X.conj().T@G_tilde_k_h/Beta_k))
 
        return f
        
    
    def myegrad(x):
        X = np.reshape(x,(Nt,Nrf_t)) # order need more check (Frf)
        
        
        eg = np.zeros((Nt,Nrf_t))
        for k in range(Nk):
            Lam_k = Lam[:,:,k]
            G_tilde_k_h = G_tilde_h[:,:,k]
            Beta_k = Beta[0,k]
            M = np.linalg.inv(Lam_k) + G_tilde_k_h.conj().T@X@X.conj().T@G_tilde_k_h/Beta_k
            
            eg = eg + G_tilde_k_h@np.linalg.inv(M)@np.linalg.inv(M)@G_tilde_k_h.conj().T@X/Beta_k
        
        eg = -(eg)*F_mask            
        eg = np.reshape(eg,(-1,)) # reshape from matrix to vector form (row)
        
        return eg
    
    problem = Problem(manifold = manifold, cost = mycost, egrad = myegrad)
    
    solver = ConjugateGradient()
    
    y = solver.solve(problem = problem,x = np.reshape(Frf,(-1,)))
    y = np.reshape(y,(Nt,Nrf_t))
    
    
    return y

# ---------------------------- The phase shifts of RIS (MO) ---------------------------- #

def RIS_MO(Phi,Beta,G_hat_h,Lam,C):
    
    

    manifold = ComplexCircle(N_phi*N_phi)
    
    
    def mycost(x):
        X = np.reshape(x,(N_phi,N_phi)) # order need more check
        
        
        f = 0
        for k in range(Nk): # deal with subcarriers separately
            Lam_k = Lam[:,:,k]
            G_hat_k_h = G_hat_h[:,:,k]
            Beta_k = Beta[0,k]        
            C_k = C[:,:,k]
            
            f = f + np.trace(np.linalg.inv(np.linalg.inv(Lam_k)+G_hat_k_h.conj().T@X@C_k@X.conj().T@G_hat_k_h/Beta_k))
        
        
        return f
        
    def myegrad(x):
        X = np.reshape(x,(N_phi,N_phi)) # order need more check (Phi)
        # X[F_mask==0] = 0
        G = np.ones((N_phi))
        RIS_mask = np.diag(G)
        
        eg = np.zeros((N_phi,N_phi))
        for k in range(Nk):
            Lam_k = Lam[:,:,k]
            G_hat_k_h = G_hat_h[:,:,k]
            Beta_k = Beta[0,k]
            C_k = C[:,:,k]
            
            M = np.linalg.inv(Lam_k) + G_hat_k_h.conj().T@X@C_k@X.conj().T@G_hat_k_h/Beta_k
            
            eg = eg + G_hat_k_h@np.linalg.inv(M)@np.linalg.inv(M)@G_hat_k_h.conj().T@X@C_k/Beta_k
        
        eg = -(eg)*RIS_mask    

        eg = np.reshape(eg,(-1,)) # reshape from matrix to vector form (row)
        

        return eg
         
    
    
    problem = Problem(manifold = manifold, cost = mycost, egrad = myegrad)
    
    solver = ConjugateGradient()
    
    y = solver.solve(problem = problem,x = np.reshape(Phi,(-1,)))
    y = np.reshape(y,(N_phi,N_phi))
    
    
    return y

# ------------------------------------ The Wrf(MO) ------------------------------------ #

def Wrf_MO(Wrf,Alpha,G,Lam):
    
    B = np.ones((Mr,1))
    W_mask = np.zeros((Nr,Nrf_r))

    for i in range(Nrf_r):
        W_mask[i*Mr:(i+1)*Mr,i] = B[:,0]
        
    manifold = ComplexCircle(Nr*Nrf_r)
    
    def mycost(x):
        X = np.reshape(x,(Nr,Nrf_r)) # order need more check
        
        f = 0
        for k in range(Nk):
            Lam_k = Lam[:,:,k]
            G_k = G[:,:,k]
            Alpha_k = Alpha[0,k]
            
            f = f + np.trace(np.linalg.inv(np.linalg.inv(Lam_k) + np.linalg.inv(Lam_k)@G_k.conj().T@X@X.conj().T@G_k/Alpha_k))
        
        return f
    
    def myegrad(x):
        X = np.reshape(x,(Nr,Nrf_r)) # order need more check
        
        
        eg = np.zeros((Nr,Nrf_r))
        for k in range(Nk):
            Lam_k = Lam[:,:,k]
            G_k = G[:,:,k]
            Alpha_k = Alpha[0,k]
            
            N = np.linalg.inv(Lam_k) + np.linalg.inv(Lam_k)@G_k.conj().T@X@X.conj().T@G_k/Alpha_k
            
            eg = eg + G_k@np.linalg.inv(N)@np.linalg.inv(N)@np.linalg.inv(Lam_k)@G_k.conj().T@X/Alpha_k
        
        eg = -(eg)*W_mask            
        eg = np.reshape(eg,(-1,))
        
        return eg
    
    problem = Problem(manifold = manifold,cost=mycost,egrad=myegrad)
    
    solver = ConjugateGradient()
    
    
#  inner_iteration in file mypymanopt.solver.solver
    y = solver.solve(problem = problem,x = np.reshape(Wrf,(-1,)))
    y = np.reshape(y,(Nr,Nrf_r))
   
    
    return y

#---------------------------------------------------------------------------
#  WMMSE_MO_iterative
#--------------------------------------------------------------------------- 


def WMMSE_MO(para):
    H1,H2,n_power = para
    
    Alpha = np.zeros((1,Nk))
    Beta = np.zeros((1,Nk))
    xi = np.zeros((1,Nk))
    G_tilde_h = np.zeros((Nt,Ns,Nk),dtype=np.complex128)
    G = np.zeros((Nr,Ns,Nk),dtype=np.complex128)
    G_hat_h = np.zeros((N_phi,Ns,Nk),dtype=np.complex128)
    H_eff = np.zeros((Nr,Nt,Nk),dtype=np.complex128)
    Phi = np.zeros((Nr,Nt),dtype=np.complex128)
    
    wmmse = np.zeros((Nk))
    
    A = np.ones((Mt,1))
    B = np.ones((Mr,1))
    F_mask = np.zeros((Nt,Nrf_t))
    W_mask = np.zeros((Nr,Nrf_r))
    
    C = np.zeros((N_phi,N_phi,Nk))
    
    for i in range(Nrf_t):
        F_mask[i*Mt:(i+1)*Mt,i] = A[:,0]
        
    for i in range(Nrf_r):
        W_mask[i*Mr:(i+1)*Mr,i] = B[:,0]        
        
    Frf = np.exp(1j*np.random.uniform(0,2*np.pi,(Nt,Nrf_t)))*F_mask # generate the initial Frf
    Wrf = np.exp(1j*np.random.uniform(0,2*np.pi,(Nr,Nrf_r)))*W_mask # generate the initial Wrf
    
    
    
    Fbb = np.random.randn(Nrf_t,Ns,Nk) + 1j*np.random.rand(Nrf_t,Ns,Nk)
    Wbb = np.random.randn(Nrf_r,Ns,Nk) + 1j*np.random.rand(Nrf_r,Ns,Nk)
    phi = np.exp(1j*np.random.uniform(0,2*np.pi,(N_phi))) # phi is the vector / Phi is the matrix
    Phi = np.diag(phi)
    
    
    Lam = np.zeros((Ns,Ns,Nk),dtype=np.complex128) # weighted matrix
    
    for i in range(Nk):
        Lam[:,:,i] = np.eye(Ns,dtype=np.complex128)
    
    
    I = 0
    delta = 1
    new_wmmse = 10
    while I<Iter and delta >= 10**(-4): # outer iterations <= I and continuing criterion 
        # Frf base on MO =============================================================================================================
    
        for k in range(Nk):
            Beta[0,k] = n_power*Nt*Nr/Nrf_t/Nrf_r*np.trace(Lam[:,:,k]@Wbb[:,:,k].conj().T@Wbb[:,:,k])
            
            H_eff[:,:,k] = H2[:,:,k]@Phi@H1[:,:,k]
            G_tilde_h[:,:,k] = H_eff[:,:,k].conj().T@Wrf@Wbb[:,:,k]
        
        Frf = Frf_MO(Frf,Beta,G_tilde_h,Lam)  
          
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
           
            
        Phi = RIS_MO(Phi,Beta,G_hat_h,Lam,C)
            
        # Wrf base on MO =============================================================================================================
        for k in range(Nk):
            xi_k = xi[0,k]
            Alpha[0,k] = n_power*Nr*Nrf_r/xi_k/xi_k
            
            H_eff[:,:,k] = H2[:,:,k]@Phi@H1[:,:,k]
            G[:,:,k] = H_eff[:,:,k]@Frf@Fbb[:,:,k]/xi_k
            
        Wrf = Wrf_MO(Wrf,Alpha,G,Lam)            
        
        # Wbb_k & Lambda_k ===========================================================================================================
        for k in range(Nk):
            Alpha_k = Alpha[0,k]
            G_k = G[:,:,k]
            
            Wbb[:,:,k] = np.linalg.inv(Wrf.conj().T@G_k@G_k.conj().T@Wrf + Alpha_k*np.eye(Nrf_r))@Wrf.conj().T@G_k
            if I>=0:
                Lam[:,:,k] = np.eye(Ns) + G_k.conj().T@Wrf@Wrf.conj().T@G_k/Alpha_k
            wmmse[k] = np.trace(np.eye(Ns)) - np.log(np.linalg.det( Lam[:,:,k]))    
        
        old_wmmse = new_wmmse
        ave_wmmse = np.mean(wmmse)
        new_wmmse = ave_wmmse
        delta = old_wmmse - new_wmmse
        I = I+1
     
    return Frf,Fbb,Phi,Wrf,Wbb,Lam


#---------------------------------------------------------------------------
#  main
#---------------------------------------------------------------------------

if __name__ == '__main__' :
    
    testing_data = sio.loadmat('.\\sparse_SV_channel_RIS\\testing_data\\H1_H2_Nt_%d_N_phi_%d_Nr_%d_Ns_%d_Testing_data_H.mat'%(Nt,N_phi,Nr,Ns))
   
    channel_1 = testing_data['H1']
    channel_2 = testing_data['H2']
    
    
    realization = channel_1.shape[3] # channel(Nr,Nt,k,realization)
    H_eff = np.zeros((Nr,Nt,Nk),dtype=np.complex128)       
    SNR_dB = np.array(range(-10,25,5))    
    
    SNR_lin = 10**(SNR_dB/10)    
    SNR_len =len(SNR_lin)
    
    ## test power_Ft
    temp_Ft = np.zeros([Nt,Ns,Nk,realization])
    
    R = np.zeros([SNR_len,realization])
    exe_time_avg = np.zeros((SNR_len))
    print('start testing')

    
    for s in range(SNR_len):
        start = time.perf_counter()    
        snr = SNR_lin[s]
        n_power = 1/snr 

        for i in range(realization):
            print('\rSNR=%d No. %d '%(SNR_dB[s],i) , end='',flush=True)
            H1 = channel_1[:,:,:,i]
            H2 = channel_2[:,:,:,i]
    
            para_WMMSE_MO = (H1,H2,n_power)
            Frf,Fbb,Phi,Wrf,Wbb,Lam = WMMSE_MO(para_WMMSE_MO)
            
            for k in range(Nk):
                H_eff[:,:,k] = H2[:,:,k]@Phi@H1[:,:,k]    
                Ft = Frf@Fbb[:,:,k]
                Ft_h = Ft.conj().T
                
                temp_Ft[:,:,k,i] = Ft
                
                Wt = Wrf@Wbb[:,:,k]
                Wt_h = Wt.conj().T
                               
                pinv_Wt = np.linalg.inv(Wt_h@Wt + 1e-10)@Wt_h
                R[s,i] =  R[s,i] + np.log2(np.linalg.det(np.eye(Ns)+snr*pinv_Wt@H_eff[:,:,k]@Ft@Ft_h@H_eff[:,:,k].conj().T@Wt))
        
            R[s,i] = R[s,i]/Nk


# timing per SNR          
        elapsed = time.perf_counter() - start            
        exe_time_avg[s] =  elapsed/realization
        print('time = %f'%(exe_time_avg[s]))
        
                  
# Spectral efficiency
    SE = np.sum(R,axis=1)/realization
#   SE_opt = np.sum(Ropt,axis=1)/realization

    plt.figure(figsize=(10,10)) 
    plt.plot(SNR_dB,SE,'r')       
    plt.show()
  
