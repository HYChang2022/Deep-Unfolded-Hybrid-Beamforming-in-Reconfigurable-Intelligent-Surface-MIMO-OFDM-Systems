# -*- coding: utf-8 -*-
"""
Created on Tue Oct  17 12:47:38 2023

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

antenna_array = 'ULA'
Nt = 32
Nr = 32

Ns = 2
Nrf_t = 4
Nrf_r = 4 # original_Nrf_r = 4
Mt = int(Nt/Nrf_t)
Mr = int(Nr/Nrf_r)
Nk = 16
Iter = 6
N_phi = 64 # the number of RIS (RIS_dim = N_phi*N_phi)


def Phi_process(R_i,G_i,N_phi,Ns,AR_G_irs,gain_G_path,gain_R_path,AT_R_irs,trans_Pt,n_power,Nk):
    R = R_i
    G = G_i
    snr = trans_Pt/n_power/Ns
    gain2_RG = np.zeros((Ns,1))
    Phi = np.zeros((N_phi,Ns))
    m = Ns
    v_div = np.zeros((N_phi))
    mM = np.floor(N_phi/m) ## must be an integer
    mM = np.int(mM)
    for i_b in range(Ns):
        alpha_i = gain_G_path[i_b]
        beta_i = gain_R_path[i_b]
        gain2_RG[i_b] = np.abs(alpha_i*beta_i)**2 # truncated SVD: take the first Ns elements of D(i,i)
        aT_irs = np.diag(AT_R_irs[:,i_b].conj().T)
        ar_irs = AR_G_irs[:,i_b]
        atr_ii = aT_irs @ ar_irs 
        Phi[:,i_b] = atr_ii
        ang_atr = np.arctan2(np.imag(atr_ii), np.real(atr_ii))
        A = ang_atr[(i_b)*mM:((i_b+1)*mM-1)]
        v_div[(i_b)*mM:((i_b+1)*mM-1)] = np.exp(1j*ang_atr[(i_b)*mM:((i_b+1)*mM-1)]) 
    
    
    epsilon_sig = snr*gain2_RG
    
    def RIS_MO(Phi,N_phi,Ns,epsilon_sig,v_div):    
       
        manifold = ComplexCircle(N_phi)
        
        def mycost(v): 
            obj = 0
            for it in range(Ns): # deal with subcarriers separately
                ppi = Phi[:,it]
                PPi = ppi@ppi.conj().T
                tP = v.conj().T*PPi*v
                ppt = np.real(1+ epsilon_sig[it]*tP)
                obj = obj - np.log2(ppt)
                
            return obj   
        
        def myegrad(v): # v(64,)
            objg = 0
            for it in range(Ns):
                ppi = Phi[:,it] # ppi(64,)
                PPi = ppi@ppi.conj().T # PPi(64,64)
                tP = v.conj().T*PPi*v # 
                ppt = np.real(1+ epsilon_sig[it]*tP)  
                
                objg = objg - (2/(np.log(2)*ppt))*(epsilon_sig[it]*PPi*v)
                
            return objg
        
        problem = Problem(manifold = manifold, cost = mycost, egrad = myegrad)
        
        solver = ConjugateGradient()
        
       
        y = solver.solve(problem = problem,x = v_div)  
        y = np.reshape(y,(N_phi,1))
       
        
        return y
        
    v_man = RIS_MO(Phi,N_phi,Ns,epsilon_sig,v_div)
    
    
   
    H_man = np.zeros((Nr,Nt,Nk))
    v_man = np.reshape(v_man,(-1))
    A = np.diag(v_man.conj().T)
    B = v_man.conj().T
   
    for k in range(Nk):
        H_man[:,:,k] = R[:,:,k]@np.diag(v_man.conj())@G[:,:,k]
    
      
    return H_man, v_man



def Frf_MO(Frf,Beta,G_tilde_h,Lam):
    
    A = np.ones((Mt,1))
    F_mask = np.zeros((Nt,Nrf_t))

    for i in range(Nrf_t):
        F_mask[i*Mt:(i+1)*Mt,i] = A[:,0]
        
    manifold = ComplexCircle(Nt*Nrf_t)
    
    def mycost(x):
        X = np.reshape(x,(Nt,Nrf_t)) # order need more check
      
        f = 0
        for k in range(Nk):
            Lam_k = Lam[:,:,k]
            G_tilde_k_h = G_tilde_h[:,:,k]
            Beta_k = Beta[0,k]
            f = f + np.trace(np.linalg.inv(np.linalg.inv(Lam_k)+G_tilde_k_h.conj().T@X@X.conj().T@G_tilde_k_h/Beta_k))

        return f
    
    def myegrad(x):
        X = np.reshape(x,(Nt,Nrf_t)) # order need more check
       
        eg = np.zeros((Nt,Nrf_t))
        for k in range(Nk):
            Lam_k = Lam[:,:,k]
            G_tilde_k_h = G_tilde_h[:,:,k]
            Beta_k = Beta[0,k]
            M = np.linalg.inv(Lam_k) + G_tilde_k_h.conj().T@X@X.conj().T@G_tilde_k_h/Beta_k
            
            eg = eg + G_tilde_k_h@np.linalg.inv(M)@np.linalg.inv(M)@G_tilde_k_h.conj().T@X/Beta_k
        
        eg = -(eg)*F_mask            
        eg = np.reshape(eg,(-1,))
        
        return eg
    
    problem = Problem(manifold = manifold,cost=mycost,egrad=myegrad)
    
    solver = ConjugateGradient()
    
    y = solver.solve(problem = problem,x = np.reshape(Frf,(-1,)))
    y = np.reshape(y,(Nt,Nrf_t))
   
    return y

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
    
    y = solver.solve(problem = problem,x = np.reshape(Wrf,(-1,)))
    y = np.reshape(y,(Nr,Nrf_r))
    
    return y

#---------------------------------------------------------------------------
#  WMMSE_MO_iterative
#--------------------------------------------------------------------------- 


def WMMSE_MO(para):
    H,n_power = para
    
    Alpha = np.zeros((1,Nk))
    Beta = np.zeros((1,Nk))
    xi = np.zeros((1,Nk))
    G_tilde_h = np.zeros((Nt,Ns,Nk),dtype=np.complex128)
    G = np.zeros((Nr,Ns,Nk),dtype=np.complex128)
    wmmse = np.zeros((Nk))
    
    A = np.ones((Mt,1))
    B = np.ones((Mr,1))
    F_mask = np.zeros((Nt,Nrf_t))
    W_mask = np.zeros((Nr,Nrf_r))
    
    for i in range(Nrf_t):
        F_mask[i*Mt:(i+1)*Mt,i] = A[:,0]
        
    for i in range(Nrf_r):
        W_mask[i*Mr:(i+1)*Mr,i] = B[:,0]        
        
    Frf = np.exp(1j*np.random.uniform(0,2*np.pi,(Nt,Nrf_t)))*F_mask
    Wrf = np.exp(1j*np.random.uniform(0,2*np.pi,(Nr,Nrf_r)))*W_mask
    

    Fbb = np.random.randn(Nrf_t,Ns,Nk) + 1j*np.random.rand(Nrf_t,Ns,Nk)
    Wbb = np.random.randn(Nrf_r,Ns,Nk) + 1j*np.random.rand(Nrf_r,Ns,Nk)
   
    Lam = np.zeros((Ns,Ns,Nk),dtype=np.complex128) # weighted matrix
    
    for i in range(Nk):
        Lam[:,:,i] = np.eye(Ns,dtype=np.complex128)
    
    
    I = 0
    delta = 1
    new_wmmse = 10
    while I<Iter and delta >= 10**(-4):
        # Frf base on MO =============================================================================================================
        for k in range(Nk):
            Beta[0,k] = n_power*Nt*Nr/Nrf_t/Nrf_r*np.trace(Lam[:,:,k]@Wbb[:,:,k].conj().T@Wbb[:,:,k])            
            G_tilde_h[:,:,k] = H[:,:,k].conj().T@Wrf@Wbb[:,:,k]
        Frf = Frf_MO(Frf,Beta,G_tilde_h,Lam)  
          
        # Fbb_k ======================================================================================================================
        for k in range(Nk):
            Beta_k = Beta[0,k]  
            G_tilde_k_h = G_tilde_h[:,:,k]
            F_tilde = Frf.conj().T@G_tilde_k_h@Lam[:,:,k]@G_tilde_k_h.conj().T@Frf + Beta_k*np.eye(Nrf_t)
            xi_k = 1/np.sqrt(Nt/Nrf_t/Ns*(np.linalg.norm(np.linalg.inv(F_tilde)@Frf.conj().T@G_tilde_k_h@Lam[:,:,k],ord='fro')**2))
            
            Fbb[:,:,k] = xi_k*np.linalg.inv(F_tilde)@Frf.conj().T@G_tilde_k_h@Lam[:,:,k]
            xi[0,k] = xi_k
        
        # Wrf base on MO =============================================================================================================
        for k in range(Nk):
            xi_k = xi[0,k]
            Alpha[0,k] = n_power*Nr*Nrf_r/xi_k/xi_k
            G[:,:,k] = H[:,:,k]@Frf@Fbb[:,:,k]/xi_k
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
        
    return Frf,Fbb,Wrf,Wbb

if __name__ == '__main__' :
    

    # load channel
    testing_data = sio.loadmat('.\\sparse_SV_channel_RIS\\testing_data\\H1_H2_Nt_%d_N_phi_%d_Nr_%d_Ns_%d_Testing_data_H.mat'%(Nt,N_phi,Nr,Ns))
    
    # load steering vector (associated with RIS)
    ar_RIS_data = sio.loadmat('.\\sparse_SV_channel_RIS\\testing_data\\ar_RIS_ULA_to_USPA_Nt_%d_N_phi_%d_Ns_%d_Steering_vector_H.mat'%(Nt,N_phi,Ns))
    at_RIS_data = sio.loadmat('.\\sparse_SV_channel_RIS\\testing_data\\at_RIS_USPA_to_ULA_N_phi_%d_Nr_%d_Ns_%d_Steering_vector_H.mat'%(N_phi,Nr,Ns))
    
    # load channel gain
    alpha_data = sio.loadmat('.\\sparse_SV_channel_RIS\\testing_data\\alpha_ULA_to_USPA_Channel_gain_H.mat')
    beta_data = sio.loadmat('.\\sparse_SV_channel_RIS\\testing_data\\beta_USPA_to_ULA_Channel_gain_H.mat')
    
    
    channel_1 = testing_data['H1']
    channel_2 = testing_data['H2']
    ar_RIS = ar_RIS_data['ar_RIS']
    at_RIS = at_RIS_data['at_RIS']
    alpha = alpha_data['alpha']
    beta = beta_data['beta']
    
    realization = channel_1.shape[3]
    
    trans_Pt = Ns      

    SNR_dB = np.array(range(-10,25,5))    
    
    SNR_lin = 10**(SNR_dB/10)    
    SNR_len =len(SNR_lin)

    R = np.zeros([SNR_len,realization])
    exe_time_avg = np.zeros((SNR_len))
    print('start testing')
    
    for s in range(SNR_len):
        start = time.perf_counter()    
        snr = SNR_lin[s]
        n_power = 1/snr

        for i in range(realization):
            # print('\rSNR=%d No. %d '%(SNR_dB[s],i) , end='',flush=True)
            G_i = channel_1[:,:,:,i]
            R_i = channel_2[:,:,:,i]
            AR_G_irs = ar_RIS[:,:,i]
            AT_R_irs = at_RIS[:,:,i]
            gain_G_path = alpha[:,i]
            gain_R_path = beta[:,i]
            H,v_phi = Phi_process(R_i,G_i,N_phi,Ns,AR_G_irs,gain_G_path,gain_R_path,AT_R_irs,trans_Pt,n_power,Nk)
            
            para_WMMSE_MO = (H,n_power)
            Frf,Fbb,Wrf,Wbb = WMMSE_MO(para_WMMSE_MO)
            
            for k in range(Nk):
                Ft = Frf@Fbb[:,:,k]
                Ft_h = Ft.conj().T
                
                Wt = Wrf@Wbb[:,:,k]
                Wt_h = Wt.conj().T
                
                
                pinv_Wt = np.linalg.inv(Wt_h@Wt + 1e-10)@Wt_h
                     
                R[s,i] =  R[s,i] + np.log2(np.linalg.det(np.eye(Ns)+snr*pinv_Wt@H[:,:,k]@Ft@Ft_h@H[:,:,k].conj().T@Wt))
                         
            R[s,i] = R[s,i]/Nk 
            
        elapsed = time.perf_counter() - start            
        exe_time_avg[s] =  elapsed/realization
        print('time = %f'%(exe_time_avg[s]))
        
        
# Spectral efficiency
    SE = np.sum(R,axis=1)/realization


    plt.figure(figsize=(10,10)) 
    plt.plot(SNR_dB,SE,'r')       
    plt.show()

