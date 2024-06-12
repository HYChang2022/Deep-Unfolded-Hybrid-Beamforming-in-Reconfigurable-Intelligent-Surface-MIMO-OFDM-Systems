# -*- coding: utf-8 -*-
"""
Created on Tue May 23 16:05:07 2023

@author: Benson
"""

import numpy as np
import math
import scipy.io as sio
from array_response import array_response as AR

USPA = 'USPA' # USPA or ULA
ULA = 'ULA'

## System parameter
Ns = 2 # of streams
Nt = 32 # of transmit antennas
Nr = 32 # of receive antennas
N_phi = 16 # the RIS dimension
K = 16 # of subcarrier (OFDM)

## Channel parameter
los = 1
Nc = 5 # of clusters
Nray = 10 # of rays in each cluster 
L = Nc*Nray + 1
angles_sigma = (10/180)*math.pi # standard deviation of the angles in azimuth and elevation both of Rx and Tx 

D = 64 # length of cyclic prefix
Tau = D # maxinum time delay
channel_num = 2
realization = 1000

## Power distribution ratio (the ratio of the LOS and NLOS)
Mu = 10 # unit_dB

Mu_trans = 10**(-Mu/10)

## location setting
BS_loc = np.array([2, 0, 10])
RIS_loc = np.array([0, 148, 10])
UE_loc = np.array([5, 150, 1.8])

d_H1 = np.linalg.norm(RIS_loc-BS_loc) # distance between BS and RIS
d_H2 = np.linalg.norm(RIS_loc-UE_loc) # distance between RIS and UE

angles_sigma = (10/180)*math.pi # standard deviation of the angles in azimuth and elevation both of Rx and Tx
b = angles_sigma / np.sqrt(2) # Laplace parameter : scale(lambda or b)

gamma1 = math.sqrt((Nt*N_phi)/L) # normalization factor (BS-RIS)
gamma2 = math.sqrt((N_phi*Nr)/L) # normalization factor (RIS-UE)


# ## Path Loss
# PL_0 = 61.4 # (dB)
# PL_g = 2 # path loss exponent
# PL_H1 = PL_0 + 10*PL_g*np.log10(d_H1)
# PL_H2 = PL_0 + 10*PL_g*np.log10(d_H2)

## Without Path Loss

PL_H1 = 0
PL_H2 = 0

# Var_H1 = 10**(-PL_H1/10)
# Var_H2 = 10**(-PL_H1/10)

Var_H1 = 1 # channel_gain_H1
Var_H2 = 1 # channel_gain_H2

Tx_dBi = 0
Tx = 10**(Tx_dBi/10)
Rx_dBi = 0
Rx = 10**(Rx_dBi/10)


## initial setting
H1 = np.zeros((N_phi,Nt,K,realization),dtype = complex)
H2 = np.zeros((Nr,N_phi,K,realization),dtype = complex)

at_BS = np.zeros((Nt,Nc*Nray+1,realization),dtype = complex)
ar_RIS = np.zeros((N_phi,Nc*Nray+1,realization),dtype = complex)
at_RIS = np.zeros((N_phi,Nc*Nray+1,realization),dtype = complex)
ar_UE = np.zeros((Nr,Nc*Nray+1,realization),dtype = complex)

alpha = np.zeros((Nc*Nray+1,realization),dtype = complex)
beta = np.zeros((Nc*Nray+1,realization),dtype = complex)

for t in range(channel_num):
    if t == 0:
        for i in range(realization):
        
            AoD = np.zeros((Nc,2,Nray)) 
            AoA = np.zeros((Nc,2,Nray))
            AoD_new = np.zeros((Nc*Nray+1,2))
            AoA_new = np.zeros((Nc*Nray+1,2))
            # temp00 = np.zeros((Nt,Nc*Nray+1),dtype = complex)
            # temp01 = np.zeros((N_phi,Nc*Nray+1),dtype = complex)
            # temp_alpha =  np.zeros((Nc*Nray+1),dtype = complex)
            
            AoA_new[0,0] = np.arcsin((RIS_loc[0]-BS_loc[0])/np.sqrt((RIS_loc[0]-BS_loc[0])**2+(RIS_loc[1]-BS_loc[1])**2))  
            AoA_new[0,1] = np.arccos((RIS_loc[2]-BS_loc[2])/d_H1)
            AoD_new[0,0] = math.pi/2-AoA_new[0,0];
        
            for c in range(Nc):
                AoD_m = math.pi*np.random.rand(2,1)-math.pi/2 # cluster mean (-pi/2~pi/2)
                AoA_m = math.pi*np.random.rand(2,1)-math.pi/2 # cluster mean (-pi/2~pi/2)
            
                AoD[c,0,:] = np.random.laplace(AoD_m[0],b,(1,Nray))
                AoD[c,1,:] = np.random.laplace(AoD_m[1],b,(1,Nray))
                AoA[c,0,:] = np.random.laplace(AoA_m[0],b,(1,Nray))
                AoA[c,1,:] = np.random.laplace(AoA_m[1],b,(1,Nray))
                AoD_new[c*Nray+1:(c+1)*Nray+1,0] = np.reshape(AoD[c,0,:],Nray)
                AoD_new[c*Nray+1:(c+1)*Nray+1,1] = np.reshape(AoD[c,1,:],Nray)
                AoA_new[c*Nray+1:(c+1)*Nray+1,0] = np.reshape(AoA[c,0,:],Nray)
                AoA_new[c*Nray+1:(c+1)*Nray+1,1] = np.reshape(AoA[c,1,:],Nray)
            
            A_T1 = np.zeros((Nt,Nc*Nray+1),dtype = complex)   
            A_R1 = np.zeros((N_phi,Nc*Nray+1),dtype = complex) 
        
                    
            for l in range(Nc*Nray+1):
                at = AR(AoD_new[l,0],AoD_new[l,1],Nt,ULA)
                ar = AR(AoA_new[l,0],AoA_new[l,1],N_phi,USPA)
                A_T1[:,l] = np.reshape(at,Nt)
                A_R1[:,l] = np.reshape(ar,N_phi)
            

            
            alpha_i = np.zeros((Nc*Nray+1))
            gain_H1_domi = np.sqrt(Var_H1/2)*(np.random.randn(1,1)+1j*np.random.randn(1,1))
            gain_H1_ndomi = np.sqrt(Mu_trans*Var_H1/2)*(np.random.randn(Nc*Nray,1)+1j*np.random.randn(Nc*Nray,1))
            alpha_i = Tx*gamma1*np.append(gain_H1_domi,gain_H1_ndomi)
            
            alpha[:,i] = alpha_i
            
            # test_H1 = A_R@np.diag(alpha)@A_T.T
            
            alpha_ii = np.reshape(alpha_i,[Nc*Nray+1,1])
            
            tau = Tau*np.random.rand(Nc*Nray+1,1)
            H1_k_temp = np.zeros((N_phi,Nt,K), dtype = complex ) 
            
            for k in range(K):
                P_1 = alpha_ii * np.exp(-1j*2*np.pi*tau*k/K)
                alpha_i_k = np.reshape(P_1,Nc*Nray+1)
                H1_k_temp[:,:,k] = A_R1@ np.diag(alpha_i_k)@A_T1.T
                
        
            H1[:,:,:,i] = H1_k_temp   
            at_BS[:,:,i] = A_T1
            ar_RIS[:,:,i] = A_R1
            
            print('\r channel_H1: No. %d realization '%(i+1) ,end="",flush = True)    

    else:
        print('\n---------------------------------')
        
        for i in range(realization):
        
            AoD = np.zeros((Nc,2,Nray)) 
            AoA = np.zeros((Nc,2,Nray))
            AoD_new = np.zeros((Nc*Nray+1,2))
            AoA_new = np.zeros((Nc*Nray+1,2))
            # temp10 = np.zeros((N_phi,Nc*Nray+1),dtype = complex)
            # temp11 = np.zeros((Nr,Nc*Nray+1),dtype = complex)
            temp_beta =  np.zeros((Nc*Nray+1),dtype = complex)
            
            AoA_new[0,0] = np.arcsin((UE_loc[0]-RIS_loc[0])/np.sqrt((UE_loc[0]-RIS_loc[0])**2+(UE_loc[1]-RIS_loc[1])**2))  
            AoA_new[0,1] = np.arccos((UE_loc[2]-RIS_loc[2])/d_H2)
            AoD_new[0,0] = math.pi/2-AoA_new[0,0];
        
            for c in range(Nc):
                AoD_m = math.pi*np.random.rand(2,1)-math.pi/2 # cluster mean (-pi/2~pi/2)
                AoA_m = math.pi*np.random.rand(2,1)-math.pi/2 # cluster mean (-pi/2~pi/2)
            
                AoD[c,0,:] = np.random.laplace(AoD_m[0],b,(1,Nray))
                AoD[c,1,:] = np.random.laplace(AoD_m[1],b,(1,Nray))
                AoA[c,0,:] = np.random.laplace(AoA_m[0],b,(1,Nray))
                AoA[c,1,:] = np.random.laplace(AoA_m[1],b,(1,Nray))
                AoD_new[c*Nray+1:(c+1)*Nray+1,0] = np.reshape(AoD[c,0,:],Nray)
                AoD_new[c*Nray+1:(c+1)*Nray+1,1] = np.reshape(AoD[c,1,:],Nray)
                AoA_new[c*Nray+1:(c+1)*Nray+1,0] = np.reshape(AoA[c,0,:],Nray)
                AoA_new[c*Nray+1:(c+1)*Nray+1,1] = np.reshape(AoA[c,1,:],Nray)
        
            A_T2 = np.zeros((N_phi,Nc*Nray+1),dtype = complex)   
            A_R2 = np.zeros((Nr,Nc*Nray+1),dtype = complex) 
    
            for l in range(Nc*Nray+1):
                at = AR(AoD_new[l,0],AoD_new[l,1],N_phi,USPA)
                ar = AR(AoA_new[l,0],AoA_new[l,1],Nr,ULA)
                A_T2[:,l] = np.reshape(at,N_phi)
                A_R2[:,l] = np.reshape(ar,Nr)
        

        
            beta_i = np.zeros((Nc*Nray+1))
            gain_H2_domi = np.sqrt(Var_H2/2)*(np.random.randn(1,1)+1j*np.random.randn(1,1))
            gain_H2_ndomi = np.sqrt(Mu_trans*Var_H2/2)*(np.random.randn(Nc*Nray,1)+1j*np.random.randn(Nc*Nray,1))
            beta_i = Rx*gamma2*np.append(gain_H2_domi,gain_H2_ndomi)
            beta[:,i] = beta_i
            # test_H1 = A_R@np.diag(alpha)@A_T.T
            
            beta_ii = np.reshape(beta_i,[Nc*Nray+1,1])
            
            
            tau = Tau*np.random.rand(Nc*Nray+1,1)
            H2_k_temp = np.zeros((Nr,N_phi,K), dtype = complex ) 
            
            for k in range(K):
                P_2 = beta_ii * np.exp(-1j*2*np.pi*tau*k/K)
                beta_i_k = np.reshape(P_2,Nc*Nray+1)
                H2_k_temp[:,:,k] = A_R2@np.diag(beta_i_k)@A_T2.T
        
            H2[:,:,:,i] = H2_k_temp   
            at_RIS[:,:,i] = A_T2
            ar_UE[:,:,i] = A_R2
            print('\r channel_H2: No. %d realization '%(i+1) ,end="",flush = True)        



data_H1_H2 = {'H1' : H1, 'H2' : H2}

# data_H1 = {'H1' : H1}
# data_H2 = {'H2' : H2}

data_ar_RIS = {'ar_RIS' : ar_RIS}
data_at_RIS = {'at_RIS' : at_RIS}

data_alpha = {'alpha' : alpha}
data_beta = {'beta' : beta}


# save channel 
sio.savemat('.\\sparse_SV_channel_RIS\\testing_data\\H1_H2_Nt_%d_N_phi_%d_Nr_%d_Ns_%d_Testing_data_H.mat'%(Nt,N_phi,Nr,Ns),data_H1_H2)

# save steering vector 
sio.savemat('.\\sparse_SV_channel_RIS\\testing_data\\ar_RIS_%s_to_%s_Nt_%d_N_phi_%d_Ns_%d_Steering_vector_H.mat'%(ULA,USPA,Nt,N_phi,Ns),data_ar_RIS)
sio.savemat('.\\sparse_SV_channel_RIS\\testing_data\\at_RIS_%s_to_%s_N_phi_%d_Nr_%d_Ns_%d_Steering_vector_H.mat'%(USPA,ULA,N_phi,Nr,Ns),data_at_RIS)

# save channel gain (alpha/beta)
sio.savemat('.\\sparse_SV_channel_RIS\\testing_data\\alpha_%s_to_%s_Channel_gain_H.mat'%(ULA,USPA),data_alpha)
sio.savemat('.\\sparse_SV_channel_RIS\\testing_data\\beta_%s_to_%s_Channel_gain_H.mat'%(USPA,ULA),data_beta)



# test
Norm_H1 = np.zeros((K,realization),dtype = complex) 
Norm_H2 = np.zeros((K,realization),dtype = complex) 

for i in range(realization): 
    for k in range(K):
        Norm_H1[k,i] = np.linalg.norm(H1[:,:,k,i],'fro')
        Norm_H2[k,i] = np.linalg.norm(H2[:,:,k,i],'fro')
        



















