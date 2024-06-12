%% demo test

clear all;
clc;

%% load channel 

% load channel H1 & H2

Nt = 32; % origin_Nt = 64
Nr = 32; % origin_Nr = 32 
M_h = 8; M_v = 8; M = M_h*M_v;% origin_M_h = origin_M_v = 8;
Nt_RF = 4; % origin_Nt_RF = 4
Nr_RF = 2; % origin_Nr_RF = 4
Ns = 2; % origin_Ns = 4
Nk = 16; % of carriers

% load channel
testing_data = load(sprintf('./sparse_SV_channel_RIS/testing_data/H1_H2_Nt_%d_N_phi_%d_Nr_%d_Ns_%d_Testing_data_H.mat',Nt,M,Nr,Ns));

% load steering vector (associated with RIS)
ar_RIS_data = load(sprintf('./sparse_SV_channel_RIS/testing_data/ar_RIS_ULA_to_USPA_Nt_%d_N_phi_%d_Ns_%d_Steering_vector_H.mat',Nt,M,Ns));
at_RIS_data = load(sprintf('./sparse_SV_channel_RIS/testing_data/at_RIS_USPA_to_ULA_N_phi_%d_Nr_%d_Ns_%d_Steering_vector_H.mat',M,Nr,Ns));

% load channel gain
alpha_data = load('./sparse_SV_channel_RIS/testing_data/alpha_ULA_to_USPA_Channel_gain_H.mat');
beta_data = load('./sparse_SV_channel_RIS/testing_data/beta_USPA_to_ULA_Channel_gain_H.mat');


channel_1 = testing_data.H1;
channel_2 = testing_data.H2;
ar_RIS = ar_RIS_data.ar_RIS;
at_RIS = at_RIS_data.at_RIS;
alpha = alpha_data.alpha;
beta = beta_data.beta;

Nc = 5; % of clusters
Nray = 10; % of rays in each cluster


path_G = Nc*Nray+1;
path_R = Nc*Nray+1;

realization = size(channel_1,4);


%% test (Created on Mon Mar 20 14:35:47 2023)

% fix SNR = 10dB
% snr_dB = 10;


snr_dB = [-10:5:20];
snr_lin = 10.^(snr_dB/10);
snr_len = length(snr_lin);

trans_Pt = Ns; 

% sigma2 = 1./snr_lin;

tic

% our method (MO => optimal Phi => effective channel)
R_FD_fc = zeros(snr_len,realization);
R_HB_fc = zeros(snr_len,realization);
R_HB_pc = zeros(snr_len,realization);
H_matrix = zeros(Nt,Nr,Nk,realization,snr_len);

for s = 1:snr_len
    snr = snr_lin(s);
    sigma2 = 1/snr_lin(s);
    
    for i = 1:realization
        fprintf('SNR=%d NO.%d \n ',snr_dB(s),i);
        G_i = channel_1(:,:,:,i);
        R_i = channel_2(:,:,:,i);
        AR_G_irs = ar_RIS(:,:,i);
        AT_R_irs = at_RIS(:,:,i);
        gain_G_path = alpha(:,i);
        gain_R_path = beta(:,i);
        [H_man,u] = test_Mani_0714(R_i,G_i,M,Ns,AR_G_irs,gain_G_path,gain_R_path,AT_R_irs,trans_Pt,sigma2,Nk);
        H_matrix(:,:,:,i,s) = H_man;
       
     % hybrid precoding
        mode2 = 2; % 
        snr = trans_Pt./sigma2./Ns;
        [Cap_opt_fc, Cap_hyb_fc, Cap_hyb_pc] = Cap_Approx_pow1(H_man,Nt_RF,Nr_RF,Ns,Nt,Nr,snr,mode2,Nk);
        R_FD_fc(s,i) = Cap_opt_fc/Nk;
        R_HB_fc(s,i) = Cap_hyb_fc/Nk;
        
        R_HB_pc(s,i) = Cap_hyb_pc/Nk;

%         Condi_man = Cal_condi(H_man,Ns);
    end
    ASE_FD_fc(s)=sum(R_FD_fc(s,:))/realization;
    ASE_HB_fc(s)=sum(R_HB_fc(s,:))/realization;
    ASE_HB_pc(s)=sum(R_HB_pc(s,:))/realization;
    fprintf('------------------------------------\n');
end
time = toc
average_time = time/(realization*snr_len);
% Plot figure
figure(1); 
SNR = linspace(-10,20,7);
hold on;
plot(SNR,ASE_FD_fc,'-^','color',[1 0 0],'LineWidth',1.5);
plot(SNR,ASE_HB_fc,'-o','color',[0 1 0],'LineWidth',1.5);
plot(SNR,ASE_HB_pc,'-square','color',[0 0 1],'LineWidth',1.5);

axis([-10, 20, 0, 50]);
set(gca,'FontSize',12);
title('Average Spectral Efficiency vs. SNR Performance (N_{RIS} = 64)','FontSize',14);
xlabel('SNR (dB)','FontSize',14);
ylabel('Average Spectral Efficiency (bit/s/Hz)','FontSize',14);
legend('ASE-FD-SVD(fc)','ASE-HB(fc)','ASE-HB(pc)','FontSize',12);

% filepath = 'D:/code/sparse_SV_channel_RIS/WMMSE_MO/';
% filename = 'RIS_channel_T_SVD.mat';
% save(fullfile(filepath, filename),'H_matrix'); % size(H_matrix) = (Nt,Nr,Nk,realization,snr_len);

grid on;
display(abs(ASE_FD_fc));
display(abs(ASE_HB_fc));
display(abs(ASE_HB_pc));
