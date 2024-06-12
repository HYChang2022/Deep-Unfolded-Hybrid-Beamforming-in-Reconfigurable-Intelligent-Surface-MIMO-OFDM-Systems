
close all;
clear all;
clc;
%% 与main_irs_upa只有载波数Nc和仿真次数Nsym不同不同


%% Parameter setting
% load channel H1 & H2


fc=28e9; % Frequencey
lamada=3e8/fc; % wavelegenth;


Nt = 32; % origin_Nt = 64
Nr = 32; % origin_Nr = 32 
Num_IRS = [8 8];
NumIRS = Num_IRS(1,1)*Num_IRS(1,2); % origin_M_h = origin_M_v = 8;
Nt_RF = 4; % origin_Nt_RF = 4
Nr_RF = 2; % origin_Nr_RF = 4
N_s = 2; % origin_Ns = 4
Nc = 16; % of carriers

N_cl = 5; % of clusters
Nray = 10; % of rays in each cluster

%% load channel
testing_data = load(sprintf('./sparse_SV_channel_RIS/testing_data/H1_H2_Nt_%d_N_phi_%d_Nr_%d_Ns_%d_Testing_data_H.mat',Nt,NumIRS,Nr,N_s));

% load steering vector (associated with RIS)
ar_RIS_data = load(sprintf('./sparse_SV_channel_RIS/testing_data/ar_RIS_ULA_to_USPA_Nt_%d_N_phi_%d_Ns_%d_Steering_vector_H.mat',Nt,NumIRS,N_s));
at_RIS_data = load(sprintf('./sparse_SV_channel_RIS/testing_data/at_RIS_USPA_to_ULA_N_phi_%d_Nr_%d_Ns_%d_Steering_vector_H.mat',NumIRS,Nr,N_s));

% load channel gain
alpha_data = load('./sparse_SV_channel_RIS/testing_data/alpha_ULA_to_USPA_Channel_gain_H.mat');
beta_data = load('./sparse_SV_channel_RIS/testing_data/beta_USPA_to_ULA_Channel_gain_H.mat');


channel_1 = testing_data.H1;
channel_2 = testing_data.H2;
ar_RIS = ar_RIS_data.ar_RIS;
at_RIS = at_RIS_data.at_RIS;
alpha = alpha_data.alpha;
beta = beta_data.beta;


path_G = Nc*Nray+1;
path_R = Nc*Nray+1;

realization = size(channel_1,4);

%% system parameter
% add_path();
% Nsym = 300; %number of symbols
% N_RF = 4; %number of RF chains
% %  SNR_dB=20;

% SNR_dB=[-10:5:20]; % SNR range
%% test (Created on Mon Mar 20 14:35:47 2023)
SNR_dB = [-10:5:20];

trans_Pt = N_s; 
channel_power = trans_Pt;
% channel_power = 1;

%%
R_hybrid_PCA=zeros(1,length(SNR_dB));

tic

for i=1:length(SNR_dB)
    SNRdB = SNR_dB(i);
    
    sigma2 = 10^(-SNRdB/10);%generate channel noise convarience

    snr = trans_Pt./sigma2./N_s;
     
    for R_num=1:realization
        H1 = channel_1(:,:,:,R_num);
        H2 = channel_2(:,:,:,R_num);
        A_irs_r = ar_RIS(:,:,R_num);
        A_irs_t = at_RIS(:,:,R_num);
        gain_G_path = alpha(:,R_num);
        gain_R_path = beta(:,R_num);
        %% transmit part
%         bitseq=randi([0 1],b,N_s);
%         symbol_data=bitseq(:)';
%         symbol = QAM16_mod(symbol_data,N_s);
%         x=symbol.';
        %% mmWave channel matrix generation
%         [H, A_BS,A_MS] = channel_f(Nt, Nr, 1, Nc);
%       这里的信道根据Los/Nlos径略有修改信道模型的line86-92
%         [H1,A_t,A_irs_r] = freqency_sparse_SV_channel0(Nc,N_cl,N_ray, sigma_2_alpha, sigma_ang,Nt,NumIRS,los);
%         [H2,A_irs_t,A_r] = freqency_sparse_SV_channel0(Nc,N_cl,N_ray, sigma_2_alpha, sigma_ang,NumIRS,Nr,los);
%        
        %% IRS design
        H_IRS=phase_rotation_design(A_irs_r,A_irs_t,Nc);
        %equivalent channel
        H=zeros(Nr,Nt,Nc); H_temp=zeros(Nr,Nt,Nc); 
        for eqv=1:Nc

            H(:,:,eqv)=H2(:,:,eqv)*H_IRS(:,:,eqv)*H1(:,:,eqv);
%             H_temp(:,:,eqv)=H3(:,:,eqv);
        end
        %%   RF precoder/combiner
        %% RF PCA
        [F_RF_PCA,PCAang_F_BB, W_RF_PCA,PCAang_W_BB] = PCAang_pre_and_com(H,N_s,Nt_RF,Nr_RF,sigma2,1,0);

        %%  baseband precoding    
        for carrier=1:Nc
            [U,S,V] = svd(H(:,:,carrier));
            V_1 = V(:,1:N_s);
            total_power = N_s/sigma2;
            power_allo_equal = eye(N_s)*(total_power/N_s); % equal power allocation
            %% digital GMD
            [G(:,:,carrier),M,D(:,:,carrier)] = gmd(U,S(:,1:N_s),V(:,1:N_s));
            GH(:,:,carrier) = (G(:,:,carrier))';
            QQ=(GH(:,:,carrier));
            G_1(:,:,carrier) = QQ(1:N_s,:);
            %% hybrid GMD
            [U1,S1,V1] = svd(W_RF_PCA'*H(:,:,carrier)*F_RF_PCA);
            [G_PCA(:,:,carrier),M1,D1(:,:,carrier)] = gmd(U1,S1,V1);
         
            

        end   
        %%%%%%%%%%%%%%%%%%%%spectrum efficiency%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

       R_hybrid_PCA(:,i)= R_hybrid_PCA(:,i)+SE_BER(2, H, G_PCA, W_RF_PCA, D1, F_RF_PCA, channel_power, snr);
       
       fprintf('SNR=%d dB, Realization=%d\n',SNR_dB(i),R_num);
    end

end

R_hybrid_PCA = R_hybrid_PCA/realization;
  
%%  SE PLOT

plot(SNR_dB, R_hybrid_PCA ,'b-o','Linewidth',1.5)
hold on
xlabel('SNR (dB)')
ylabel('Spectrum Efficiency(bps/Hz)')

legend('PCA with RIS');
grid on

time = toc
average_time_PCA = time/(realization*length(SNR_dB));
display('ASE=');
display(R_hybrid_PCA);
display(average_time_PCA);


%% test channel power (H1 & H2)
Norm_H1 = zeros(Nc,1);
Norm_H2 = zeros(Nc,1);
Norm_phi = zeros(Nc,1);
Norm_Heff = zeros(Nc,1);
Norm_F_BB = zeros(Nc,1);

for k= 1:Nc
    Norm_H1(k,1) = norm(H1(:,:,k),'fro');
    Norm_H2(k,1) = norm(H2(:,:,k),'fro');
    Norm_phi(k,1) = norm(H_IRS(:,:,k),'fro');     
    Norm_Heff(k,1) = norm(H(:,:,k),'fro');
    Norm_F_BB(k,1) = norm(PCAang_F_BB(:,:,k),'fro');

end


