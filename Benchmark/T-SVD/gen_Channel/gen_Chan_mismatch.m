function [est_R,est_G,err_AoD_pG,err_AoA_pG,est_AR_G_irs,est_AT_G_bs,...
    est_AR_R_re,est_AT_R_irs]= gen_Chan_mismatch(Nr,Nt,M_h,M_v,...
   path_R,path_G,Rx_dBi,Tx_dBi,gain_G_path,AoD_G,AoA_G,gain_R_path,AoA_R,AoD_R,beta)  % % Tx-antenn gain BS

    
%%  Reference https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8809226


Tx = 10^(Tx_dBi/10);
Rx = 10^(Rx_dBi/10);

% beta = pi/90;  % 2бу


err_AoD_pG = -beta + 2*beta*rand(path_G,2);
err_AoA_pG = -beta + 2*beta*rand(path_G,1);


est_AoA_G = AoA_G + err_AoA_pG;
est_AoD_G = AoD_G + err_AoD_pG;

%% generate channel

% array response vector
kd = pi; M = M_h*M_v;

A_ula_t = @(x) sqrt(1/Nt)*exp( (0:Nt-1)'*1j*kd*sin(x));
A_ula_r = @(x) sqrt(1/Nr)*exp( (0:Nr-1)'*1j*kd*sin(x));
A_ura_M = @(x) sqrt(1/M)* (  kron(exp(1j*kd*(0:M_h-1)'*sin(x(1))*sin(x(2))),exp(1j*kd*(0:M_v-1)'*cos(x(2))) ) ) ; % UPA

%% BS-RIS channel
est_AT_G_bs = zeros(Nt,path_G);
est_AR_G_irs = zeros(M,path_G);
for i = 1:path_G
    aod = est_AoD_G(i);
    aoa = est_AoA_G(i,:);
    est_AT_G_bs(:,i) = A_ula_t(aod);
    est_AR_G_irs(:,i) = A_ura_M(aoa);
end
est_G = est_AR_G_irs*diag(gain_G_path)*est_AT_G_bs'; % Tx-antenn gain BS

%% RIS- UE channel
err_AoA_pR = -beta + 2*beta*rand(path_R,1);
err_AoD_pR = -beta + 2*beta*rand(path_R,2);

est_AoA_R = AoA_R + err_AoD_pR;
est_AoD_R = AoD_R + err_AoA_pR;

est_AT_R_irs = zeros(M,path_R); % IRS
est_AR_R_re = zeros(Nr,path_R); % transmitter
for i = 1:path_R
 
     aoa_r =est_AoA_R(i);
     aod_r = est_AoD_R(i,:);
    est_AR_R_re(:,i) = A_ula_r(aoa_r);
    est_AT_R_irs(:,i) = A_ura_M(aod_r);
end
est_R = est_AR_R_re*diag(gain_R_path)*est_AT_R_irs'; % Rx ---UE antenna gain
end