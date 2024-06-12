function [R,G,gain_G_path,AoD_G,AoA_G,AR_G_irs,AT_G_bs,...
    AR_R_re,gain_R_path,AT_R_irs,AoA_R,AoD_R]= gen_Chan_dBi_mis(Nr,Nt,M_h,M_v,...
    BS_loc,RIS_loc,UE_loc,K_factor,path_R,path_G,Rx_dBi,Tx_dBi)  % % Tx-antenn gain BS

%% 

if nargin < 1
    Rx_dBi = 0;%24.1;
    Tx_dBi = 0;%24.1;
    Nr = 64;
    Nt = 64;
    M_h = 16;
    M_v = 16;
    BS_loc = [5,0,10];
    RIS_loc = [0,90,10];
    UE_loc = [5,100,1.8];
    K_factor = 1;
    path_R = 4;
    path_G = 4;
end

%% input Tx dBi
%          Rx dBi
%          Nr
%          M% IRS端不考虑dBi
%          Nt

%          BS_loc
%          RIS_loc
%          UE_loc
%          K_factor--1,表示10dB

%          path_G
%          path_R
%% 定义path loss  % 不考虑RIS
Tx = 10^(Tx_dBi/10);
Rx = 10^(Rx_dBi/10);
% G = Tx* AR_G_irs*diag(gain_G_path)*AT_G_bs';

% G = 28e9; % 28GHz
c = 3e8; % light speed
PL_los = @(x) 61.4+2*10*log10(x)+ 5.8*randn(1);  %% BS-

PL_nlos = @(x)72.0+2.92*10*log10(x) + 8.7*randn(1);
% 计算距离

d_G = norm(RIS_loc - BS_loc);
d_R = norm(RIS_loc - UE_loc);

% 计算 channel gain
PL_G = PL_los(d_G);
PL_R = PL_los(d_R);

var_G = 10^(-PL_G/10);
var_R = 10^(-PL_R/10);

%% generate channel

% array response vector
kd = pi; M = M_h*M_v;

A_ula_t = @(x) sqrt(1/Nt)*exp( (0:Nt-1)'*1j*kd*sin(x));
A_ula_r = @(x) sqrt(1/Nr)*exp( (0:Nr-1)'*1j*kd*sin(x));
A_ura_M = @(x) sqrt(1/M)* (  kron(exp(1j*kd*(0:M_h-1)'*sin(x(1))*sin(x(2))),exp(1j*kd*(0:M_v-1)'*cos(x(2))) ) ) ; % UPA

%% BS-RIS channel G
gain_G_domi = sqrt(var_G/2)*(randn(1,1)+1j*randn(1,1)); 
gain_G_ndom = sqrt(10^(-K_factor)*var_G/2)*(randn(path_G-1,1)+1j*randn(path_G-1,1));
gain_G_path = sqrt(Nt*M/path_G)*[gain_G_domi;gain_G_ndom]; 
gain_G_path = Tx*sort(gain_G_path,'descend'); % yifang wanyi
AoD_G = zeros(path_G,1);
AoA_G = zeros(path_G,2);
AT_G_bs = zeros(Nt,path_G);
AR_G_irs = zeros(M,path_G);
for i = 1:path_G
    if i == 1
        aoa =zeros(2,1);  % x(1)---azimuth
        aoa(1) = asin( (RIS_loc(1)-BS_loc(1))/ (sqrt( (RIS_loc(1)-BS_loc(1))^2 + (BS_loc(2) - BS_loc(2))^2)));
        aoa(2) = acos ( (RIS_loc(3)-BS_loc(3))/d_G);
        aod = pi/2-aoa(1);
    else 
        aod = unifrnd(-pi/2,pi/2,1);
        aoa = unifrnd(-pi/2,pi/2,2,1);
    end
    AoD_G(i) = aod;
    AoA_G(i,:) = aoa;
    AT_G_bs(:,i) = A_ula_t(aod);
    AR_G_irs(:,i) = A_ura_M(aoa);
end
G = AR_G_irs*diag(gain_G_path)*AT_G_bs'; % Tx-antenn gain BS

%% RIS-UE channel

AT_R_irs = zeros(M,path_R); % IRS
AR_R_re = zeros(Nr,path_R); % transmitter
gain_R_domi = sqrt(var_R/2)*(randn(1,1)+1j*randn(1,1));
gain_R_ndom = sqrt(10^(-K_factor)*var_R/2)*(randn(path_R-1,1)+1j*randn(path_R-1,1));
% gain_R_ndom = sort(gain_R_ndom,'descend');
gain_R_path = sqrt(M*Nr/path_R)*[gain_R_domi;gain_R_ndom]; 
gain_R_path = Rx*sort( gain_R_path,'descend');
AoA_R = zeros(path_R,1);
AoD_R = zeros(path_R,2);
for i = 1:path_R
    if i == 1  %% LOS
        aod_r =zeros(2,1);  % x(1)---azimuth
        aod_r(1) = asin( (UE_loc(1)-RIS_loc(1))/ (sqrt( (UE_loc(1)-RIS_loc(1))^2 + (UE_loc(2)- RIS_loc(2))^2)));
        aod_r(2) = acos ( (UE_loc(3)-RIS_loc(3))/d_G);
        aoa_r = pi/2-aoa(1);
    else
        aoa_r = unifrnd(-pi/2,pi/2,1);
        aod_r = unifrnd(-pi/2,pi/2,2,1);
    end
    AoA_R(i) = aoa_r;
    AoD_R(i,:) = aod_r;
    AR_R_re(:,i) = A_ula_r(aoa_r);
    AT_R_irs(:,i) = A_ura_M(aod_r);
end
R = AR_R_re*diag(gain_R_path)*AT_R_irs'; % Rx --- UE antenna gain


