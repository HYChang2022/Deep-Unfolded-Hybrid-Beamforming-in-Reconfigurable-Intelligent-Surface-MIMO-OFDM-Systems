function F_BB = reversed_F_BB(H,F_RF,N_s,sigma2_n)
% Based on the Frequency Selective Hybrid Precoding for Limited Feedback
% mmWave Systems
%
% Inputs:
%   H: channel matrix
%   F_RF: RF precoder

% parameters
N_t = size(H,1);
N_RF = size(F_RF,2);
K = size(H,3);
F_BB = zeros(N_RF,N_s,K);

H_eff = zeros(N_t,N_RF,K);
for k = 1:K
    [~,S,V] = svd(H(:,:,k));
    H_eff(:,:,k) = S*V'*F_RF*(F_RF'*F_RF)^(-0.5);
end
waterfilling = waterfilling_2D_matrix(H_eff,N_s,sigma2_n);
for k = 1:K
    [~,~,v] = svd(H_eff(:,:,k));
    F_BB(:,:,k) = (F_RF'*F_RF)^(-0.5)*v(:,1:N_s)*diag(waterfilling(:,k));
end
