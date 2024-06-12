function [W_MMSE,E_yy] = opt_combiner_MMSE(H,F_RF,F_BB,sigma2_n)

% parameters
N_r = size(H,1);
K = size(H,3);
N_s = size(F_BB,2);
% MMSE and E_yy
W_MMSE = zeros(N_r,N_s,K);
E_yy = zeros(N_r,N_r,K);
for k = 1:K
    F_tmp = H(:,:,k)*F_RF*F_BB(:,:,k);
%     W_MMSE(:,:,k) = ((F_tmp'*F_tmp+sigma2_n*N_s/rho*eye(N_s))\F_tmp'/sqrt(rho))';
    E_yy(:,:,k) = (F_tmp*F_tmp') + sigma2_n*eye(N_r);
    W_MMSE(:,:,k) = (F_tmp'/E_yy(:,:,k))';
end