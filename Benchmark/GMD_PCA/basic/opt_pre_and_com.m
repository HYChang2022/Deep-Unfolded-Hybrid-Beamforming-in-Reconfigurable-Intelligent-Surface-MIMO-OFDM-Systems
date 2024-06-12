function [F_opt,W_opt] = opt_pre_and_com(H,N_s,sigma2_n)%,rho

% parameters
N_t = size(H,2);
N_r = size(H,1);
K = size(H,3);

% find optimal F_opt
F_opt = zeros(N_t,N_s,K);
W_opt = zeros(N_r,N_s,K);
waterfilling = waterfilling_2D_matrix(H,N_s,sigma2_n);
for k = 1:K
    [U,~,V] = svd(H(:,:,k));
    F_opt(:,:,k) = V(:,1:N_s)*diag(waterfilling(:,k));
    W_opt(:,:,k) = U(:,1:N_s);
end

%[W_opt,~] = opt_combiner_MMSE(H,eye(N_r),F_opt,sigma2_n,rho);