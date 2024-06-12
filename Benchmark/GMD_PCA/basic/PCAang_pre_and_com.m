function [F_RF,F_BB,W_RF,W_BB] = PCAang_pre_and_com(H,N_s,N_t_RF,N_r_RF,sigma2_n,com,Q)

% parameters
N_t = size(H,2);
N_r = size(H,1);
K = size(H,3);

% Initial checks
switch nargin
    case 6
        Q = 0;
    case 7
    otherwise
        disp('Error! Input arguments error in function bit_error!');
end

% find optimal F_opt
[F_opt,~] = opt_pre_and_com(H,N_s,sigma2_n);

F_RF = PCA_angle(F_opt,N_t_RF);
if Q
    F_RF = q_RF(F_RF,Q);
end
F_BB = reversed_F_BB(H,F_RF,N_s,sigma2_n);
if com
    % MMSE detaction on W --> W_MMSE
    [W_MMSE,E_yy] = opt_combiner_MMSE(H,F_RF,F_BB,sigma2_n);
    W_RF = PCA_angle(W_MMSE,N_r_RF,E_yy);
    if Q
        W_RF = q_RF(W_RF,Q);
    end
    W_BB = zeros(N_r_RF,N_s);
    for k = 1:K
        W_BB(:,:,k) = ((W_RF')*E_yy(:,:,k)*W_RF)\(W_RF')*E_yy(:,:,k)*W_MMSE(:,:,k);
    end
    % W_BB = reversed_W_BB(H,F_RF,F_BB,W_RF,W_BB);
else
    W_RF = NaN; W_BB = NaN;
end