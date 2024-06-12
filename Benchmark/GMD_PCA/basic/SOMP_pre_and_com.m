function [F_RF,F_BB,W_RF,W_BB] = SOMP_pre_and_com(H,A_t,A_r,N_s,N_t_RF,N_r_RF,sigma2_n,com,Q)

% parameters
N_t = size(H,2);
K = size(H,3);

% Initial checks
switch nargin
    case 8
        Q = 0;
    case 9
    otherwise
        disp('Error! Input arguments error in function bit_error!');
end
% find optimal F_opt
F_opt = zeros(N_t,N_s,K);
for k = 1:K
    [~,~,V] = svds(H(:,:,k));
    F_opt(:,:,k) = V(:,1:N_s);
end
[F_RF,~] = precoding_SOMP(F_opt,N_t_RF,N_s,A_t);
if Q
    F_RF = q_RF(F_RF,Q); 
end
for k = 1:K
    F_BB(:,:,k) = (F_RF'*F_RF)\F_RF'*F_opt(:,:,k);
    n = norm(F_RF*F_BB(:,:,k),'fro');
    if n>0
        F_BB(:,:,k) = sqrt(N_s)*F_BB(:,:,k)/n;
    end
end
if com
    % F_BB = reversed_F_BB(H,F_RF,N_s);
    % MMSE detaction on W --> W_MMSE
    [W_MMSE,E_yy] = opt_combiner_MMSE(H,F_RF,F_BB,sigma2_n);
    [W_RF,W_BB] = combining_SOMP(W_MMSE,N_r_RF,A_r,E_yy);
    if Q
        W_RF = q_RF(W_RF,Q);
    end
    for k = 1:K
        W_BB(:,:,k) = ((W_RF')*E_yy(:,:,k)*W_RF)\(W_RF')*E_yy(:,:,k)*W_MMSE(:,:,k);
    end
    % W_BB = reversed_W_BB(H,F_RF,F_BB,W_RF,W_BB);
else
    W_RF = NaN; W_BB = NaN;
end