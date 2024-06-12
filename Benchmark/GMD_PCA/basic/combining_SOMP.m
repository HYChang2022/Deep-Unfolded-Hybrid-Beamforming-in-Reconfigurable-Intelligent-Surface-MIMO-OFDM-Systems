function [W_RF,W_BB] = combining_SOMP(W_opt,N_r_RF,A_r,E_yy)
% This function is aim to realize SOMP under MIMO precoding scenario.

% Inputs:
%   W_opt: optimal F, have K pages.
%   N_r_RF: number of RF chains in transmitter, also seen as number of iterations.
%   A_r: steering matrix
%   E_yy: main of receiving signal

% Outputs:
%   W_RF,W_BB: design requirement, W_BB have K page.

% check inputs
switch nargin
    case 4  % without cl_indx, path can be chose within a same cluster
        indx = size(W_opt);
        K = indx(1,3);
        W_RF = [];
        W_res = W_opt;
        for i = 1:N_r_RF
            corr = [];
            for k = 1:K
                Phi = A_r'*E_yy(:,:,k)*W_res(:,:,k);
                corr(:,:,k) = Phi*(Phi');
            end
            [~,indx_max] = max(diag(sum(corr,3)));
            W_RF = [W_RF,A_r(:,indx_max)];
            W_BB = [];
            for k = 1:K
                W_BB(:,:,k) = ((W_RF')*E_yy(:,:,k)*W_RF)\(W_RF')*E_yy(:,:,k)*W_opt(:,:,k);
                W_res(:,:,k) = (W_opt(:,:,k) - W_RF*W_BB(:,:,k));
                n = norm(W_res(:,:,k),'fro');
                if n>0
                    W_res(:,:,k) = W_res(:,:,k)/n;
                end
            end
        end
    otherwise
        disp('Error: not enough arguments in function combining_SOMP!');return;
end