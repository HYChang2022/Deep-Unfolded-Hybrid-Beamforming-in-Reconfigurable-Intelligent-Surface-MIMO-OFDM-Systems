function [F_RF,F_BB] = precoding_SOMP(F_opt,N_t_RF,N_s,A_t)
% This function is aim to realize SOMP under MIMO precoding scenario.

% Inputs:
%   F_opt: optimal F, have K pages.
%   N_t_RF: number of RF chains in transmitter, also seen as number of iterations.
%   N_s: number of data steam
%   A_t: steering matrix

% Outputs:
%   F_RF,F_BB: design requirement, F_BB have K page.

% check inputs
switch nargin
    case 4 
        indx = size(F_opt);
        K = indx(1,3);
        F_RF = [];
        F_res = F_opt;
        for i = 1:N_t_RF
            corr = [];
            for k = 1:K
                Phi = A_t'*F_res(:,:,k);%codebook dimension is too huge
                corr(:,:,k) = Phi*(Phi');
            end
            [~,indx_max] = max(diag(sum(corr,3)));
            F_RF = [F_RF,A_t(:,indx_max)];
            F_BB = [];
            for k = 1:K
                F_BB(:,:,k) = (F_RF'*F_RF)\F_RF'*F_opt(:,:,k);
                F_res(:,:,k) = F_opt(:,:,k) - F_RF*F_BB(:,:,k);
                n = norm(F_res(:,:,k),'fro');
                if n>0
                    F_res(:,:,k) = F_res(:,:,k)/n;
                end
            end
        end
        for k = 1:K
            n = norm(F_RF*F_BB(:,:,k),'fro');
            if n>0
                F_BB(:,:,k) = sqrt(N_s)*F_BB(:,:,k)/n;
            end
        end
    otherwise
        disp('Error: not enough arguments in function precoding_SOMP!');return;
end