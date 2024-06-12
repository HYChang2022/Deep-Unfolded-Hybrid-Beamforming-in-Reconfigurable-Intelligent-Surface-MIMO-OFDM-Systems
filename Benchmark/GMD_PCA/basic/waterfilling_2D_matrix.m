function Lambda = waterfilling_2D_matrix(H,N_s,noise)
% This function is to create water filling solution matrix in both
% frequency domain and data stream domain.
% Reference: Dynamic Subarrays for Hybrid Precoding in Wideband mmWave MIMO
% Systems (16) & (17).

% Inputs:
%   H: multicarrier channel matrix
%   N_s: number of data stream

% Outputs:
%   Lambda: waterfilling solution matrix

K = size(H,3);
Lambda = zeros(1,N_s*K);
lambda = zeros(N_s,K);

if sum(sum(isnan(H)))
    Lambda = ones(1,N_s*K);
else
    for k = 1:K
        [~,d,~] = svd(H(:,:,k));
        d = diag(d);
        lambda(:,k) = d(1:N_s,1)';
    end
    lambda = reshape(lambda,1,[]);

    bot = power(1./lambda,2)*noise;
    bot = sort(bot);
    sum_p = 0;
    for k = 1:K*N_s-1
        sum_p = sum_p+k*(bot(1,k+1)-bot(1,k));
        if sum_p>N_s*K
            break;
        end
    end
    level = bot(1,k+1)-(sum_p-N_s*K)/k;
    if sum_p<N_s*K
        level = bot(1,K*N_s)+(N_s*K-sum_p)/K/N_s;
    end
    for k = 1:K*N_s
        a = level-noise/(lambda(1,k)^2);
        if a>0
            Lambda(1,k) = sqrt(a);
        end
    end
    Lambda = reshape(Lambda,N_s,K);
end