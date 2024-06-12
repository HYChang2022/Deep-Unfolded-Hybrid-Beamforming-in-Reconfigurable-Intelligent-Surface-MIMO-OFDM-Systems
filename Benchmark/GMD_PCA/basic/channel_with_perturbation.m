function H_per = channel_with_perturbation(H,NMSE)
% NMSE = \sum_k||H-H_per||_F^2/\sum_k||H||_F^2 = KMN\Sigma^2/\sum_k||H||_F^2
% H_per = H+N, where N is complex Gaussian
[M,N,K] = size(H);
H_per = zeros(size(H));
nor = 0;
for k = 1:K
    nor = nor+norm(H(:,:,k),'fro')^2;
end
Sigma = sqrt(10^(NMSE/10)*nor/K/M/N);   %%%%%%%%%??????????????
for k = 1:K
    N0 = (randn(M,N)+1i*randn(M,N))/sqrt(2);
    H_per(:,:,k) = H(:,:,k)+Sigma*N0;
    H_per(:,:,k) = H_per(:,:,k)*norm(H(:,:,k),'fro')/norm(H_per(:,:,k),'fro');
end