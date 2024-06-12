function RF = PCA_angle(opt,N_RF,weight)

% parameters
K = size(opt,3);
N = size(opt,1);
N_s = size(opt,2);

% check nargin
switch nargin
    case 2;
    case 3
        % weight
        for k = 1:K
            opt(:,:,k) = weight(:,:,k)*opt(:,:,k);%^0.5
        end
    otherwise
        disp('Error! Input argument error in PCA_angle!');
        return
end

matrix = reshape(opt,N,N_s*K);
[basis,~,~] = svd(matrix);
RF = exp(1i*angle(basis(:,1:N_RF)))/sqrt(N);