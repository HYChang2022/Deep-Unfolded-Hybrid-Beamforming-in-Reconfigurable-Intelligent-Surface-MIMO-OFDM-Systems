function X = laplacernd(mu,sigma,size)
% This is a function to generate random number/matrix submit to Laplace
% distribution.
%
% Inputs:
%	mu: mean
%	sigma: standard deviation
%	size: size of the generate number/matrix X
switch nargin
    case 2
        p = rand(1,1)-0.5; 
    case 3
        p = rand(size)-0.5;
    otherwise
        disp('Error! Input arguments error in function laplacernd!');
end
b=sigma/sqrt(2);      %���ݱ�׼������Ӧ��b
X=mu-b*sign(p).*log(1-2*abs(p)); %���ɷ���������˹�ֲ����������