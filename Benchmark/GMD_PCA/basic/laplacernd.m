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
b=sigma/sqrt(2);      %根据标准差求相应的b
X=mu-b*sign(p).*log(1-2*abs(p)); %生成符合拉普拉斯分布的随机数列