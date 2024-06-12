function [X_hat] = QAM16_slicer(X,N)
%input X:received signal vector
%input N:number of signal
%output X_hat:received symbol vector
if nargin<2
    N = length(X); 
end
sq10=sqrt(10); b = [-2 0 2]/sq10; c = [-3 -1 1 3]/sq10;
Xr = real(X);  Xi = imag(X);
for i=1:N
   R(Xr<b(1)) = c(1);  I(Xi<b(1)) = c(1);
   R(b(1)<=Xr&Xr<b(2)) = c(2);  I(b(1)<=Xi&Xi<b(2)) = c(2);
   R(b(2)<=Xr&Xr<b(3)) = c(3);  I(b(2)<=Xi&Xi<b(3)) = c(3);
   R(b(3)<=Xr) = c(4);  I(b(3)<=Xi) = c(4);
end
X_hat = R + 1i*I;