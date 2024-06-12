function X_hat=QAM64_slicer(X,N)
%input X:received signal vector
%input N:number of signal
%output X_hat:received symbol vector
if nargin<2
    N = length(X); 
end
sq42=sqrt(42); b = [-6 -4 -2 0 2 4 6]/sq42; c = [-7 -5 -3 -1 1 3 5 7]/sq42;
Xr = real(X);  Xi = imag(X);
for i=1:N
   R(Xr<b(1)) = c(1);  I(Xi<b(1)) = c(1);
   R(b(1)<=Xr&Xr<b(2)) = c(2);  I(b(1)<=Xi&Xi<b(2)) = c(2);
   R(b(2)<=Xr&Xr<b(3)) = c(3);  I(b(2)<=Xi&Xi<b(3)) = c(3);
   R(b(3)<=Xr&Xr<b(4)) = c(4);  I(b(3)<=Xi&Xi<b(4)) = c(4);
   R(b(4)<=Xr&Xr<b(5)) = c(5);  I(b(4)<=Xi&Xi<b(5)) = c(5);
   R(b(5)<=Xr&Xr<b(6)) = c(6);  I(b(5)<=Xi&Xi<b(6)) = c(6);
   R(b(6)<=Xr&Xr<b(7)) = c(7);  I(b(6)<=Xi&Xi<b(7)) = c(7);
   R(b(7)<=Xr) = c(8);  I(b(7)<=Xi) = c(8);
end
X_hat = R' + 1i*I';