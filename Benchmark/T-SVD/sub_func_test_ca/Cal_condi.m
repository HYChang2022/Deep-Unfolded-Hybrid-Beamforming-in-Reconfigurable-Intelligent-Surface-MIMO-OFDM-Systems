%% calculate condition number
% input 
%   cascade channel H
%   Ns
% output
%   condition number

function condi_num = Cal_condi(H,Ns)
[U,S,V] = svd(H);
s = diag(S);
s = s(1:Ns);
condi_num = real(s(1))/ real(s(end));
end
