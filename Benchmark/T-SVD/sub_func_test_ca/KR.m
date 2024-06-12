function C = KR(A,B)
%%  Khatri Rao product
[rA,cA] = size(A);
[rB,cB] = size(B);

if cA~=cB
    disp('error, the number of columns should be the same')
end

C = zeros(rA*rB,cA);

for i = 1:cA
    C(:,i) = kron(A(:,i),B(:,i));
end

end