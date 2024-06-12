function [s] =QAM64_demapper(qam64,N) 
%input qam: received symbols
%input N: number os symbols
%output s: bits sequence
if nargin<2
    N = length(qam64); 
end
QAM_table1=[-7 -5 -1 -3 7 5 1 3];
for i=1:8
    for j=1:8
        QAM_table2(8*(j-1)+i)=QAM_table1(i)+1j*QAM_table1(j);
    end
end
QAM_table2=QAM_table2/sqrt(42);
temp = [];
for n=0:N-1
   temp=[temp dec2bin(find(QAM_table2==qam64(n+1))-1,6)]; 
end
%s=zeros(1,length(temp));
for n=1:length(temp)
   s(n)=bin2dec(temp(n));
end