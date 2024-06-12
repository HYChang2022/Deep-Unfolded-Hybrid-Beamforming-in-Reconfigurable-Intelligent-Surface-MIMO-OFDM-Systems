function qam64=QAM64_mod(bitseq,N)
%input bitseq: bits sequence
%input N: number of symbols
%output qam16: transmitted symbols
bitseq = bitseq(:).';
QAM_table1=[-7 -5 -1 -3 7 5 1 3];
for i=1:8
    for j=1:8
        QAM_table2(8*(j-1)+i)=QAM_table1(i)+1j*QAM_table1(j);
    end
end
QAM_table2=QAM_table2/sqrt(42);
if nargin<2
    N=floor(length(bitseq)/6); 
end
for n=1:N
   qam64(n) = QAM_table2(bitseq(6*n-[5 4 3 2 1 0])*[32;16;8;4;2;1]+1);
end