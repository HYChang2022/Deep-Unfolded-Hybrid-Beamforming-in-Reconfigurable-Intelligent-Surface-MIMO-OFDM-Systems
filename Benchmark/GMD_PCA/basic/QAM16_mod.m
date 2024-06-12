function qam16=QAM16_mod(bitseq,N)
%input bitseq: bits sequence
%input N: number of symbols
%output qam16: transmitted symbols
bitseq = bitseq(:).';  
QAM_table =[-3+3i, -1+3i, 3+3i, 1+3i, -3+i, -1+i, 3+i, 1+i,-3-3i, -1-3i, 3-3i, 1-3i, -3-i, -1-i, 3-i, 1-i]/sqrt(10);
if nargin<2
    N=floor(length(bitseq)/4); 
end
for n=1:N
   qam16(n) = QAM_table(bitseq(4*n-[3 2 1])*[8;4;2]+bitseq(4*n)+1);
end