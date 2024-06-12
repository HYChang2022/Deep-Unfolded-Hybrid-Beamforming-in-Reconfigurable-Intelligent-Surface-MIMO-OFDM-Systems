function D = Q_codebook(N, Q, q)
% �µ�DFT�뱾��������ʵ�ֹ�����
% N Ϊ��������
% Q Ϊ��������
% a Ϊ����Ƕ�ϸ�ֳɳ̶�
%%
choose = 0; %��ʱ����
if nargin == 1
    q = N * 8;
    Q = N;
elseif nargin == 2
    q = 2^Q * q;
    Q = 2^Q;
else
%     disp('ok')
    if q < 0
        q = -q; choose = 0;
    end
    q = 2^Q * q;
    Q = 2^Q;

end

RF = DFT_set(N, log2(q)); % ������õ����뱾
if length(N) == 2
    N_total = N(1) * N(2);
else
    N_total = N;
end 

% ����
if choose == 1
    ang = angle(RF) + 0.0001; %�������������
    n = round(ang / 2 / pi * Q);
    RF = exp(1i * n / Q * 2 * pi) / sqrt(N_total);
    D = (unique(RF','rows'))';
elseif choose == 0
    D = RF;
end
