function [IRS_phase] = phase_rotation_design(IRS_in,IRS_out,Nc)
%ͨ�����Ż��˻����ֵѡȡIRS��λ
[M,~]=size(IRS_in);
IRS_phase=zeros(M,M,Nc);
%ֻ����los���ĽǶ�ȡ����
opt_theta=(IRS_in(:,1).*conj(IRS_out(:,1)));
amp=abs(opt_theta);
opt_phase=conj(opt_theta./amp(1));
value=(IRS_out')*diag(opt_phase)*IRS_in;
for k=1:Nc
    IRS_phase(:,:,k)=diag(opt_phase);
end
end

