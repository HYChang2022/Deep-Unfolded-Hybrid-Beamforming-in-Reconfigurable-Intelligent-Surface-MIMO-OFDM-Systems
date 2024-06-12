function [IRS_phase] = phase_rotation_design(IRS_in,IRS_out,Nc)
%通过最优化乘积最大值选取IRS相位
[M,~]=size(IRS_in);
IRS_phase=zeros(M,M,Nc);
%只利用los径的角度取共轭
opt_theta=(IRS_in(:,1).*conj(IRS_out(:,1)));
amp=abs(opt_theta);
opt_phase=conj(opt_theta./amp(1));
value=(IRS_out')*diag(opt_phase)*IRS_in;
for k=1:Nc
    IRS_phase(:,:,k)=diag(opt_phase);
end
end

