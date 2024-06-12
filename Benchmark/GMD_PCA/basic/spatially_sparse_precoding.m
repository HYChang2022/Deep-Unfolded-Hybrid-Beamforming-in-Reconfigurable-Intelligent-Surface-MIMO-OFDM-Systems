function [F,F_BB]=spatially_sparse_precoding(Nrf,N_s,H,At,X,carrier, F_union_opt,sigma2)

    %%%%%%%%%%%%consider broadband%%%%%%%%%%
  
%     F_union_opt=[];
%     for subcarrier=1:Nc
%         [~,~,V] = svd(H(:,:,subcarrier));
%         F_union_opt =[F_union_opt  V(:,1:Nrf)];
%     end
%%%%%共同RFcominer
%     [U_opt,~,~]=svd(F_union_opt);
% 
%     F2=sqrt(1/Nt)*exp(1i*angle(U_opt(:,1:Nrf)));
%     F2=sqrt(1/Nt)*exp(1i*angle(V(:,1:Nrf)));
%%%%%%%%子载波独立combiner
   [~,~,U_opt] = svd(H(:,:,carrier));
    F_opt=U_opt(:,1:N_s);   
  
%%%%单个子载波
%     F_opt=V(:,1:N_s);
   %%%%%%%%%%%%end%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     for i=1:Nc
%         [~,~,V_digital]=svd(H(:,:,i));
%         F_opt=V_digital(:,1:N_s);
%         F_BB=inv(F'*F)*F'*F_opt;
%         F_BB = sqrt(N_s)*F_BB/norm(F*F_BB,'fro');
%     end

   
 F=[]; 
% % [~,~,V]=svd(H);
% % F_opt=V(:,1:N_s);

%  F_res=F_opt;
%   At=F_union_opt;
%  F_res=X;
 F_res=F_opt;
    for i=1:N_s
        pha=At'*F_res;
        [~,k]=max(diag(pha*pha'));
        F=[F At(:,k)];
        F_BB=F\F_opt;
         F_res=(F_opt-F*F_BB)/norm(F_opt-F*F_BB,'fro');
%         F_res=(X-F*F_BB)/norm(X-F*F_BB,'fro');
    end
    %F_BB = F_BB/norm(F*F_BB,'fro');
    F_BB = sqrt(Nrf)*F_BB/norm(F*F_BB,'fro');
    error_sparse = norm(F_opt - F*F_BB,'fro')^2;
    norm_opt = norm(F_opt,'fro')^2;

% end

    
    
