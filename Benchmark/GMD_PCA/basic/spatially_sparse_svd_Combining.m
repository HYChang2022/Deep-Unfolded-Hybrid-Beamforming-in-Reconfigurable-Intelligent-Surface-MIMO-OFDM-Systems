function [W,W_BB]=spatially_sparse_svd_Combining(Nrf,N_s,H,Ar,Y,carrier, W_union_opt,sigma2)

     
%%%%%%%%%%%%%%%%%%%%broad band%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     
%          W_union_opt=[];
%             for subcarrier=1:Nc
%                 [U,~,~] = svd(H(:,:,subcarrier));
%                 W_union_opt =[W_union_opt  U(:,1:Nrf)];
%             end
          
  %%共同F_rf
%               [V_opt,~,~]=svd(W_union_opt);           
%              W2=sqrt(1/Nr)*exp(1i*angle(V_opt(:,1:Nrf)));
%              W2=sqrt(1/Nr)*exp(1i*angle(U(:,1:Nrf)));
%%%各个子载波独立F_RF
              [V_opt,~,~]=  svd(H(:,:,carrier));
              W_gmd=V_opt(:,1:N_s);

%%%单载波
%              W_gmd=U(:,1:N_s);
%%%%%%%%%%%%%%%%%%%%%%%%end%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            for i=1:Nc
%                [U_digital,~,~]=svd(H(:,:,i));
%                UH=U_digital';
%                W_gmd=UH(:,1:N_s);
%                W_BB=inv(W'*W)*W'*W_gmd; 
%                W_BB=sqrt(N_s)*W_BB/norm(W*W_BB,'fro');
%            end

% [U,S,V]=svd(H);
% % UH = U';
% % W_gmd = U(:,1:N_s);
W=[];
% W_res = W_gmd;
% W_res=Y;
 W_res=W_gmd;
% Ar = W_union_opt;
ArH = Ar';
for i=1:N_s
    pha=ArH*W_res;
    [~,k]=max(diag(pha*pha'));
    W=[W Ar(:,k)];
    W_BB=W\W_gmd;
       W_res=(W_gmd-W*W_BB)/norm(W_gmd-W*W_BB,'fro');
%       W_res=(Y-W*W_BB)/norm(Y-W*W_BB,'fro');
end
W_BB=sqrt(Nrf)*W_BB/norm(W*W_BB,'fro');

% end

    
