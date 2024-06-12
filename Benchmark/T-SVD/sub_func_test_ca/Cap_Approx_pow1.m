%% another acapacity approx
% 2020 0410 
% input: 
%        H    
%        N_t_RF
%        N_r_RF
%        Ns
%        mode 1---> normal 
%        mode 2---> waterfilling
% output: F_BB N_RF*Ns
%         F_RF Nt *N_RF
%         W_BB
%         W_RF

% function [Cap_opt] = Cap_Approx_pow1(H,Ns,Nt,Nr,snr,mode,Nk)
% 
%     for k = 1:Nk
%         H_k = H(:,:,k); 
%         [U,S,V] = svd(H_k);
%         if mode == 1
%             F_opt_k = V(:,1:Ns);
%             W_opt_k = U(:,1:Ns);
%         else
%             snr_s2 = snr*(diag(S)).^2;
%             Pn = 1./snr_s2(1:Ns).';
%             Pt = Ns;
%             P = waterfill(Pt, Pn);
%             F_opt_k = V(:,1:Ns)*diag(sqrt(P));
%             W_opt_k = U(:,1:Ns);
%         end
%         F_opt(:,:,k) = F_opt_k;
%         W_opt(:,:,k) = W_opt_k; 
%     end
% 
% 
%     %% Hybrid precoding and combining design in MIMO-OFDM system
%     options_pro.func_tolerance = 1e-2;
%     
%     % Fully-connected architecture
%     FRF_enc = exp( 1i*unifrnd(0,2*pi, Nt, Nt_RF) );
%     [F_RF_fc, F_BB_fc, ~ ] = fast_hybrid_precoding_OFDM_wrapper_FC(F_opt, Nt_RF, FRF_enc);
%     for k = 1:Nk 
%         F_BB_fc(:,:,k) = sqrt(Ns) * F_BB_fc(:,:,k) ./ norm(F_RF_fc * F_BB_fc(:,:,k), 'fro');
%         F_app_fc(:,:,k) = F_RF_fc*F_BB_fc(:,:,k);
%     end
%         
%     WRF_enc = exp( 1i*unifrnd(0,2*pi, Nr, Nr_RF) );  
%     [W_RF_fc, W_BB_fc, ~] = fast_hybrid_precoding_OFDM_wrapper_FC( W_opt, Nr_RF, WRF_enc);
%     for k = 1:Nk 
%         W_app_fc(:,:,k) = W_RF_fc*W_BB_fc(:,:,k);
%     end
% 
%     % Partially-connected architecture
%     F_mask = zeros(Nt,Nt_RF);
%     W_mask = zeros(Nr,Nr_RF);
%     Mt = round(Nt/Nt_RF);
%     Mr = round(Nr/Nr_RF);
%     for t = 0:Nt_RF-1
%         F_mask(t*Mt+1:(t+1)*Mt,t+1)=ones(Mt,1);
%     end
%     for r = 0:Nr_RF-1
%         W_mask(r*Mr+1:(r+1)*Mr,r+1)=ones(Mr,1);
%     end
%     FRF_enc = exp( 1i*unifrnd(0,2*pi, Nt, Nt_RF) );
%     [F_RF_pc, F_BB_pc, ~ ] = fast_hybrid_precoding_OFDM_wrapper_PC(F_opt, Nt_RF, FRF_enc, F_mask);
%     for k = 1:Nk 
%         F_BB_pc(:,:,k) = sqrt(Ns) * F_BB_pc(:,:,k) ./ norm(F_RF_pc * F_BB_pc(:,:,k), 'fro');
%         F_app_pc(:,:,k) = F_RF_pc*F_BB_pc(:,:,k);
%     end
%         
%     WRF_enc = exp( 1i*unifrnd(0,2*pi, Nr, Nr_RF) );  
%     [W_RF_pc, W_BB_pc, ~] = fast_hybrid_precoding_OFDM_wrapper_PC( W_opt, Nr_RF, WRF_enc, W_mask);
%     for k = 1:Nk 
%         W_app_pc(:,:,k) = W_RF_pc*W_BB_pc(:,:,k);
%     end
% 
%     %% Caulate Capacity for fully-digital and hybird beamforming 
%     % Cap_opt = log2(det( eye(Ns) + (snr/Ns)* pinv(W_opt)*H*F_opt*F_opt'*H'*W_opt));
% 
%     Cap_opt_fc = 0;

%     for k = 1:Nk
%         Cap_opt = Cap_opt_fc + log2(det( eye(Ns) + (snr)* pinv(W_opt(:,:,k))*H(:,:,k)*F_opt(:,:,k)*F_opt(:,:,k)'*H(:,:,k)'*W_opt(:,:,k)));
%     end
% end

%% test_time T-SVD(FC)

function [ Cap_hyb_fc] = Cap_Approx_pow1(H,Nt_RF,Nr_RF,Ns,Nt,Nr,snr,mode,Nk)

    for k = 1:Nk
        H_k = H(:,:,k); 
        [U,S,V] = svd(H_k);
        if mode == 1
            F_opt_k = V(:,1:Ns);
            W_opt_k = U(:,1:Ns);
        else
            snr_s2 = snr*(diag(S)).^2;
            Pn = 1./snr_s2(1:Ns).';
            Pt = Ns;
            P = waterfill(Pt, Pn);
            F_opt_k = V(:,1:Ns)*diag(sqrt(P));
            W_opt_k = U(:,1:Ns);
        end
        F_opt(:,:,k) = F_opt_k;
        W_opt(:,:,k) = W_opt_k; 
    end


    % Hybrid precoding and combining design in MIMO-OFDM system
    options_pro.func_tolerance = 1e-2;
    
    % Fully-connected architecture
    FRF_enc = exp( 1i*unifrnd(0,2*pi, Nt, Nt_RF) );
    [F_RF_fc, F_BB_fc, ~ ] = fast_hybrid_precoding_OFDM_wrapper_FC(F_opt, Nt_RF, FRF_enc);
    for k = 1:Nk 
        F_BB_fc(:,:,k) = sqrt(Ns) * F_BB_fc(:,:,k) ./ norm(F_RF_fc * F_BB_fc(:,:,k), 'fro');
        F_app_fc(:,:,k) = F_RF_fc*F_BB_fc(:,:,k);
    end
        
    WRF_enc = exp( 1i*unifrnd(0,2*pi, Nr, Nr_RF) );  
    [W_RF_fc, W_BB_fc, ~] = fast_hybrid_precoding_OFDM_wrapper_FC( W_opt, Nr_RF, WRF_enc);
    for k = 1:Nk 
        W_app_fc(:,:,k) = W_RF_fc*W_BB_fc(:,:,k);
    end
   
    % Caulate Capacity for fully-digital and hybird beamforming 
 
   
    Cap_hyb_fc = 0;

    for k = 1:Nk       
        Cap_hyb_fc = Cap_hyb_fc + log2(det( eye(Ns) + (snr)* pinv(W_app_fc(:,:,k))*H(:,:,k)*F_app_fc(:,:,k)*F_app_fc(:,:,k)'*H(:,:,k)'*W_app_fc(:,:,k))); % 0308 snr = rho/sigma^2, tongyi le
    end
end



