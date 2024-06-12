function [H_man,v_man] =  test_Mani_0714(R,G,M,Ns,AR_G_irs,gain_G_path,gain_R_path,AT_R_irs,trans_Pt,sigma2,Nk)

    % AR_G_irs = AR_G_irs;
    % gain_G_path = gain_G_path;
    % gain_R_path = gain_R_path;
    % AT_R_irs = AT_R_irs;
    
    snr = trans_Pt/sigma2/Ns;
    %% calculate epsilon _sig & v_div ,v_div can also be randomized for initialization
    gain2_RG =zeros(Ns,1);
    Pi = zeros(M,Ns);
    m = Ns;
    v_div = zeros(M,1); 
    mM = floor(M/m); %% must be an integer
    for i_b = 1:Ns
        alpha_i = gain_G_path(i_b);
        beta_i = gain_R_path(i_b);
        gain2_RG(i_b) = abs(alpha_i*beta_i)^2; % truncated SVD: take the first Ns elements of D(i,i)
        aT_irs = diag(AT_R_irs(:,i_b)');
        ar_irs = AR_G_irs(:,i_b);
        atr_ii = aT_irs *ar_irs ;
        Pi(:,i_b) = atr_ii;
        ang_atr = angle(atr_ii);
        v_div((i_b-1)*mM+1:i_b*mM) = exp(1j*ang_atr((i_b-1)*mM+1:i_b*mM));    
    end

    epsilon_sig = snr*gain2_RG;
    v_man = test_Man_0714(Pi,M,Ns,epsilon_sig,v_div);
    
    %% Build effective channel
    % H_man = zeros(Nr,Nt,Nk);
    for k = 1:Nk
        H_man(:,:,k) = R(:,:,k)*diag(v_man')*G(:,:,k);
    end
end
