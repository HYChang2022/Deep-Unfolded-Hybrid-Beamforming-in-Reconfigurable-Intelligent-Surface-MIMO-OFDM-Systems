function R = SE_BER(choose, H, W_bb, W_rf, F_bb, F_rf, Pt, snr, S, N_subc, Symbol_nomalized, msg, Scale, M)
% 一个关于频谱效率和误比特率的通用函数(mao, 2018/08/19)
% 标准公式： y = W_bb' * W_rf' * H * F_rf * F_bb * x + W_bb' * W_rf' * n
% F_a is the normalizing fator
% SE: choose > 0; BER: choose < 0
%--------------------------------------------------------------------------
% choose:
% 1: full-digital multi-carrier and multi-user with multi-stream for each user
% -2/2: sigle-user and multi-stream
% 3/4: multi-carrier and multi-user with multi-stream for each user (Heath)
% -2 stand for GMD with SIC
R = 0;
Ns = size(W_rf, 2);
switch choose
    case 1
        [~, ~, Nc, K] = size(H);
        for k = 1 : K
            for j = 1 : Ns
                for i = 1 : Nc
                    W_bb_k = W_bb(:, j, i, k);
                    F_bb_k = F_bb(:, j, i, k);
                    H_k = H(:, :, i, k);
                    signal = (Pt / Ns / K) .* W_bb_k' * H_k * F_bb_k * ...
                        F_bb_k' * H_k' * W_bb_k;
                    noise = (Pt / snr) * W_bb_k' * W_bb_k;
                                        
                    interference = 0;
                    % data stream interference
                    for k_j = 1 : Ns
                        if k_j ~= j
                            F_bb_ki = F_bb(:, k_j, i, k);
                            interference = interference + W_bb_k' * ...
                                H_k * F_bb_ki * F_bb_ki' * H_k' * W_bb_k;
                        end
                    end
                    
                    % user interference
                    for k_i = 1 : K
                        if k_i ~= k
                            for k_j = 1 : Ns
                                F_bb_ki = F_bb(:, k_j, i, k_i);
                                interference = interference + W_bb_k' * ...
                                    H_k * F_bb_ki * F_bb_ki' * H_k' * W_bb_k;
                            end
                        end
                    end
                    interference = (Pt / K / Ns) * interference;
                    R = R + log2(1 + abs(signal / (noise + interference)));
                end
            end
        end
        R = R / Nc;
%-------------------------------------------------------------------------- 
    case 2
        Nc = size(H, 3);
        for i = 1 : Nc
            W_bb_k = W_bb(:, :, i);
            F_bb_k = F_bb(:, :, i);
            H_k = H(:, :, i);
            signal = (Pt / Ns) * W_bb_k' * W_rf' * H_k * F_rf * F_bb_k * ...
                F_bb_k' * F_rf' * H_k' * W_rf * W_bb_k;
            noise = (Pt / snr) * W_bb_k' * W_rf' * W_rf * W_bb_k;
            interference = 0; noise = noise + interference; % 无干扰
            R = R + log2(abs(det(eye(Ns) + signal * pinv(noise))));
        end
        R = R / Nc;

%--------------------------------------------------------------------------       
    case 3
        [~, Nt_total, Nc, K] = size(H);
        Mt = K * Ns;
        F_rf2 = reshape(F_rf, Nt_total, Mt);
        for k = 1 : K
            for i = 1 : Nc
                W_bb_k = W_bb(:, :, i, k);
                F_bb_k = F_bb(:, :, i, k);
                W_rf_k = W_rf(:, :, k);
                H_k = H(:, :, i, k);
                signal = (Pt / Ns / K) .* W_bb_k' * W_rf_k' * H_k * F_rf2 * ...
                    F_bb_k * F_bb_k' * F_rf2' * H_k' * W_rf_k * W_bb_k;
                noise = (Pt / snr) * W_bb_k' * W_rf_k' * W_rf_k * W_bb_k;
                interference = 0;
                for k_i = 1 : K
                    if k_i ~= k
                        F_bb_ki = F_bb(:, :, i, k_i);
                        interference = interference + W_bb_k' * W_rf_k' * ...
                            H_k * F_rf2 * F_bb_ki * F_bb_ki' * F_rf2' * H_k' * W_rf_k * W_bb_k;
                    end
                end
                interference = (Pt / K / Ns) * interference;
                noise = noise + interference;
                R = R + log2(abs(det(eye(Ns) + signal * pinv(noise))));
            end
        end
        R = R / Nc;
%--------------------------------------------------------------------------
    case 4
        [~, Nt_total, Nc, K] = size(H);
        Mt = K * Ns;
        F_rf2 = reshape(F_rf, Nt_total, Mt);
        for k = 1 : K
            for j = 1 : Ns
                for i = 1 : Nc
                    W_bb_k = W_bb(:, j, i, k);
                    F_bb_k = F_bb(:, j, i, k);
                    W_rf_k = W_rf(:, :, k);
                    H_k = H(:, :, i, k);
                    signal = (Pt / Ns / K) .* W_bb_k' * W_rf_k' * H_k * F_rf2 * ...
                        F_bb_k * F_bb_k' * F_rf2' * H_k' * W_rf_k * W_bb_k;
                    noise = (Pt / snr) * W_bb_k' * W_rf_k' * W_rf_k * W_bb_k;
                    
                    
                    interference = 0;
                    % data stream interference
                    for k_j = 1 : Ns
                        if k_j ~= j
                            F_bb_ki = F_bb(:, k_j, i, k);
                            interference = interference + W_bb_k' * W_rf_k' * ...
                                H_k * F_rf2 * F_bb_ki * F_bb_ki' * F_rf2' * H_k' * W_rf_k * W_bb_k;
                        end
                    end
                    
                    % user interference
                    for k_i = 1 : K
                        if k_i ~= k
                            for k_j = 1 : Ns
                                F_bb_ki = F_bb(:, k_j, i, k_i);
                                interference = interference + W_bb_k' * W_rf_k' * ...
                                    H_k * F_rf2 * F_bb_ki * F_bb_ki' * F_rf2' * H_k' * W_rf_k * W_bb_k;
                            end
                        end
                    end
                    interference = (Pt / K / Ns) * interference;
                    R = R + log2(1 + abs(signal / (noise + interference)));
                end
            end
        end
        R = R / Nc;
%--------------------------------------------------------------------------        
%%  for GMD 
    case -2
        [~, Nt_total, Nc, K] = size(H);
        Mt = K * Ns;
        F_rf2 = reshape(F_rf, Nt_total, Mt);
        Symbol_nomalized_hat = zeros(K * Ns, N_subc, Nc); % 接收机接收到的符号
        for k = 1 : K
            for i = 1 : Nc
                M2=S;
                F_bb2 = reshape(F_bb(:, :, i, :), Mt, K * Ns);%三维数据转为二维数据
                
                % 信道作用
                Y = H(:, :, i, k) * F_rf2 * F_bb2 * Symbol_nomalized(:, :, i);
                noise = (randn(size(Y)) + 1i * randn(size(Y))) / sqrt(2);
                noise = sqrt(Pt / snr) * noise;
                Symbol_nomalized_hat(((k - 1) * Ns + 1) : (k * Ns), :, i) =  W_bb(:, :, i, k)' * W_rf(:, :, k)' * (Y + noise);               
                symbol_sliced_hybrid_SOMP = VBLAST_decoder(Symbol_nomalized_hat,Ns,M2);
            end
        end
        
        % 误码验证
        Symbol_hat = reshape(Symbol_nomalized_hat / Scale, Nc * N_subc * Ns * K,1);
        msg_hat =qamdemod(Symbol_hat,M,'gray');
        R =  sum(sum(de2bi(msg_hat) ~= msg));   
        
%--------------------------------------------------------------------------
%% 
    case -3
        [~, Nt_total, Nc, K] = size(H);
        Mt = K * Ns;
        F_rf2 = reshape(F_rf, Nt_total, Mt);
        Symbol_nomalized_hat = zeros(K * Ns, N_subc, Nc); % 接收机接收到的符号
        for k = 1 : K
            for i = 1 : Nc
%                 S = (zeros(1, K * Ns) + 1) * F_a(i); % F_a is the normalizing fator
                S_k = (diag(S(i, ((k - 1) * Ns + 1) : (k * Ns)))) ^ (- 1);
                F_bb2 = reshape(F_bb(:, :, i, :), Mt, K * Ns);%三维数据转为二维数据
                
                % 信道作用
                Y = H(:, :, i, k) * F_rf2 * F_bb2 * Symbol_nomalized(:, :, i);
                noise = (randn(size(Y)) + 1i * randn(size(Y))) / sqrt(2);
                noise = sqrt(Pt / snr) * noise;
                Symbol_nomalized_hat(((k - 1) * Ns + 1) : (k * Ns), :, i) = S_k * W_bb(:, :, i, k)' * W_rf(:, :, k)' * (Y + noise);
            end
        end
        
        % 误码验证
        Symbol_hat = reshape(Symbol_nomalized_hat / Scale, Nc * N_subc * Ns * K,1);
        msg_hat =qamdemod(Symbol_hat,M,'gray');
        R =  sum(sum(de2bi(msg_hat) ~= msg));
     otherwise
        error('error for input parameters of choose');
end
