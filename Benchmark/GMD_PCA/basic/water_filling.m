function power = water_filling(sigma,total_power,N_RF)
%input sigma: eigenvalues of channel matrix H
%input N-0: covarience of noise
%input E_s: regularization parameter
%input M_t: total power constraints
%output power: resulte of water filling algorithm
[Nr,Nt] = size(sigma);
Nr = N_RF;
power_vector = zeros(1,Nr);
threshold = 0.01; %eigenvalue threshold
sum_1 = 0;
% for jj = 1:Nr
%     if(sigma(jj,jj) >= threshold)
%         sum_1 = sum_1 + 1/(sigma(jj,jj)^2);
%     else
%         break;
%     end
% end
for jj = 1:Nr
     sum_1 = sum_1 + 1/(sigma(jj,jj)^2);
end
flag = 0;
Nr = jj;

while(1)
    mu = total_power/(Nr-flag) + (Nr-flag)^(-1)*sum_1;
    for jj = 1:Nr-flag
        if(sigma(jj,jj) >= threshold)
            power_vector(jj) =  (mu-1/(sigma(jj,jj)^2));
        end
    end
    if(min(power_vector) >= 0)
        break;
    end
    index_n = find(power_vector < 0);
    power_vector(index_n) = 0;
    for kk = 1:length(index_n)
        sum_1 = sum_1 - 1/(sigma(Nr-kk-flag+1,Nr-flag-kk+1)^2);
    end
    flag = flag + length(index_n);
end
power = zeros(Nr,Nr);
for ii = 1:Nr
    power(ii,ii) = power_vector(ii);
end


