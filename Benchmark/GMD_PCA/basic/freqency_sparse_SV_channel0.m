function [H_f,A_t,A_r] = freqency_sparse_SV_channel0(K,N_cl,N_ray,sigma_2_alpha,sigma_ang,N_t,N_r,los)
% This is a programme creating the channel for Spatially Sparse Precoding
% in mmWave MIMO System. The channel is based on extended Saleh-Valenzuela
% model.

% Inputs£∫
%   K: number of carrier
%   fc: ‘ÿ≤®∆µ¬ 
%   N_cl£∫multipath: number of cluster
%   N_ray: multipath: number of rays per cluster
%   sigma_2_alpha: cluster power
%   sigma_ang: angle spread, sigma^t_phi=sigma^r_phi=sigma^t_theta=sigma^r_theta
%   N_t£∫number of transimtter antennas
%   N_r£∫number of receiver antennas

% Outputs£∫
%   H£∫–≈µ¿æÿ’Û
%   A_t£∫steering vectors of transmitter
%   A_r£∫steering vectors of receiver
%   cl_indx: multipath cluster index in A_t and A_r
% set assistant parameters
% lambda = 3e8/fs;            % length of carrier wave
% d = lambda/2;               % interval of antennas
% k0 = 2*pi/lambda;           % just mean it ®r£®®s£ﬂ®t£©®q
ang_t_phi = 2*pi;%60/180*pi;      % azimuth angle range of transmitter
ang_t_theta = pi/2;%20/180*pi;	% elevation angle range of receiver
ang_r_phi = 2*pi;               % receivers are omni-directional
ang_r_theta = pi/2;               % receivers are omni-directional
D = 64;%128;0                    % length of cyclic prefix
% Ts = 1/(2.538e9);                % symbol period
Tau = D;               % maxinum time delay
sigma_ang = sigma_ang/180*pi;   % from digree to radian

% creating steering vector
check = length(N_t)*length(N_r);
switch check
    case 1
        n_t = (0:(N_t-1))';     % transmitter
        phi1 = ((ang_t_phi-sigma_ang)*(rand(N_cl,1)-0.5)+0.5*sigma_ang*ones(N_cl,1))*ones(1,N_ray)+...
            laplacernd(0,sigma_ang,[N_cl,N_ray]); % azimuth
        phi1 = reshape(phi1',[N_cl*N_ray,1]);
        A_t = exp(-1i*pi*n_t*sin(phi1'))/sqrt(N_t);
        n_r = (0:(N_r-1))';     % receiver
        phi2 = ((ang_r_phi-sigma_ang)*(rand(N_cl,1)-0.5)+0.5*sigma_ang*ones(N_cl,1))*ones(1,N_ray)+...
            laplacernd(0,sigma_ang,[N_cl,N_ray]); % azimuth
        phi2 = reshape(phi2',[N_cl*N_ray,1]);
        A_r = exp(-1i*pi*n_r*sin(phi2'))/sqrt(N_r);
        N_T = N_t;
        N_R = N_r;
    case 4
        n_t = (0:N_t(1,1)-1)';     % transmitter
        m_t = (0:N_t(1,2)-1)';
        N_T = N_t(1,1)*N_t(1,2);
        N_R = N_r(1,1)*N_r(1,2);
        pic_phi1 = ((ang_t_phi-sigma_ang)*rand(N_cl,1)+0.5*sigma_ang*ones(N_cl,1))*ones(1,N_ray)+...
            laplacernd(0,sigma_ang,[N_cl,N_ray]); % azimuth
        phi1 = reshape(pic_phi1',[N_cl*N_ray,1]);
        pic_theta1 = ((ang_t_theta-sigma_ang)*rand(N_cl,1)+0.5*sigma_ang*ones(N_cl,1))*ones(1,N_ray)+...
            laplacernd(0,sigma_ang,[N_cl,N_ray]); % elevation
        theta1 = reshape(pic_theta1',[N_cl*N_ray,1]);
        A_t = zeros(N_T,N_cl*N_ray);
        for path = 1:N_cl*N_ray
            e_a1 = exp(-1i*pi*sin(phi1(path,1))*cos(theta1(path,1))*n_t)/sqrt(N_t(1,1));
            e_e1 = exp(-1i*pi*sin(theta1(path,1))*m_t)/sqrt(N_t(1,2));
            A_t(:,path) = kron(e_a1,e_e1);
        end
        n_r = (0:(N_r(1,1)-1))';     % receiver
        m_r = (0:(N_r(1,2)-1))';
        pic_phi2 = ((ang_r_phi-sigma_ang)*rand(N_cl,1)+0.5*sigma_ang*ones(N_cl,1))*ones(1,N_ray)+...
            laplacernd(0,sigma_ang,[N_cl,N_ray]); % azimuth
        phi2 = reshape(pic_phi2',[N_cl*N_ray,1]);
        pic_theta2 = ((ang_r_theta-sigma_ang)*rand(N_cl,1)+0.5*sigma_ang*ones(N_cl,1))*ones(1,N_ray)+...
            laplacernd(0,sigma_ang,[N_cl,N_ray]); % elevation
        theta2 = reshape(pic_theta2',[N_cl*N_ray,1]);
        A_r = zeros(N_R,N_cl*N_ray);
        for path = 1:N_cl*N_ray
            e_a2 = exp(-1i*pi*sin(phi2(path,1))*cos(theta2(path,1))*n_r)/sqrt(N_r(1,1));
            e_e2 = exp(-1i*pi*sin(theta2(path,1))*m_r)/sqrt(N_r(1,2));
            A_r(:,path) = kron(e_a2,e_e2);
        end
    otherwise
        disp('Error: invalid anttena arguments in function freqency_sparse_SV_channel!');return;
end
tau = Tau*rand(1,N_cl*N_ray);%tau = kron(Tau*rand(1,N_cl),ones(1,N_ray));
%choose los_vs_nlos
if los==1
    L= (N_cl-1)*N_ray+1;
    alpha_los=sqrt(sigma_2_alpha/2/L)*(randn(1,1)+1i*randn(1,1));
    alpha_nlos=sqrt(sigma_2_alpha*0.005/2/L)*(randn(1,(N_cl-1)*N_ray)+1i*randn(1,(N_cl-1)*N_ray));
    alpha=[alpha_los zeros(1,N_ray-1) alpha_nlos];
else
    L=N_cl*N_ray;
    alpha = sqrt(sigma_2_alpha*0.005/2/L)*(randn(1,N_cl*N_ray)+1i*randn(1,N_cl*N_ray));    % create alpha
    
end

H_f = zeros(N_R,N_T,K);
for k = 1:K
    P = alpha.*exp(-1i*2*pi*tau*k/K);
    H_f(:,:,k) = A_r*diag(P)*A_t';
%     H_f(:,:,k) =  H_f(:,:,k)/norm( H_f(:,:,k),'fro');
end
 H_f = H_f*sqrt(N_T*N_R);
%   for kk=1:K
%       H_f(:,:,kk) = 10*H_f(:,:,kk)/norm(H_f(:,:,kk),'fro');
%  end
% switch nargin
%     case 8
%     case 9
%         global pic_phi1 pic_theta1;
%         figure(1)
%         plot(pic_phi1',pic_theta1','.') 
%         hold on
%     otherwise
%         disp('Error: not enough inputs in function freqency_sparse_SV_channel!');return;
% end