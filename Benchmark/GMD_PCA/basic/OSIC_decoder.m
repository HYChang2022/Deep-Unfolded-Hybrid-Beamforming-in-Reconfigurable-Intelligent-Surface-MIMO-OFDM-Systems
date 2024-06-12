function [X_hat]=OSIC_decoder(y,H,sigma2,nT,OSIC_type)
%input y: received signal vector
%input H: channel matrix
%input sigma2: noise variance
%input nT: number of transmit antennas
%input OSIC_type
%       - 1 : Post_detection_SINR
%       - 2 : Post_detection_SNR
% Output X_hat : estimated signal vector
if OSIC_type==1  % Post_detection_SINR
  Order=[];  % detection order
  index_array=[1:nT]; % yet to be detected signal index
  % V-BLAST
  for stage = 1:nT
     Wmmse=inv(H'*H+sigma2*eye(nT+1-stage))*H';  % MMSE filter
     WmmseH=Wmmse*H;
     SINR=[];
     for i =1:nT-(stage-1)
        tmp= norm(WmmseH(i,[1:i-1 i+1:nT-(stage-1)]))^2 ...
              + sigma2*norm(Wmmse(i,:))^2;
        SINR(i)=abs(WmmseH(i,i))^2/tmp; % SINR calculation
     end
     [~,index_temp] = max(SINR);    % ordering using SINR
     Order = [Order index_array(index_temp)]; 
     index_array = index_array([1:index_temp-1 index_temp+1:end]); 
     x_temp(stage) = Wmmse(index_temp,:)*y;     % MMSE filtering
     X_hat(stage) = QAM16_slicer(x_temp(stage),1); % slicing
     y_tilde = y - H(:,index_temp)*X_hat(stage); % interference subtraction
     H_tilde = H(:,[1:index_temp-1 index_temp+1:nT-(stage-1)]); % new H
     H = H_tilde;   y = y_tilde;
  end
  X_hat(Order) = X_hat;
  
else % OSIC with Post_detection_SNR ordering
  Order=[];   
  index_array=[1:nT]; % set of indices of signals to be detected
  % V-BLAST
  for stage=1:nT
     G = inv(H'*H)*H'; 
     norm_array=[];
     for i=1:nT-(stage-1) % detection ordering
        norm_array(i) = norm(G(i,:));
     end
     [~,index_min]=min(norm_array); % ordering in SNR
     Order=[Order index_array(index_min)];
     index_array = index_array([1:index_min-1 index_min+1:end]); 
     x_temp(stage) = G(index_min,:)*y;  % Tx signal estimation
     X_hat(stage) = QAM16_slicer(x_temp(stage),1);  % slicing
     y_tilde = y-H(:,index_min)*X_hat(stage); % interference subtraction
     H_tilde = H(:,[1:index_min-1 index_min+1:nT-(stage-1)]); % new H
     H = H_tilde;   y = y_tilde;
  end
  X_hat(Order) = X_hat;
end
