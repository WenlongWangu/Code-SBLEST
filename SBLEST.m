function [W, alpha,V,Cov_mean_train] = SBLEST(X,Y, Maxiters,K, tau, e)
% ************************************************************************
% SBLEST    : Spatio-Temporal-filtering-based single-trial EEG classification
%
% --- Inputs ---
% Y         : Observed label vector
% X         : M EEG signals from train set
%             M cells: Each X{1,i} represents a trial with size of [C*T]
% Maxiters  : Maximum number of iterations (5000 is suggested in this paper)
% e         : Convergence threshold 
%
% --- Output ---
% W         : The estimated low-rank matrix
% alpha     : The classifier parameter
% V         : Each column of V represents a spatio-temporal filter

% Reference:

% Copyright:

% ************************************************************************

[R,Cov_mean_train] = compute_covariance_train(X,K,tau);

% Cov = compute_covariance(X,K,tau);

% [R,Cov_mean_train] = compute_enhanced_cov_matrix(Cov);

%% Check properties of R
[M, D_R] = size(R); % M: # of samples;D_R: dimention of vec(R_m)
KC = round(sqrt(D_R));
epsilon =e;
Loss_old = 0;
if (D_R ~= KC^2)
    disp('ERROR: Columns of A do not align with square matrix');
    return;
end

% Check properties of R: symmetric ?
for c = 1:M
    row_cov = reshape(R(c,:), KC, KC);
    if ( norm(row_cov - row_cov','fro') > 1e-4 )
        disp('ERROR: Measurement row does not form symmetric matrix');
        return
    end
end

%% Initializations

U = zeros(KC,KC); % The estimated low-rank matrix W set to be Zeros

Psi = eye(KC); % the covariance matrix of Gaussian prior distribution is initialized to be Unit diagonal matrix

lambda = 1;% the variance of the additive noise set to 1 by default

%% Optimization loop
for i = 1:Maxiters
   %% Compute estimate of X 
    RPR = zeros(M, M); %  Predefined temporal variables RT*PSI*R
    B = zeros(KC^2, M); %  Predefined temporal variables
    for c = 1:KC
        start = (c-1)*KC + 1; stop = start + KC - 1;
        Temp = Psi*R(:,start:stop)'; 
        B(start:stop,:)= Temp; 
        RPR =  RPR + R(:,start:stop)*Temp;  
    end

    Sigma_y = RPR + lambda*eye(M); 
    u = B*( Sigma_y\Y ); % Maximum a posterior estimation of u
    U = reshape(u, KC, KC);
    U = (U + U')/2; % make sure U is symmetric
       
   %% Update the dual variables of PSI : PHi_i
    Phi = cell(1,KC);
    SR = Sigma_y\R;
    for c = 1:KC
        start = (c-1)*KC + 1; stop = start + KC - 1;
        Phi{1,c} = Psi -Psi * ( R(:,start:stop)'*SR(:,start:stop) ) * Psi;
    end
    
          %% Update covariance parameters Psi: Gx
    PHI = 0;    
    UU = 0;
    for c = 1:KC
        PHI = PHI +  Phi{1,c};
        UU = UU + U(:,c) * U(:,c)';
    end
    Psi = ( (UU + UU')/2 + (PHI + PHI')/2 )/KC;% make sure Psi is symmetric

   %% Update lambda
   theta = 0;
    for c = 1:KC
        start = (c-1)*KC + 1; stop = start + KC - 1;
        theta = theta +trace(Phi{1,c}* R(:,start:stop)'*R(:,start:stop)) ;
    end
    lambda = (sum((Y-R*u).^2) + theta)/M;  

   %% Output display and  convergence judgement
        Loss = Y'*Sigma_y^(-1)*Y + log(det(Sigma_y));        
        delta_loss = norm(Loss - Loss_old,'fro')/norm( Loss_old,'fro');  
        if (delta_loss < epsilon)
            disp('EXIT: Change in Loss below threshold');
            break;
        end
        Loss_old = Loss;
         if (~rem(i,100))
            disp(['Iterations: ', num2str(i),  '  lambda: ', num2str(lambda),'  Loss: ', num2str(Loss), '  Delta_Loss: ', num2str(delta_loss)]);
         end   
end
     %% Eigendecomposition of W
     W = U;
     [~,D,V] = eig(W);% each column of V represents a spatio-temporal filter
     alpha = diag(D); % the classifier parameter
     
end

function [R,Cov_mean_train] = compute_covariance_train(X,K, tau)
M = length(X);
[C,T] = size(X{1,1});
KC = K*C;
Cov = cell(1,M);
Sig_Cov = zeros(KC,KC);
for m = 1:M
    X_m = X{1,m};
    X_m_hat = [];
    for k = 1:K
        n_delay = (k-1)*tau;
        if n_delay ==0
            X_order_k = X_m;
        else
            X_order_k(:,1:n_delay) = 0;
            X_order_k(:,n_delay+1:T) = X_m(:,1:T-n_delay);
        end
        X_m_hat = cat(1,X_m_hat,X_order_k);
    end
    Cov{1,m} = X_m_hat*X_m_hat';
    Cov{1,m}= Cov{1,m}/trace(Cov{1,m}); % trace normalizaiton
    Sig_Cov = Sig_Cov + Cov{1,m};
end

Cov_mean_train =Sig_Cov/M;
Cov_whiten = zeros(M,KC,KC);
for m = 1:M
    temp_cov= Cov_mean_train^(-1/2)*Cov{1,m}*Cov_mean_train^(-1/2);% whiten
     Cov_whiten(m,:,:)  = (temp_cov + temp_cov')/2; 
    R(m,:) =  vec(logm(squeeze(Cov_whiten(m,:,:))));% logm
end 
   R = real(R);
end

% function [Cov] = compute_covariance(X,K,tau)
% 
% M = length(X);
% [~,T] = size(X{1,1});
% Cov = cell(1,M);
% for m = 1:M
%     X_m = X{1,m};
%     X_m_hat = [];
%     for k = 1:K
%         n_delay = (k-1)*tau;
%         if n_delay ==0
%             X_order_k = X_m;
%         else
%             X_order_k(:,1:n_delay) = 0;
%             X_order_k(:,n_delay+1:T) = X_m(:,1:T-n_delay);
%         end
%         X_m_hat = cat(1,X_m_hat,X_order_k);
%     end
%     Cov{1,m} = X_m_hat*X_m_hat';
%     Cov{1,m}= Cov{1,m}/trace(Cov{1,m}); % trace normalizaiton
% end
% end
% 
% function [R,Cov_mean_train] = compute_enhanced_cov_matrix(Cov)
% M = length(Cov);
% KC = size(Cov{1,1},1);
% Cov_mean_train = zeros(KC,KC);
% 
% for m =1:M
%     Cov_mean_train =Cov_mean_train+Cov_mean_train/M;
% end
% 
% Cov_whiten = zeros(M,KC,KC);
% for m = 1:M
%     temp_cov= Cov_mean_train^(-1/2)*Cov{1,m}*Cov_mean_train^(-1/2);% whiten
%      Cov_whiten(m,:,:)  = (temp_cov + temp_cov')/2; 
%     R(m,:) =  vec(logm(squeeze(Cov_whiten(m,:,:))));% logm
% end 
% end
