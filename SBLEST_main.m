clc; clear; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Subject,Order,Coefficient
subject_all = [ 'al';'aa';'av';'aw';'ay'];
numSub = length(subject_all);
Accuracy = zeros(numSub, 1); 
Maxiters = 5000; 
e = 2e-6;
load('Tau_selected.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 两层循环
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 循环1：subject确定
for num_S = 1:numSub
    subject = subject_all(num_S,:);
    load(['data_SBLEST/' subject '.mat']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tau=tau_cv(num_S);
  tau =0;
    if tau==0
        K=1;
    else
        K=2;
    end
    disp(['Subject : ' ,subject, '      Order: ', num2str(K),  '      Time delay: ', num2str(tau)]);
    disp('Running SBLEST : update W, bias, psi and lambda');
    [W, alpha,V,Cov_mean_train] = SBLEST(X_train, Y_train, Maxiters,K,tau,e);
    [R_test] = compute_covariance_test(X_test,K,tau,Cov_mean_train);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% step（5）对测试集做预测，并计算准确率  
    predict_Y = R_test*vec(W);
    [accuracy] = compute_acc (predict_Y, Y_test);
    Accuracy(num_S) = accuracy;
    disp(['Subject : ' ,subject, '      Accuracy: ', num2str(accuracy)]);
end % Subjects


function [R_test] = compute_covariance_test(X,K,tau,Cov_mean_train)
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

Cov_whiten = zeros(M,KC,KC);
for m = 1:M
    Cov_whiten(m,:,:) = Cov_mean_train^(-1/2)*Cov{1,m}*Cov_mean_train^(-1/2);% whiten
    R_test(m,:) =  vec(logm(squeeze(Cov_whiten(m,:,:))));% logm
end 
end


function [accuracy] = compute_acc (predict_Y, Y_test)
    Y_predict = zeros(length(predict_Y),1);
    for i = 1:length(predict_Y)
        if (predict_Y(i) > 0)
            Y_predict(i) = 1;
        else
            Y_predict(i) = -1;
        end
    end      
    error_num = 0; 
    total_num = length(predict_Y);
    for i = 1:total_num
        if (Y_predict(i) ~= Y_test(i))
            error_num = error_num + 1;
        end
    end
    accuracy = (total_num-error_num)/total_num;
end
