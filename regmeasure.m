function [MSE,nMSE,nRMSE,RMSE,MAE,R2score] = regmeasure(Y,X,W,me,B)
tasknum = length(Y);
if nargin <5
    B = zeros(tasknum,1);
end
for t = 1:tasknum
    %% prediction label
    yre = X{t}*W(:,t)+B(t);
    %% MSE
    MSE(t) = norm(yre-Y{t})^2/length(Y{t});
    %% nMSE
    nMSE(t) = MSE(t)/var(Y{t});
    %% RMSE
    RMSE(t) = sqrt(MSE(t));
    %% nRMSE
    nRMSE(t) = RMSE(t)/var(Y{t});
    %% MAE
    MAE(t) = sum(abs(yre-Y{t}))/length(Y{t});
    %% R2score
    R2score(t) = 1-norm(yre-Y{t})^2/norm(mean(Y{t})-Y{t})^2;
end
%% get the mean value
if me == 1
    MSE = mean(MSE);
    nMSE = mean(nMSE);
    nRMSE = mean(nRMSE);
    RMSE = mean(RMSE);
    MAE = mean(MAE);
    R2score = mean(R2score);
end
end