clc
clear
close all
%% data input
addpath('funs')
% load('synTask5_nonoise.mat')
%% Generate data
task_num = 25;   %任务数
sample_number = 400;  % 单个任务样本数量
ni = ones(task_num, 1) * sample_number;       % sample size %
m  = length(ni);              % task number 
d  = 200;                     % dimension

XSigma = diag(0.5 * ones(d, 1)) + 0.5 * ones(d, d);  % 
sigma_max = 2;
% generated the model W
r = 5; % rank of W
sigma = 0.05;
L = randn(d,r)*sqrt(sigma);
R = randn(r,task_num)*sqrt(sigma); 
B = L*R;   % actually W
% generated data
X = cell(m, 1);
Y = cell(m, 1);
opts.q = 1;
for tt = 1: m    
    % generate data matrices
    X{tt} = zeros(ni(tt), d);
    for ss = 1: ni(tt)
        X{tt}(ss, :) = mvnrnd(zeros(1, d), XSigma);
    end
    % generate targets.
    X{tt} = standardize(X{tt});
%       sigma_i = 1;                  % d_1
    sigma_i = 2^(- (tt-1)*3/100); % d_2
    % sigma_i = 2^(- (tt-1)*3/25);  % d_3
    % sigma_i = 2^(- (tt-1)/4);     % d_4
    Y{tt} = X{tt} * B(:, tt)+1+randn(ni(tt), 1) * sigma_i * sigma_max;
    Yt{tt} = X{tt} * B(:, tt);  % test data
    fprintf('Task %u sigma: %.4f\n', tt, sigma_i * sigma_max);
end
W = B;

clearvars -except X Y W

% X = Cellvert(X);
%% initialize
itermax = 1; % maximum iterations
TrainRate = 0.5%:0.1:0.9;
W_truth = W;   % truth label
tasknum = length(X);



%% main loop
for rate = 1:length(TrainRate)
   %  Train and test dataset
   trainrate = TrainRate(rate);
   [X_tr, Y_tr, X_te, Y_te] = mtSplitPerc(X, Y, trainrate);

   %% our method KMSV
   % KMSV
   opt.K = 5;
   opt.rho = 100;
   opt.st = 0.2;
   [W_pre, Bt,obj,s] = NTRMTL(Cellvert(X_tr), Cellvert(Y_tr), opt);clear opts
   [MSE,nMSE,nRMSE,~,~,~] = regmeasure(Y_te, X_te, W_pre,1,Bt);  
   nMSEKMSV = MSE
%    MSE_NTR = [MSE_NTR, nMSE]; 
   errorW_KMSV = norm(W_truth-W_pre,'fro')^2/tasknum;   
   % KMSV-new
   opt.K = 5;
   opt.rho = 100;
   opt.st = 10;
   [W_pre, Bt,obj,s_new] = NTRMTL_Ext(Cellvert(X_tr), Cellvert(Y_tr), opt);clear opts
   [MSE,nMSE,nRMSE,~,~,~] = regmeasure(Y_te, X_te, W_pre,1,Bt);
   nMSEKMSVnew = MSE
%    MSE_NTRext = [MSE_NTRext, nMSE]; 
   errorW_KMSVnew = norm(W_truth-W_pre,'fro')^2/tasknum;
end








