% Multi-Task Learning
% a method to solve the multi-task Supprot Vector Machine problem
% \min _W \sum_{t=1}^T f(W_t^TX_t,Y_t,M_t)+\rho_t \sum_{i=1}^k
% \sigma_i^2(W)

% X_train {nt * d} * t
% Y_train {nt * ct} * t

function [Wt, Bt,Obj,s] = NTRMTL(X, Y, opt)
% X = multi_transpose(X);  
%% 初始化 W  b_t=0;;   NO  M_t=0
task_num = length(X); % 任务个数
dim = size(X{1}, 1);
itermax = 1000;
rho = opt.rho;
st = opt.st;
epsu = 1e-6;
if dim>task_num
k = opt.K + dim - task_num; % 实际 k
else
    k = opt.K;
end
% precomputation
XY = cell(task_num, 1);
XX = cell(task_num, 1);
W0_prep = [];  
Bt = zeros(task_num, 1);
% precomputation
for t_idx = 1: task_num
    XX{t_idx} = X{t_idx}*X{t_idx}';
    XY{t_idx} = X{t_idx}*Y{t_idx}';
    W0_prep = cat(2, W0_prep, XY{t_idx}); % cat(2,A,B) is the same as [A,B]. cat(1,A,B) is the same as [A;B].
%     Bt{t_idx} = zeros(1,size(Y{t_idx}, 2));
end
[U, ~, ~] = svd(W0_prep);
U3 = U(:, 1:(dim - k));
FF0 = eye(dim) - U3  *U3';
Obj(1)  = fun(W0_prep, X, Y,Bt,task_num) + rho*trace(FF0*W0_prep*W0_prep');
objk(1) = fun(W0_prep, X, Y,Bt,task_num);
Wt = W0_prep;
FF = FF0;

%% 算法过程
for iter = 1:itermax
    % update W_t b_t
    for t=1:task_num
        nt = length(Y{t});
        Wt(:,t) = ((XX{t} + rho * FF + st * eye(dim))\(X{t}) * (Y{t} - Bt(t))'); % 加上对角极小值防止奇异矩阵
%        Wt(:,t) = ((X{t}) * X{t}' + rho * FF + 1e-9 * eye(dim))\(X{t}) * (Y{t} - Bt(t)*ones(1,nt))';
        Bt(t) = (Y{t} - Wt(:,t)'*X{t})*ones(nt,1) / nt;
    end
    % update FF^T  WW^T need: rank n-k   k , and dimension dim,    
    [U, S, ~] = svd(Wt);
    U3 = U(:, 1:(dim - k));
    FF = eye(dim) - U3  *U3';
    s = sort(diag(S),'ascend');
    
    % convergence condition
    % condition 1
    if sum(s(1:opt.K))>=epsu
        rho = rho*2;
    elseif sum(s(1:opt.K+1))<= epsu
        rho = rho/2;
    else
        break
    end
    
    Obj(iter+1)  = fun(Wt, X, Y,Bt,task_num) + rho*trace(FF*Wt*Wt');
    objk(iter+1) = fun(Wt, X, Y,Bt,task_num);
    %% 条件2

%     if iter > itermax || abs(Obj(iter + 1)-Obj(iter)) < epsu
%         break;
%     end
%     disp(Obj(iter + 1)-Obj(iter)) % 函数非凸？F范数
end

end

function obj = fun(Wt, X, Y,Bt,task_num)
obj = 0;
for t = 1:task_num
    w = Wt(:,t);
    final = w'* X{t} + (Bt(t) - Y{t});
    obj = obj + norm(final, 'fro');
end
end
