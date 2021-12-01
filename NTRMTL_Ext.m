% Multi-Task Learning
% a method to solve the multi-task Supprot Vector Machine problem
% \min _W \sum_{t=1}^T f(W_t^TX_t,Y_t,M_t)+\rho_t \sum_{i=1}^k
% \sigma_i^2(W)

% X {d * nt} * t
% Y {1 * nt} * t

function [Wt, Bt,Obj,s1] = NTRMTL_Ext(X, Y, opt)
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

% update F and G

[U, S, V] = svd(W0_prep);
F0 = U(:,1:(dim-k));
G0 = V(:,1:(dim-k));
Obj(1)  = fun(W0_prep, X, Y,Bt,task_num) + rho*(sum(diag(S))-trace(F0'*W0_prep*G0));

Wt = W0_prep;
F = F0;
G = G0;
%% 算法过程
for iter = 1:itermax
    % update W_t b_t
    D = 1/2*(Wt*Wt'+0.00001 * eye(dim))^(-1/2);
    for t=1:task_num
        nt = length(Y{t});
        Wt(:,t)=(XX{t}+rho * D+st * eye(dim))\(X{t}*(Y{t}-Bt(t))'+1/2*rho*F*G(t,:)'); % X{t}*Zt'+1/2*lambda*F*Gt'-Bt{t}*ones(1,Ct(t)) -> X{t}*(Zt'-ones(nt,1)*(Bt{t})')+1/2*lambda*F*Gt'
        Bt(t) =(Y{t} - Wt(:,t)'*X{t})*ones(nt,1) / nt; % Zt*ones(nt,1) -> Zt        
    end
    
    % update FF^T  WW^T need: rank n-k   k , and dimension dim,  
    [U, S, V] = svd(Wt);
    F = U(:,1:(dim-k));
    G = V(:,1:(dim-k));
    s = sort(diag(S),'descend');
    s1 = sort(diag(S),'ascend');
    % convergence condition
    % condition 1
    if sum(diag(S)) - sum(s(1:(dim-k)))>=epsu
        rho = rho*2;
    elseif sum(diag(S)) - sum(s(1:(dim-k-1)))<= epsu
        rho = rho/2;
    else
        break
    end
    
    Obj(iter+1)  = fun(Wt, X, Y,Bt,task_num) + rho*(sum(s)-trace(F'*Wt*G));
    objk(iter+1) = fun(Wt, X, Y,Bt,task_num);
    
    % 条件2
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
