function [W,S,Loss,Con] = DLPLSR(X,Y,nl,opt)
% -------------------------------------------------------------------------
%  FAST SFGC  (Self-Representation + Fuzzy Graph Construction)
%  接口:  [W,S,Loss,Con] = SFGC_fast(XL,XU,YL,opt)
%  opt 结构体字段:
%       opt.l1 , opt.l2 , opt.l3 , opt.l4   (对应 λ1…λ4)
% -------------------------------------------------------------------------
warning off

% ----- 1. 数据准备 --------------------------------------------------------
                    % d × n
[d,~] = size(X);   c = size(Y,1);
YL = Y(:,1:nl); XL = X(:,1:nl);
S = (X'*X);  S = (S+S')/2;           % 初始化 S
Ls = diag(sum(S,2)) - S;
W = eye(d,c);                        % 初始化 W

DX  = 2 - 2*(X'*X);                  % 余弦距离平方
XXT = X*X';                          % d × d
XLXL= XL*XL';                        % d × d
B  = XL*YL';                       % d × c

% ----- 2. 超参数 ----------------------------------------------------------
lam1 = opt.l1;  lam2 = opt.l2;  lam3 = opt.l3;  lam4 = opt.l4;

% ----- 3. 迭代参数 --------------------------------------------------------
maxIter = 20;  epsConv = 1e-3;
Loss = zeros(maxIter,5);  Con = zeros(maxIter,1);
for it = 1:maxIter

    % === 4.1 W 更新 (GPI) ================================================
    Dw = 1 ./ sqrt(sum(W.^2,2) + eps);
    J  = XLXL + lam1*X*Ls*X' + lam2*XXT - lam2*X*S*X' + lam4*diag(Dw);
    W  = GPI(J, B, W);                % 内部自带早停

    % === 4.2 S 更新 ======================================================
    E   = 1/(2*lam3)*(lam1/2)*DX - lam2*X'*(W*W')*X;    % n×n
    S   = update_S(E);
    S   = (S+S')/2; 
    Ls  = diag(sum(S,2)) - S;

    % === 4.3 Loss / Convergence =========================================
    Loss(it,1) = norm(W'*XL - YL,'fro').^2;
    Loss(it,2) = lam1 * trace(X*Ls*X');
    Loss(it,3) = lam2 * (trace(W'*XXT*W) - trace(W'*X*S*X'*W)) ...
               + lam3 * norm(S,'fro').^2;
    Loss(it,4) = lam4 * sum( sqrt(sum(W.^2,2)) );
    Loss(it,5) = sum(Loss(it,1:4));

    if it==1, Con(it)=Inf;
    else,     Con(it)=abs(Loss(it,5)-Loss(it-1,5))/Loss(it,5); end
    if Con(it)<epsConv || it==maxIter
        Loss = Loss(1:it,:);  Con = Con(1:it);  break;
    end
end
end

function S = update_S(E)
S = zeros(size(E));
for i = 1:size(E,1)
    ppp = E(i,:);
    ppp(i) = [];
    sss = MY_NSS(ppp);
    sss = VecInsert(sss,i,0);
    S(i,:) = sss;
end
end

function [x] = MY_NSS(v)
%% solve
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=1
%  Transform it into f(lambda_m) = 1/n*sum_{i=1}^n max(lambada_m - u_j,0) -
%  lambda_m = 0; x_j = max(u - lambda_m, 0);
%  if umin > 0, lambda_m = 0; x = u;
%  else : Newton Method.
%%
n = length(v);
u = v-mean(v) + 1/n;
umin = min(u);
if umin >= 0
    x = u;
else
    f = 1;
    iter = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        p = lambda_m - u;
        k  = p>0;
        g = sum(k)/n - 1;
        f = sum(p(k))/n - lambda_m;
        lambda_m = lambda_m - f/g;
        iter = iter + 1;
        if iter>100
            break;
        end
    end
    x = max(-p,0);
end
end

function data=VecInsert(mat,ind,num)
n=length(mat);
data(ind)=num;
data(1:ind-1)=mat(1:ind-1);
data(ind+1:n+1)=mat(ind:n);
end

function [W]=  GPI(J, M, W0)
tol = 1e-5; Itermax = 100; iter = 0; err = Inf;
J = (J+J')/2;


 % J_hat = eps* eye(size(J,1)) - J;
J_hat  = J;
while err>tol && iter < Itermax
    P = 2*J_hat*W0 + 2*M;
    [U,~,V] = svd(P,'econ');
    W = U*V';
    err = max(abs(W(:)-W0(:)));
    W0 = W;
    iter = iter + 1;
end
end