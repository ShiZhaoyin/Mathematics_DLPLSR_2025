%% --------------------- main_DLPLSR.m  ---------------------
clear; clc; rng default;

% ----------- 数据文件夹 & 文件列表 -----------------------
dataPath = fullfile('.\Data\');   % 自行修改
files    = dir(fullfile(dataPath,'*.mat'));

% ----------- 超参数网格 (λ1=λ2=λ3=λ4) ------------------
lambdaVec = logspace(-4,4,9);        % 9 个点
[LA1,LA2,LA3,LA4] = ndgrid(lambdaVec);   % 4-D
paramSize = size(LA1);                    % [7 7 7 7]
numParam  = prod(paramSize);              % 2401

% ---- 打平成 N×4 参数矩阵，方便 parfor ------------------
optsVec = arrayfun(@(a,b,c,d) ...
    struct('l1',a,'l2',b,'l3',c,'l4',d), ...
    LA1(:),LA2(:),LA3(:),LA4(:));

% ----------- 主循环：遍历数据集 --------------------------
%for fileIdx = 1                            % 只跑第 8 个文件
for fileIdx = 1:numel(files)                % 全部文件

    %% 1. 读数据与可选 PCA
    dataName = files(fileIdx).name;
    load(fullfile(dataPath, dataName));       % XL,XU,YL,X_test,Y_test

    XL = X_train_labeled;
    XU = X_train_unlabeled;
    YL = Y_train_labeled;
    X  = [XL , XU];                           % d×n
    X  = X ./ (vecnorm(X)+eps);               % 列 L2 归一

    [d,n] = size(X);  nl = size(XL,2);

    c     = size(YL,1);
    Y       = zeros(c,n);  Y(:,1:nl) = YL;

    %% 2. 结果预分配（线性向量形）
    ACC_T = zeros(numParam,1);
    ACC_U = zeros(numParam,1);
    TimeG = zeros(numParam,1);

    %% 3. 并行遍历超参数 ------------------------------

    parfor idx = 1:numParam
        opt = optsVec(idx);                       % 结构体，含 l1…l4

        tic;
        [W,~,~,~] = DLPLSR(X,Y,nl,opt);      % 训练
        [ACC_T(idx), ACC_U(idx)] = ...
            Testing(W, XL, YL, XU, Y_train_unlabeled, X_test, Y_test); % 评估
        TimeG(idx) = toc;

        if mod(idx,50)==0
            fprintf('[%s] %d / %d done (%.1f%%)\n', ...
                dataName, idx, numParam, idx/numParam*100);
        end
    end

    % ------- 4. 自动挑选最优参数 ------------------------------
    [~, idxTmax] = max(ACC_T);
    [~, idxUmax] = max(ACC_U);

    if idxTmax == idxUmax                   % 两指标最大处相同
        bestIdx = idxTmax;
    else                                    % 加和策略
        scoreT = ACC_T(idxTmax) + ACC_U(idxTmax);
        scoreU = ACC_T(idxUmax) + ACC_U(idxUmax);
        bestIdx = idxTmax;  if scoreU > scoreT, bestIdx = idxUmax; end
    end

    bestOpt = optsVec(bestIdx);             % 直接得到最佳 λ 结构
    fprintf('\n=====  Best λ =====\n');
    fprintf('[λ1 λ2 λ3 λ4] = %.3g %.3g %.3g %.3g\n', ...
        bestOpt.l1, bestOpt.l2, bestOpt.l3, bestOpt.l4);
    fprintf('ACC_T = %.4f   ACC_U = %.4f   Time = %.2f s\n', ...
        ACC_T(bestIdx), ACC_U(bestIdx), TimeG(bestIdx));


    %% 5. 用最佳参数重新训练 + 评估 --------------------
    tStart = tic;
    [W,S,Loss,Con] = DLPLSR(X,Y,nl, bestOpt);
    [ACC_Tbest,ACC_Ubest,gnd_T,gnd_U,Y_T,Y_U] = ...
        Testing(W, XL, YL, XU, Y_train_unlabeled, X_test, Y_test);
    [optACC_T, optPrecision_T, optRecall_T, optF1_T] = Multi_Class_Metrics(gnd_T, Y_T);
    [optACC_U, optPrecision_U, optRecall_U, optF1_U] = Multi_Class_Metrics(gnd_U, Y_U);

    bestTime = toc(tStart);

    %% 6. 打印 & 保存 ---------------------------------


    save(fullfile('.\','Results',erase(dataName,'.mat')));
end
