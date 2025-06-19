   dataName = 'Jaffe.mat'

    load(fullfile(dataPath, dataName));       % XL,XU,YL,X_test,Y_test

    XL = X_train_labeled;
    XU = X_train_unlabeled;
    YL = Y_train_labeled;
    X  = [XL, XU];                           % d×n
    X  = X ./ (vecnorm(X)+eps);               % 列 L2 归一

    [d,n] = size(X);  nl = size(XL,2);

     c     = size(YL,1);
    Y       = zeros(c,n);  Y(:,1:nl) = YL;

    opts.lambda1 = 1;
    opts.lambda2 = 1;
    opts.lambda3 = 1;
        tic;
        [W,,S,Loss,Con] = DLPLSR(X,Y,nl,opts);      % 训练
        [ACC_T,ACC_U,gnd_T,gnd_U,Y_T,Y_U] = ...
        Testing(W, XL, YL, XU, Y_train_unlabeled, X_test, Y_test);
        TimeG(idx) = toc;

    [optACC_T, optPrecision_T, optRecall_T, optF1_T] = Multi_Class_Metrics(gnd_T, Y_T);
    [optACC_U, optPrecision_U, optRecall_U, optF1_U] = Multi_Class_Metrics(gnd_U, Y_U);

