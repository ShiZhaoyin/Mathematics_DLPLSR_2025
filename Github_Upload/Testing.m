function [ACC_T,ACC_U,gnd_T,gnd_U,Y_T,Y_U] = Testing(W,X_train_labeled, Y_train_labeled, X_train_unlabeled, Y_train_unlabeled, X_test, Y_test)
N_U = size(X_train_unlabeled,2); N_T = size(X_test,2);
%%
D1 = pdist2(X_test'*W,X_train_labeled'*W);
[~,idx] = min(D1,[],2);
Y_test_hat = Y_train_labeled(:,idx);
[~,y_test_hat] = max(Y_test_hat,[],1);
[~,y_test] = max(Y_test,[],1);
ACC_T = sum(y_test_hat == y_test)/N_T; 
gnd_T = y_test;   Y_T = y_test_hat;  
%% 
D2 = pdist2(X_train_unlabeled'*W,X_train_labeled'*W);
[~,idx] = min(D2,[],2);
Y_unlabeled_hat = Y_train_labeled(:,idx);
[~,y_unlabeled_hat] = max(Y_unlabeled_hat,[],1);
[~,y_unlabeled] = max(Y_train_unlabeled,[],1);
ACC_U = sum(y_unlabeled_hat == y_unlabeled)/N_U; 
gnd_U = y_unlabeled;   Y_U = y_unlabeled_hat;  
end