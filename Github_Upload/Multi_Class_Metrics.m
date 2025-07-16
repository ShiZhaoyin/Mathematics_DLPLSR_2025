function [ACC, Precision, Recall, F1] = Multi_Class_Metrics(gnd, label)
    Confusion = confusionmat(gnd, label); 
    N = length(gnd);
    C = size(Confusion, 1);

    ACC = sum(diag(Confusion)) / N;

    % 每类的 Precision 和 Recall
    Precision_M = diag(Confusion) ./ transpose(sum(Confusion, 1));
    Recall_M = diag(Confusion) ./ sum(Confusion, 2);

    % 将 NaN 替换为 0
    Precision_M(isnan(Precision_M)) = 0;
    Recall_M(isnan(Recall_M)) = 0;

    % F1 值
    F1_M = 2 * Precision_M .* Recall_M ./ (Precision_M + Recall_M);
    F1_M(isnan(F1_M)) = 0;

    % 平均指标
    Precision = sum(Precision_M) / C;
    Recall = sum(Recall_M) / C;
    F1 = sum(F1_M) / C;
end
