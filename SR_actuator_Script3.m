% 1. Importing the data from CSV file and verify its structure
data = readtable("C:\Users\sudhe\Downloads\Soft_Actuator_FEA_Dataset.csv"); %include the path of the downloaded file
head(data)

% 2. Compute and display descriptive data statistics for each numeric variable
numericData = data(:, varfun(@isnumeric, data, 'OutputFormat', 'uniform'));
stats = table();
stats.Variable = numericData.Properties.VariableNames';
stats.Mean = varfun(@mean, numericData, 'OutputFormat', 'uniform')';
stats.Median = varfun(@median, numericData, 'OutputFormat', 'uniform')';
stats.Std = varfun(@std, numericData, 'OutputFormat', 'uniform')';
stats.Min = varfun(@min, numericData, 'OutputFormat', 'uniform')';
stats.Max = varfun(@max, numericData, 'OutputFormat', 'uniform')';
disp(stats);

% 3. Visualization of deformation values over the distributed data
figure;
histogram(numericData.Deformation)
xlabel('Deformation'), ylabel('Frequency'), title('Histogram of Deformation');

rng(42); % For reproductivity

% 4. Here we define features/input columns and target/output for modeling
features = data{:, {'Height', 'Length', 'Pressure', 'Thickness'}};
target = data.Deformation;

% 5. Spliting the data into training (80%) & test sets (20% holdout)
numTrain = round(0.8 * height(data));
idxTrain = randperm(height(data), numTrain);
idxTest = setdiff(1:height(data), idxTrain);
Xtrain = features(idxTrain, :); ytrain = target(idxTrain);
Xtest  = features(idxTest, :);  ytest = target(idxTest);

% 6. Configuration of K-fold Cross Validation partition
K = 5;
cv = cvpartition(numTrain, 'KFold', K);

% --------------- 7. Selection of Models -------------------
% ----------- Random Forest Cross-Validation ---------------
numTrees_grid = [10, 25, 50, 100, 200, 400];
mean_rmse_rf = zeros(length(numTrees_grid),1);
for i = 1:length(numTrees_grid)
    numTrees = numTrees_grid(i);
    fold_rmse = zeros(K,1);
    for k = 1:K
        trainIdx = training(cv,k); testIdx = test(cv,k);
        XtrainCV = Xtrain(trainIdx, :); ytrainCV = ytrain(trainIdx);
        XvalCV   = Xtrain(testIdx, :);  yvalCV = ytrain(testIdx);
        mdlRF = fitrensemble(XtrainCV, ytrainCV, ...
            'Method', 'Bag', ...
            'NumLearningCycles', numTrees, ...
            'Learners', templateTree('MinLeafSize',5));
        ypredRF = predict(mdlRF, XvalCV);
        fold_rmse(k) = sqrt(mean((yvalCV - ypredRF).^2));
    end
    mean_rmse_rf(i) = mean(fold_rmse);
end

% --- 8. Random Forest CV learning curve vs. number of trees ---

figure;
plot(numTrees_grid, mean_rmse_rf, '-o', 'LineWidth',2,'MarkerSize',7);
xlabel('Number of Trees');
ylabel('Mean CV RMSE');
title('Random Forest: CV RMSE vs Number of Trees');
grid on;

% ---- 9. Gaussian Process Regression (GPR) Cross-Validaton ---
sigma_grid = [0.25, 0.5, 1, 2, 4];
mean_rmse_gpr = zeros(length(sigma_grid),1);
K = 5; % Number of folds

for i = 1:length(sigma_grid)
    sigma = sigma_grid(i);
    fold_rmse = zeros(K,1);
    for k = 1:K
        trainIdx = training(cv,k); testIdx = test(cv,k);
        XtrainCV = Xtrain(trainIdx, :); ytrainCV = ytrain(trainIdx);
        XvalCV   = Xtrain(testIdx, :);  yvalCV = ytrain(testIdx);
        mdlGPR = fitrgp(XtrainCV, ytrainCV, ...
            'KernelFunction', 'squaredexponential', ...
            'Sigma', sigma, ...
            'Standardize', true);
        ypredGPR = predict(mdlGPR, XvalCV);
        fold_rmse(k) = sqrt(mean((yvalCV - ypredGPR).^2));
    end
    mean_rmse_gpr(i) = mean(fold_rmse);
end

% --- GPR CV learning curve vs. kernel scale ---
figure;
plot(sigma_grid, mean_rmse_gpr, '-s', 'LineWidth',2,'MarkerSize',7);
xlabel('Sigma (Noise Standard Deviation)');
ylabel('Mean CV RMSE');
title('GPR: CV RMSE vs Sigma');
grid on;


% ---> 10. Auto-tuning for final best parameters and test set performance for both RF & GPR (OPTIONAL)<---
% Note: The K-fold cross validation is performed to auto tune the hyperparameter with training and validation on multiple data splitting. 
% This Extra final train/test code builds a one best mode on all training and assessment of its honest performance with independent test data set. 
% This Extra code doesn't replace K-fold CV but enhances it to final model evalution.
best_idx_rf = find(mean_rmse_rf == min(mean_rmse_rf), 1);
best_numTrees = numTrees_grid(best_idx_rf);
mdlRF_final = fitrensemble(Xtrain, ytrain, ...
    'Method', 'Bag', ...
    'NumLearningCycles', best_numTrees, ...
    'Learners', templateTree('MinLeafSize',5));
ypredRF_test = predict(mdlRF_final, Xtest);
RMSE_RF_test = sqrt(mean((ytest - ypredRF_test).^2));
fprintf('RF Test RMSE = %.4f\n', RMSE_RF_test);

best_idx_gpr = find(mean_rmse_gpr == min(mean_rmse_gpr), 1);
best_sigma = sigma_grid(best_idx_gpr);
mdlGPR_final = fitrgp(Xtrain, ytrain, ...
    'KernelFunction', 'squaredexponential', ...
    'Sigma', 0.4, ...
    'Standardize', true);
ypredGPR_test = predict(mdlGPR_final, Xtest);
RMSE_GPR_test = sqrt(mean((ytest - ypredGPR_test).^2));
fprintf('GPR Test RMSE = %.4f\n', RMSE_GPR_test);


% 11. Predict and compare by calculating the key Test Metrics
yTestPred_RF = predict(mdlRF, Xtest);
[yTestPred_GPR, yTestStd_GPR] = predict(mdlGPR, Xtest); % with uncertainty

% Metrics for RF
rmse_rf = sqrt(mean((yTestPred_RF - ytest).^2));
mae_rf  = mean(abs(yTestPred_RF - ytest));
r2_rf   = 1 - sum((ytest - yTestPred_RF).^2) / sum((ytest - mean(ytest)).^2);

%Metrics for GPR
rmse_gpr = sqrt(mean((yTestPred_GPR - ytest).^2));
mae_gpr  = mean(abs(yTestPred_GPR - ytest));
r2_gpr   = 1 - sum((ytest - yTestPred_GPR).^2) / sum((ytest - mean(ytest)).^2);

%P 12. Parity Plot of the Actual vs Predicted values
figure; hold on
scatter(ytest, yTestPred_RF, 30, 'b', 'filled');
scatter(ytest, yTestPred_GPR, 30, 'r', 'filled');
plot([min(ytest), max(ytest)], [min(ytest), max(ytest)], 'k--');
legend('RF','GPR','Ideal','Location','best');
xlabel('Actual'); ylabel('Predicted'); title('Parity Plot');

% Adding Metrics summary on the plot
xl = xlim; yl = ylim;
xpos = xl(1) + 0.05*(xl(2)-xl(1));
ypos = yl(2) - 0.02*(yl(2)-yl(1));

text(xpos, ypos, ...
    sprintf('RF: RMSE=%.2f, MAE=%.2f, R^2=%.2f\nGPR: RMSE=%.2f, MAE=%.2f, R^2=%.2f', ...
    rmse_rf, mae_rf, r2_rf, rmse_gpr, mae_gpr, r2_gpr), ...
    'VerticalAlignment','top','FontSize',12,'BackgroundColor','w');

hold off

% 13. Force to column vectors & check lengths (OPTIONAL) & the code runs perfectly without the check.
ytest = ytest(:);
yTestPred_RF = yTestPred_RF(:);
yTestPred_GPR = yTestPred_GPR(:);

if length(ytest) ~= length(yTestPred_RF)
    error('ytest and yTestPred_RF must be the same length.');
end
if length(ytest) ~= length(yTestPred_GPR)
    error('ytest and yTestPred_GPR must be the same length.');
end

% Residuals
res_rf = ytest - yTestPred_RF;
res_gpr = ytest - yTestPred_GPR;

% Histogram of Residuals for both RF & GPR
figure;
histogram(res_rf);
xlabel('Residual (Actual - Predicted)');
ylabel('Frequency'); 
title('RF Residuals');

figure;
histogram(res_gpr);
xlabel('Residual (Actual - Predicted)');
ylabel('Frequency');
title('GPR Residuals');

% Residuals vs Predicted scatterplot for RF
figure;
scatter(yTestPred_RF, res_rf, 'filled');
xlabel('Predicted');
ylabel('Residuals');
title('Residuals vs Predicted (RF)');

% Residuals vs Predicted scatterplot for GPR
figure;
scatter(yTestPred_GPR, res_gpr, 'filled');
xlabel('Predicted');
ylabel('Residuals');
title('Residuals vs Predicted (GPR)');
% Note: While the Histrogram shows the error distribution and scatterplot--
% brings out the bias.


% 14. Q-Q Plots for Residuals shows their normality of errors
figure;
subplot(1,2,1)
qqplot(res_rf)
title('Q-Q Plot of RF Residuals');
subplot(1,2,2)
qqplot(res_gpr)
title('Q-Q Plot of GPR Residuals');

% 15. Feature Importance for RF
% In here, we will conclude which features are crucial and plot them for RF predictions.
imp = predictorImportance(mdlRF);
figure;
bar(imp)
set(gca, 'XTickLabel', {'Height','Length','Pressure','Thickness'})
ylabel('Importance')
title('Feature Importance - Random Forest')


% 16. Partial Dependence plots --> This displays the change of deformation as Height & Pressure Varies

% Height vs Deformation 
figure;
plotPartialDependence(mdlRF, 1); % 1 = Height
xlabel('Height');
ylabel('Predicted Deformation');
title('Partial Dependence of Deformation on Height');

% Pressure vs Deformation
figure;
plotPartialDependence(mdlRF, 3); % 3 = Pressure
xlabel('Pressure');
ylabel('Predicted Deformation');
title('Partial Dependence of Deformation on Pressure');



% 17. GPR Uncertanity Bands
% The Plot predicts with error bars to visualize uncertainity for every test sample
figure;
errorbar(1:length(ytest), yTestPred_GPR, 2*yTestStd_GPR, 'o')
hold on; plot(1:length(ytest), ytest, 'r.', 'MarkerSize',12)
title('GPR Test Predictions with Uncertainty');
xlabel('Sample'); ylabel('Deformation'); legend('Prediction \pm 2\sigma','True Value');

% 18. Single point predictions, Confidence Assessment & Recommendations
% Interpolative point
x_new1 = [5.25, 55, 1250, 1.05];
pred_rf_1 = predict(mdlRF, x_new1);
[pred_gpr_1, std_gpr_1] = predict(mdlGPR, x_new1);

% Edge point
x_new2 = [8.0, 100, 5000, 1.2];
pred_rf_2 = predict(mdlRF, x_new2);
[pred_gpr_2, std_gpr_2] = predict(mdlGPR, x_new2);

% Print interpolative results
fprintf('RF Interpolative: %.2f\n', pred_rf_1);
fprintf('GPR Interpolative: %.2f (Uncertainty: %.2f)\n', pred_gpr_1, std_gpr_1);

% Print edge results
fprintf('RF Edge: %.2f\n', pred_rf_2);
fprintf('GPR Edge: %.2f (Uncertainty: %.2f)\n', pred_gpr_2, std_gpr_2);

% Threshold for uncertainty (example)
uncertainty_threshold = 0.65;

% Confidence assessment and recommendation for interpolative point
if std_gpr_1 < uncertainty_threshold
    fprintf('Interpolative prediction confidence is high. Model output can be trusted.\n');
else
    fprintf('Interpolative prediction has high uncertainty. Recommend running FEA to validate.\n');
end

% Confidence assessment and recommendation for edge point
if std_gpr_2 < uncertainty_threshold
    fprintf('Edge prediction confidence is high. Model output can be trusted.\n');
else
    fprintf('Edge prediction has high uncertainty. Recommend running FEA to validate.\n');
end
