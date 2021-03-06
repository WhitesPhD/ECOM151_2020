
%--------------------------------------------------------------------------
% Tutorial 5: Ridge regression
% ECOM151: big data applications for finance
%--------------------------------------------------------------------------

clear all; clc; randn('seed',3212), rand('seed',3212), warning off

%% Question 1:
% Upload the train and test data

TrainData = readtable('DataReturnsPredictability.xlsx','Sheet',1,'ReadVariableNames',true);
TestData  = readtable('DataReturnsPredictability.xlsx','Sheet',2,'ReadVariableNames',true);

TrainData(1:5,:)  % preview tables
size(TrainData)
size(TestData)

sum(ismissing(TrainData))  % check missing data
sum(ismissing(TestData))

% Question 1:

% Estimate a simple predictive regression by using least squares and
% The prediction can be simply generated by using the TestData
% The MSE should be computed based on the test data

% you can use the command "fitlm"

Mdl         = fitlm(TrainData(:,2:end)); % estimate the model using the training data 

OLSforecast = predict(Mdl, TestData{:, 2:end-1}); % forecast based on testing sample  
                                                  % note: use curly bracket to output matrix

OLSmse      = 100*mean((TestData.ExReturns - OLSforecast).^2);

%% Question 2:

% Estimate a simple predictive regression by using a Ridge regression model and
% The prediction can be generated by using the TestData
% The MSE should be computed based on the test data

% you can use the command "ridge"

% *** ridge regression revision ***
% ridge estimator solve the following problem:
% b_ridge = argmin_b ||y-xb||_2^2 + lambda*||b||_2^2

doc ridge  

lambda = linspace(0, 3000, 100);
B = ridge(TrainData.ExReturns,table2array(TrainData(:,2:end-1)),lambda);  % ridge regression; scaled (default)
% Scaled means all predictors are demeaned and then scaled to have unit
% variance; reponse variable is demeaned

RegNames = TrainData.Properties.VariableNames(2:end-1);
figure()
plot(lambda, B', 'LineWidth',2);  % don't forget to transpose B
legend(RegNames, 'Interpreter', 'none')
xlabel('Lambda value')
ylabel('Estimated coefficients')
title('Scaled Ridge regression without intercept')


% predict returns using test sample
ridgeForecast = nan(size(TestData.ExReturns, 1), length(lambda)); % create a matrix to store predicted returns for each choice of lambda

for i = 1:length(lambda)
    ridgeForecast(:,i) =  TestData{:,2:end-1}*B(:,i);
end 

ridgeErr = ridgeForecast - TestData.ExReturns;
ridgeMSE = 100*mean(ridgeErr.^2)


%% alternatively...  (unscaled)
lambda = linspace(0, 100, 100);
Beta = ridge(TrainData.ExReturns,table2array(TrainData(:,2:end-1)),lambda,0);  % ridge regression, with intercept
RegNames = ['Intercept', TrainData.Properties.VariableNames(2:end-1)];
figure()
plot(lambda, Beta', 'LineWidth',2);
legend(RegNames , 'Interpreter', 'none')
xlabel('Lambda value')
ylabel('Estimated coefficients')
title('Unscaled Ridge regression with intercept')

% predict returns using test sample
ridgeForecast2 = nan(size(TestData.ExReturns, 1), length(lambda)); % create a matrix to store predicted returns for each choice of lambda
for i = 1:length(lambda)
    ridgeForecast2(:,i) = [ ones(size(TestData,1), 1), TestData{:,2:end-1}]*Beta(:,i);
end

ridgeErr2 = ridgeForecast2 - TestData.ExReturns;
ridgeMSE2 = 100*mean(ridgeErr2.^2)

% to find the lambda that gives the smallest MSE
lambda_optimal = lambda(ridgeMSE2 <= min(ridgeMSE2))

%%  *** Remarks ***
% Scaled ridge regression gives nice iterpretation of each predictors;
% unscaled ridge regression gives better prediction result (smaller MSE)


