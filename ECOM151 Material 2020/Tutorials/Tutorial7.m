%--------------------------------------------------------------------------
% Tutorial 6: LASSO and Elastic Net 
% ECOM151: big data applications for finance
%--------------------------------------------------------------------------

%% LASSO revision
% Lasso solve the following problem
% b_lasso = argmin_b ||y - xb||_2^2 + lambda* ||b||_1
% comparison with ridge: penalty term for ridge is the sum of *suquared b's,
% while lasso penalty is sum of *absolute values of b's.
% Ridge shrinks all parameters (bhat) towards zero, but not as small as
% zeros; while lasso shrinks some (unimportant) parameters to exactly zero.
% Therefore, lasso also performs factor selection during estimation. **
% It particulary pupular among high-dimensional applications.

%% use Lasso and last week's dataset to predict stock returns 

% load data
load 'ReturnsPredicatabilityData'

TrainData(1:5,:) % to preview the table

% use the training data to estimate the lasso regression model
doc lasso

x = TrainData{:,2:end-1}; % use the curly bracket to output matrix
y = TrainData{:,end};

scaleFlag = false;  % standardized data gives nice parameter interpretation
                    % unstandardized data is better for prediction
if scaleFlag == true
    x = (x-mean(x)) ./ repmat(std(x),size(x,1), 1);
    y = (y-mean(y))/std(y);
end

%[b_lasso, fitInfo] = lasso(x, y, 'Lambda', 1e-4); % specify lambda value
[b_lasso, fitInfo] = lasso(x, y, 'CV', 10); % Lasso with 10-fold Cross-validation
%[b_lasso, fitInfo] = lasso(x, y, 'DFmax', 6); % DFmax specify the number of non-zero coefficients
                     % note that Lasso will produce a series of lasso estimates using various lambda values, 
                     % which is similar to ridge estimator, but without
                     % specifically define the candidate values of lambdas.
                     
fitInfo  % check what included in fitInfo
                     
% plot the lasso estimates
RegNames = TrainData.Properties.VariableNames(2:end-1);
figure()
plot(fitInfo.Lambda, b_lasso', 'LineWidth',2);  % don't forget to transpose b_lasso
legend(RegNames, 'Interpreter', 'none')
xlabel('Lambda value')
ylabel('LASSO estimates')
title('Lasso regression')

% find the lasso estimate with the smallest MSE
indx = find(fitInfo.MSE <= min(fitInfo.MSE));
bhat = b_lasso(:,indx)

% forecast stock returns using the testing sample
xx = TestData{:,2:end-1};
yy = TestData{:,end};
lassoForecast = [ones(size(xx,1), 1), xx]*[fitInfo.Intercept(indx);bhat];
lassoMSE = 100*mean((lassoForecast - yy).^2)

%% Elastic Net 
% revision of Elastic Net: EN solve the following problem
% b_EN = argmin_b ||y - xb||_2^2 + pen, where
% pen = lambda* [alpha*||b||_1 + (1-alpha)*||b||_2^2], where
% alpha is a scaler between 0 and 1.

[b_EN, fitInfo_EN] = lasso(x, y, 'CV', 10, 'Alpha', 0.75);
indx_EN = find(fitInfo_EN.MSE <= min(fitInfo_EN.MSE));
bhat_EN = b_EN(:,indx_EN)

% forecast stock returns using the testing sample
ENForecast = [ones(size(xx,1), 1), xx]*[fitInfo_EN.Intercept(indx_EN);bhat_EN];
ENMSE = 100*mean((ENForecast - yy).^2)



%% However, LASSO performs the best in high-dimensional setting. 
% This exercise only has 13 predictors. In reality, it is often that we
% have to face hundreds if not thousands of predictors,including many weak 
% predictors, and we do not have knowledge of which are strong preditors
% nor do the weak ones. 
% In this case, LASSO will perform factor selection during estimation. 
% It will simultanesouly select strong factors and discard weak ones.

% Now let's use the Monte Carlo (MC) simulation method to see how different estimators
% perform in the high-dimensional setting. Since the MC method allows us to know the true
% parameter values (b0), we can compare the estimated bhat using various methods 


%% Data Generate Process (DGP)
% simulate x, and y and specify true values of b0
rng(123, 'twister');
k = 50;
n = 200;
x = randn(n,k);
b0 = zeros(k,1);
b0(1:10) = 0.5; % the first 10 elements are non-zero, elsewhere are zeros
y = x*b0 + randn(n,1)*0.1;

%% OLS estimator
mdl = fitlm(x,y);
b_ols = mdl.Coefficients{2:end, 1}; % exclude the intercept
% deviation from the true b0
ols_err = norm(b_ols - b0)  % norm (L2 norm) measures the distance between two vectors


%% Lasso estimator
[B, fitInfo] = lasso(x,y,'CV',10);
b_lasso = B(:,fitInfo.MSE <= min(fitInfo.MSE));
lasso_err = norm(b_lasso - b0)




%% compare the lasso/ols estimates with the true value
[b0, b_lasso, b_ols]


%% *** remarks ***
% LASSO works the best in the high-dimensional setting, it also performs factor
% slection, which is of great importance to many applications.

% Cross-validation (CV) is typically used to determine the shrinkage parameter
% lambda. Understand the intuition and general idea behind CV.

% Elastic Net (EN) is a penalised regression model where the penalty term
% is a combination of LASSO and Ridge penalties. 

% Monte Carlo Simulation method allows us to specify a true Date Generate
% Process (DGP), then we can use simulated data to estimate models, and
% compare the estimated coefficients with the true ones.





















