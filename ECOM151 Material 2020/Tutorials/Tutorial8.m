%--------------------------------------------------------------------------
% Tutorial 7: covariance analysis, stepwise regression and categorical variables 
% ECOM151: big data applications for finance
%--------------------------------------------------------------------------


%% covariance analysis 
% find the correlation between all predictors and the response variable

% load the predictability dataset
load 'ReturnsPredictabilityData.mat'
TrainData(1:5,:)  % preview table

dataMat = TrainData{:,2:end};
coefMat = corr(dataMat);
% plot the correlation coefficient matrix in heatmap
figure()
imagesc(coefMat);
colorbar;
h = gca;
h.XTick = 1:size(coefMat,2);
h.XTickLabel = TrainData.Properties.VariableNames(2:end);
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
h.YTickLabel = []; % delete y tick label
title('Correlation coefficient heatmap')
% save the figure as .png file for reporting

% find the top 10 correlated predictors
coefTbl = array2table( abs(coefMat(end,:))' , ...  % take the absolute value
          'RowNames', TrainData.Properties.VariableNames(2:end), ...
          'VariableNames', {'abs_coef'});
sorted_coefTbl = sortrows(coefTbl, 1, 'descend');
top10 = sorted_coefTbl.Properties.RowNames(2:11); % note the first variable is the response variable 
bottom10 = sorted_coefTbl.Properties.RowNames(end-9:end);

%% Stepwise regression 

swMdl = stepwiselm(TrainData, 'ResponseVar', 'ExReturns', 'Lower', 'constant', 'Upper', 'linear', ...
                   'Verbose', 1)
               
% alternatively, use the top10 variable as predictors only.
swMdl2 = stepwiselm(TrainData, 'PredictorVars', top10, 'ResponseVar', 'ExReturns', ...
                   'Lower', 'constant', 'Upper', 'linear', 'Verbose', 2)
               
% forecast using training sample
swForecast_train = predict(swMdl2, TrainData); % the format of second input must agree with your Mdl
swMSE_train = 100*mean((swForecast_train - TrainData.ExReturns).^2);

% forecast using testing sample
swForecast_test = predict(swMdl2, TestData);
swMSE_test = 100*mean((swForecast_test - TestData.ExReturns).^2);


%% categorical variables
% categorical variable means a variale only have small number of levels. For example, 
% students' exam results can be graded as A, B, C, D and 'Fail', then this variable only
% has 5 levels. So how do we deal with CATEGORICAL variables in various models? 

% The short answer is some functions are robust with categorical variables, some
% are not.

% example: 
clear all;  load carsmall;

carTbl = table(Acceleration, Cylinders, Displacement, Horsepower, Model_Year, MPG); 
carTbl(1:5,:) % preview
summary(carTbl)

func = @(x) numel(unique(x));
varLevel = varfun(func,carTbl); % use varfun to compute each variables' level

% you will find 'cylinders' and 'model_year' have only 3 levels. so it would
% be sensible to convert them into categorical variables.
carTbl.Cylinders = categorical(carTbl.Cylinders);
carTbl.Model_Year = categorical(carTbl.Model_Year);
summary(carTbl)

% fitlm using categorical variables
Mdl = fitlm(carTbl, 'ResponseVar', 'MPG', 'CategoricalVars', {'Cylinders', 'Model_Year'})

% from the estimation result you will see that categorical variables are
% converted into dummy variables, where the first observed level is used as
% reference level.
Mdl2 = stepwiselm(carTbl, 'ResponseVar', 'MPG', 'Lower', 'constant', 'Upper', 'linear', ...
                  'CategoricalVars', {'Cylinders', 'Model_Year'}, 'Verbose', 1) % you just need to specify which vars are categorical

% these functions are robust with categorical variables: 
% fitlm, fitglm, stepwiselm, stepwiseglm, fitcensemble, fitrensemble, ...



% Meanwhile, some functions are not robust with categorical predictors such
% as: lasso, ridge, ... In this case, we can manually convert categorical
% variables to dummy variables  ----> to be continued in the next class