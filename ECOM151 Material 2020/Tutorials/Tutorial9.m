% ===============
%  Tutorial 9: Decision Trees, Random Forest
% ===============

%% Some theoretical background
% decision trees: classification tree / regression tree
doc decision trees

% Ensemble methods: 
% Ensemble methods use multiple learning algorithms to obtain better 
% predictive performance by aggregating constituent learning algorithms.

doc fitcensemble 
doc fitrensemble

% ensemble algo
doc Ensemble Algorithms  % 'bag' -> classification and regression
                         % 'AdaBoostM1' -> binary classification 
                         % 'AdaBoostM2' -> multiclass classification

%% example of using fitrensemble
load carbig
Cylinders = categorical(Cylinders);
Model_Year = categorical(Model_Year);
Origin = categorical(cellstr(Origin));
Tbl = table(Cylinders,Displacement,Horsepower,Weight,Acceleration,Model_Year,Origin, MPG);
cateVarNames = {'Cylinders', 'Model_Year', 'Origin'}; 

% check levels of each variable
func = @(x) numel(unique(x));
varLevel = varfun(func,Tbl); 


%% train bagged ensemble of a regression tree  
t = templateTree('MaxNumSplits', 5,...
    'PredictorSelection','interaction-curvature','Surrogate','on');  % check all 'name-value' paired values in the documentation
rng(1); % For reproducibility
Mdl = fitrensemble(Tbl,'MPG','Method','Bag','NumLearningCycles',200, ...
     'Learners',t);  % check all 'name-value' paired values in the documentation

% some important 'name-value' pairs for fitensemble:
% 'MaxNumSplits', 5 -> specify the maximum number of splits. 
% 'PredictorSelection','interaction-curvature' -> the default value is
% 'CART', but biased towards variables have higher levels. So if contains
% catogorical vars, set to 'interaction-curvature'.
% 'Surrogate','on' -> if you have missing data in  your dataset. However,
% if 'on', it makes computation slower.

% some important 'name-value' pairs for fitrensemble:
% 'Method' -> ensemble algorithms. Using 'bag' (bootstrap aggregating) for
% both classification and regression problems.
% 'Learners',t -> in our exercise we are using the tree learner.


%% estimate MSE
yhat = predict(Mdl,Tbl); 
MSE = mean((Tbl.MPG - yhat).^2,'omitnan'); % 'omitnan' is to make function robust to missing data 


%% predictor importance estimation 
impOOB = oobPermutedPredictorImportance(Mdl);
figure(); 
bar(impOOB)
xlabel('Predictor Variables')
ylabel('Importance')
h = gca;
h.XTick = 1:length(Mdl.PredictorNames);
h.XTickLabel = Mdl.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
title('Predictor Importance Estimates')

% find top 5 important predictors
ImportanceTbl = array2table(abs(impOOB)', 'RowNames', Mdl.PredictorNames, 'VariableNames', {'abs_coef'});
ImportanceTbl = sortrows(ImportanceTbl);
Top5 = ImportanceTbl.Properties.RowNames(end-4:end);
disp(Top5)

% ------------------------------------------------------------------------------------------------------

%% Furthermore, you can use TreeBagger() to grow a random forest % doc TreeBagger
Mdl2 = TreeBagger(50, Tbl, 'MPG',  'NumPredictorsToSample', floor(size(Tbl,2)/3), 'OOBPrediction', 'on', ...
       'OOBPredictorImportance', 'on', 'Method', 'regression','CategoricalPredictors', cateVarNames, ...
       'PredictorSelection','curvature', 'Reproducible', true)
                % key 'Name-Value' pair: 
                % 'NumPredictorsToSample', floor(size(Tbl,2)/3) -> this is where the RANDOM forest comes from
                
%% predictor importance estimation
imp = Mdl2.OOBPermutedPredictorDeltaError;

figure;
bar(imp);
title('Predictor Importance Estimation');
ylabel('Predictor importance estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = Mdl2.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

%% MSE of the random forest model
yhat2 = predict(Mdl2, Tbl);
MSE2 = mean((Tbl.MPG - yhat2).^2,'omitnan');




