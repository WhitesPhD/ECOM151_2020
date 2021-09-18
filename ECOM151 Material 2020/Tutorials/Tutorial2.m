%% Tutorial 2: An introduction of evaluating models
% ECOM151: Big Data Applications in Finance

%--------------------------------------------------------------------------
% Housekeeping
%--------------------------------------------------------------------------

clear all; clc; pause(0.01), randn('seed',3212), rand('seed',3212), warning off

% Question 1:
% a. Load and explore the dataset "SP500" as a Table object. The data
% contains open, close, high, and low prices as well as trading volume (mln
% USD) from February 2016 to February 2021. 

Data = readtable('SP500.csv','ReadVariableNames',true);

% b. Check the names of the dataset

disp(Data.Properties.VariableNames)

% c. Check the dimension of the dataset

[T, n]=size(Data);
disp(['Number of rows: ', num2str(T)])
disp(['Number of columns: ', num2str(n)])

% d. From the close prices construct a series of daily log returns

Returns = diff(log(Data.Close));

% e. Create a dummy variable "direction" to be equal to 1 if the return is
% positive and 0 otherwise. 

Direction = +(Returns>0);

%% Question 2: 
% a. what is the correlation of today's returns and lagged returns? Is it
% significantly different from zero? 

[rho, pval] = corr(Returns(2:end),Returns(1:end-1));

% b. From simple correlation, if the return yesterday was negative, what is the likely "direction" for today's return? 

% c. Plot the autocorrelation function

figure(1)
autocorr(Returns)


% Create two datasets: A "test" and a "train" dataset.
% Store data before the year 2019 in the object named "train".
% Store data on after the year 2019 in the object named "test". 

%% Question 3: Linear Regression Model

% a. Fit a linear regression model predicting the "direction" of the return using the train dataset.
%    Store the regression in an object named "m1". 

% b. Compare your predicted direction with the actual direction over the
% training period. 

% c. Tabulate the confusion matrix 

% d. What fraction of actual positive returns are predicted to be positive
% by the model? What fraction of actual negative returns are predicted to
% be negative?

% e. predict the direction of the returns out-of-sample over the test
% period. 

% f. Create a confusion matrix for the test period

% g. What fraction of actual positive returns are predicted to be positive
% by the model? What fraction of actual negative returns are predicted to
% be negative?


%% Question 4: Logistic regression model

% a. Fit a logisitc regression model predicting the direction of the return
% using the train data. 

% b. Compare your predicted direction with the actual direction over the
% training period. 

% c. Tabulate the confusion matrix 

% d. What fraction of actual positive returns are predicted to be positive
% by the model? What fraction of actual negative returns are predicted to
% be negative?

% e. predict the direction of the returns out-of-sample over the test
% period. 

% f. Create a confusion matrix for the test period

% g. What fraction of actual positive returns are predicted to be positive
% by the model? What fraction of actual negative returns are predicted to
% be negative?













