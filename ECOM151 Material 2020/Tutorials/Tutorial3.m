%% Tutorial 3: Input external data, compute returns, OLS regression 
% ECOM 151: big data for finance
% Feb 22, 2021
% Author: Chuanping Sun

%% input external data as table
   % addpath('C:\Users\tew399\Dropbox (QMUL-SEF)\TA\ECOM151 big data for finance\Matlab Tutorial - ECOM151');

    Data = readtable('SP500.csv');
    whos Data
    
       
    Data.Properties.VariableNames  % check variabless 
    Data(1:5, :)  % after loading a large table, you may want to preview part of the table
    Data(end-5:end, :)  
    
        
    [T,N] = size(Data);  % check table dimension
    disp(['Number of rows: ', num2str(T)])
    disp(['Number of columns: ', num2str(N)])
    
    
%% compute stock returns
    Returns = diff(log(Data.Close));   % return_t = log(P_t) - log(P_{t-1})
    
    figure (1)
    plot(Returns) 
    
    Direction = (Returns>=0); % create a dummy variable (logical values) indicating return is postive (take value of 1) or negative (take value of 0)
    
    % What is the correlation of today's returns and lagged returns? is it
    % significantly differnt from zero?    
    [rho, pval] = corr(Returns(2:end), Returns(1:end-1)); 
    
    
    % from the simple correlation, if the return yesterday was negative,
    % what is the likely direction for today's return?
       
    
    
    % Plot the autocorrelation fuction
    figure (2)
    autocorr(Returns)
    
%% save variables
    save('SP500data.mat','Data','Returns');
    
%% linear regression model

    % set train sample and test sample
    
    prop=0.9;
    T = length(Returns);
    breakpoint = floor(prop*T);
    train = Returns(1 : breakpoint);
    test = Returns(breakpoint+1 : end);
    
    %--------------------------------------------------------------------
    % run an AR(1) model using training sample
    y = train(2:end);  % dependent variable r_t
    x = train(1:end-1);  % explainatory variable r_{t-1}
   
    LM = fitlm(x,y)  % fit a linear model
    
    bhat = LM.Coefficients{2,1}
    
    % predict returns using training sample    
    yhat = bhat*x; % LM shows the intercept is not significant
    % yhat = predict(LM,x); % alternatively, use the predict function 
    
    figure(3)
    plot(1:length(y), y, '-' , 1:length(yhat), yhat, '-.')
    legend('y', 'yhat')
    
    %---------------------------------------------------------------------
    % construct confusion matrix for the training sample 
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    
    for i=1:length(y)
        if y(i)>=0 & yhat(i)>=0; TP = TP+1; 
        elseif y(i)<0 & yhat(i)<0; TN = TN+1; 
        elseif y(i)<0 & yhat(i)>=0; FP = FP+1; 
        else FN = FN+1;
        end 
    end
    
    Conf_mat_train = array2table([TP FP; FN TN] , 'VariableNames', {'actual_up', 'actual_down'}, ...
               'RowNames', {'pred_up', 'pred_down'}') 
    % true positive rate, i.e. proportion of true positive as in all positives     
    TPR = TP / (TP + FN); 
    fprintf('The true postive rate is: %6.2f \n', TPR); 
    
    %true negative rate
    TNR = TN / (TN + FP);
    fprintf('The true negative rate is: %6.2f \n', TNR);
    
    %propotion of all correction prediction
    rate =  (TP + TN)/length(y); 
    fprintf('The correct predction rate is: %6.2f \n', rate); 
    
    
    %% ----------------------------------------------------------------------
    % predict returns using testing sample    
    xx = test(1:end-1);
    yy = test(2:end);
    yhat = bhat*xx;
   
    figure(4)
    plot(1:length(yy), yy, '-' , 1:length(yhat), yhat, '-.')
    legend('yy', 'yhat')
    
    % ---------------------------------------------------------------------
    % construct confusion matrix for the testing sample
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    
    for i=1:length(yy)
        if yy(i)>=0 & yhat(i)>=0; TP = TP+1; 
        elseif yy(i)<0 & yhat(i)<0; TN = TN+1; 
        elseif yy(i)<0 & yhat(i)>=0; FP = FP+1; 
        else FN = FN+1;
        end 
    end
    
    Conf_mat_test = array2table([TP FP; FN TN] , 'VariableNames', {'actual_up', 'actual_down'}, ...
               'RowNames', {'pred_up', 'pred_down'}')    
    
    % true positive rate, i.e. proportion of true positive as in all positives     
    TPR = TP / (TP + FN); 
    fprintf('The true postive rate is: %6.2f \n', TPR); 
    
    %true negative rate
    TNR = TN / (TN + FP);
    fprintf('The true negative rate is: %6.2f \n', TNR);
    
    %propotion of all correction prediction
    rate =  (TP + TN)/length(yy); 
    fprintf('The correct predction rate is: %6.2f \n', rate); 
    
    
