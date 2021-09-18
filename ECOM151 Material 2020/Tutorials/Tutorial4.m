%% Tutorial 4: Logistical regression; construct momentum factor for OLS; build functions
% ECOM 151: big data for finance
% March 1, 2021


%% logistical regression
%  ***** revision of logit model *****
%  Consider a linear regression model: y = b_0 + b_1x_1 + b_2x_2.
%  Suppose y is a probability measure, which measures the likelihood of stock prices will
%  go up. However, the linear regression model will not guarantee the
%  fitted value will be between 0 and 1. 

% logistical regression do the following: 
% given a linear regression model: f(x)= b_0 + b_1x_1 + b_2x_2
% p = Pr(price will go up |x) = exp[f(x)] / {1 + exp[f(x)]}. This guarantees that p is between 0 and 1. 
% It also implies: log[p/(1-p)] = logit(p) = b_0 + b_1x_1 + b_2x_2.
% *** converted into an OLS regression model ***

%% loading data
    load SP500data ;
    
%% estimate a logistic model: use lagged returns as predictor
    prop=0.9;
    threshold = 0.5;
    T = length(Returns);
    breakpoint = floor(prop*T);
    train = Returns(1 : breakpoint);
    test = Returns(breakpoint+1 : end); 
    
    xx_train = train(1:end-1);
    yy_train = train(2:end)>=0 ;  % In a logistic model, the response variable is a binary variable
    mdl = fitglm(xx_train,yy_train, 'Distribution', 'binomial')  % fit a logistic binomial model 
    
    yhat_train = predict(mdl, xx_train); % predict y given xx
    yyhat_train = yhat_train>=threshold;  % if probability greater than 0.5 then predict up
    
    
%% ----------------------------------------------------------------------
    % construct confusion matrix for the training sample 
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    
    for i=1:length(yy_train)
        if yy_train(i)==1 & yyhat_train(i)==1; TP = TP+1; 
        elseif yy_train(i)==0 & yyhat_train(i)==0; TN = TN+1; 
        elseif yy_train(i)==0 & yyhat_train(i)==1; FP = FP+1; 
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
    rate =  (TP + TN)/(TP + TN + FP + FN); 
    fprintf('The correct predction rate is: %6.2f \n', rate);        
           
%% ------- predict return directions using the testing sample ---------
    xx_test = test(1:end-1);
    yy_test = test(2:end)>=0 ; 
    yhat_test = predict(mdl,xx_test);
    yyhat_test = yhat_test>=threshold;
    
    
%% 
    % construct confusion matrix for the testing sample 
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    
    for i=1:length(yy_test)
        if yy_test(i)==1 & yyhat_test(i)==1; TP = TP+1; 
        elseif yy_test(i)==0 & yyhat_test(i)==0; TN = TN+1; 
        elseif yy_test(i)==0 & yyhat_test(i)==1; FP = FP+1; 
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
    rate =  (TP + TN)/(TP + TN + FP + FN); 
    fprintf('The correct predction rate is: %6.2f \n', rate); 
    
    
    
 %% OLS regression adding momentum factor   
    mom = 21; % one-month momentum
    y = train(mom+1:end); 
    xmom = []; % create momentum factor
    for i = mom : length(train)-1
        xmom = [xmom; sum(train(i-mom+1:i))];
    end 
        
    xlag = train(mom:end-1);
    reg = [xlag, xmom];
    
    LM2 = fitlm(reg,y);
    yhat = predict(LM2, reg);
    
    confMat_train = compute_ConfMat(y, yhat)  
    
    % ------------------- testing sample ------------------
    y_test = test(mom+1:end);
    xmom=[]; % create momeentum factor for testing sample
    for i = mom : length(test)-1
        xmom = [xmom; sum(test(i-mom+1:i))];
    end  
    
    xlag_test = test(mom:end-1);
    reg_test = [xlag_test, xmom];
    yhat_test = predict(LM2, reg_test);
    
    confMat_test = compute_ConfMat(y_test, yhat_test) % write a function to compute repeated tasks
    
    
    
    
    
    
    
    
    
    