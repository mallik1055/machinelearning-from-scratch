fid = fopen('data/featureTypes.txt');
% Read all lines & collect in cell array
featureNames = textscan(fid,'%s','delimiter','\n'); 
%train data
X_t = csvread('data/trainData.csv',0,1)';
y_t = csvread('data/trainLabels.csv',0,1);
%validation data
X_v = csvread('data/valData.csv',0,1)';
y_v = csvread('data/valLabels.csv',0,1);
%test data
X_test = csvread('data/testData.csv',0,1)';
%featureLabels



L = [0.01 0.1 1 10 100 1000];

i=0;
rmseData = zeros(3,size(L,2));


for l = L
    
    i=i+1;
    
    [w,b,obj,cvErrs] = ridgeReg(X_t,y_t,l);
    
    rmse_t = getRMSE(w,b,X_t,y_t);
    rmse_v = getRMSE(w,b,X_v,y_v);
    rmse_loocv = sqrt(sum(cvErrs.^2)/size(cvErrs,1));
    
    rmseData(:,i) = [rmse_t,rmse_v,rmse_loocv];
end


%Plot RMSE vs lambda
figure
plot(log10(L),rmseData,'MarkerSize',10,'Marker','.');
legend('RMSE_{train}', 'RMSE_{validation}', 'RMSE_{LOOCV}');
xlabel('log_{10}(\lambda)');

%for reference
rmseData = [L;rmseData];


l_min = 1; %from examining the graph above
%Find w,b,lambda for lambda_min
%objective value, the sum of square errors (on training data), the value of the regularization term.
[w,b,obj,cvErrs] = ridgeReg(X_t,y_t,l_min);
regTerm = l_min*norm(w)^2;
rmse_t = getRMSE(w,b,X_t,y_t);

disp("For lambda_minErr = 1, obj, RMSE & Reg term are");
disp(obj);
disp((rmse_t^2)*size(X_t,2));
disp(regTerm);

%get top 10 features
%Normalize the data so that weights convey comparable information
X_t = normalize(X_t,2);
[w,b,obj,cvErrs] = ridgeReg(X_t,y_t,l);

[A B]=maxk(abs(w),10);
majorFeat = featureNames{1}(B);

%get bottom 10 features
[A B]=mink(abs(w),10);
minorFeat = featureNames{1}(B);

function err = getRMSE(w,b,X,y)
    w1 = [w;b];
    X1 = [X;ones(1,size(X,2))];
    err = getRMSEHelper(X1'*w1,y);
end


function err = getRMSEHelper(A,B)
    Z = A - B;
    N = size(A,1);
    err = sqrt(sum(Z.^2)/N);
end



function [w,b,obj,cvErrs] = ridgeReg(X,y,l)

    k = size(X,1); %num features
    n = size(X,2); %num samples
    X1 = [X;ones(1,n)];

    I1 = [eye(k),zeros(k,1);zeros(1,k),0];

    C = X1*X1' + l*I1;
    d = X1*y;
    
    Cinv = pinv(C);
    W = Cinv*d;
    w = W(1:k,:);
    b = W(k+1,:);
    Z = W'*X1 - y';
    obj = l*norm(w)^2 + sum(Z .^ 2);


    cvErrs = zeros(n,1);
    
    %calculating cvErrs
    %efficient way
    for i = 1 : n
        %x = X(:,i);
        x1 = X1(:,i);
        y_ = y(i,:);
        cvErr = (W'*x1 - y_')/(1-x1'*Cinv*x1);
        cvErrs(i) = cvErr;
    end
    
end




