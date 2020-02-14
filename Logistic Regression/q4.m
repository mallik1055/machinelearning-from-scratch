

try load('hw3_workspace')
    disp("Attention!! Loading t,CV,Test data from hw3 workspace");    
catch
    %load manually
    
    [X_t,ID_X_t,Y_t,ID_Y_t] = loadData('Train_Features.csv','Train_Labels.csv'); 
    [X_v,ID_X_v,Y_v,ID_Y_v] = loadData('Val_Features.csv','Val_Labels.csv');
    [X_test,ID_X_test] = loadXData('Test_Features.csv');
    
    save('hw3_workspace','X_t','Y_t','X_v','Y_v','ID_X_t','ID_X_v','ID_Y_t','ID_Y_v','X_test','ID_X_test');
    
end

X_t = nPAddFeat(X_t);
X_v = nPAddFeat(X_v);
X_test = nPAddFeat(X_test);


% q2(X_t,Y_t);
q1_3_4(X_t,Y_t,X_v,Y_v);
% q5(X_t,Y_t,X_v,Y_v,X_test,ID_X_test);
%revFeatEngg(X_t,Y_t,X_v,Y_v,X_test,ID_X_test);


function revFeatEngg(X_t,Y_t,X_v,Y_v,X_test,ID_X_test)

    for i = 1:size(X_t,2)
        
        fprintf("Removing col num %i\n",i);
        
        x_t = X_t;
        x_t(:,i) = [];
        
        y_t = Y_t;
        y_t(:,i) = [];
        
        x_v = X_v;
        x_v(:,i) = [];
        
        y_v = Y_v;
        y_v(:,i) = [];       
        
        
        q4(x_t,y_t,x_v,y_v,X_test,ID_X_test)      
        
        
    end


end



function q5(X_t,Y_t,X_v,Y_v,X_test,ID_X_test)
    delta = 0.00001;
    eta_1 = 3;
    eta_0 = 5;
    max_epoch = 1000;
    batch_size = 40;
    num_c = 4;
    %to check acc of featureEngineering
%      theta_init  = zeros(size(X_t,1),num_c-1); 
    theta_init  = rand(size(X_t,1),num_c-1); 
    [theta_t_opti,last_epoch_t,L_t,Acc_t] = stochGradDesc(theta_init,X_t,Y_t,max_epoch,eta_0,eta_1,delta,batch_size,num_c); 
    %check accuracy on the validation data
    acc = getAccuracy(Y_v,predictLabel(theta_t_opti,X_v,num_c));
    fprintf("Acc_v on theta_train is %i\n",acc);
%     fprintf("Max,min,end Acc_v on theta_train is %i %i %i\n",max(Acc_t),min(Acc_t),Acc_t(end));
    
%     return;

    %Now train on train+CV data
    
    X = [X_t,X_v];
    Y = [Y_t,Y_v];

    theta_init  = zeros(size(X,1),num_c-1);

    [theta_t_opti,last_epoch_t,L_t,Acc_t] = stochGradDesc(theta_init,X,Y,max_epoch,eta_0,eta_1,delta,batch_size,num_c);

    Y_test = predictLabel(theta_t_opti,X_test,num_c);

    
    %write to csv
    csv_filename = strcat('predictions-',num2str(acc),'-',datestr(now, 'yy-mm-dd-HH-MM-SS'),'.csv');
    cellData = [ID_X_test;num2cell(Y_test)];
    cellData = cellData';
    T = cell2table(cellData,'VariableNames',{'Id' 'Category'});
    writetable(T,strcat('~/Desktop/',csv_filename));
    
end

function q1_3_4(X_t,Y_t,X_v,Y_v)

    delta = 0.00001;
    eta_1 = 1;
    eta_0 = 0.1;
    max_epoch = 1000;
    batch_size = 16;
    num_c = 4;
    
    theta_init  = zeros(size(X_t,1),num_c-1);
% theta_init = rand(size(X_t,1),num_c-1);

    [theta_t_opti,last_epoch_t,L_t,Acc_t,theta_all_cell] = stochGradDesc(theta_init,X_t,Y_t,max_epoch,eta_0,eta_1,delta,batch_size,num_c);
        
    fprintf('2.3.1 \n Num_epochs = %i \n Final Loss = %i \n', last_epoch_t, L_t(end));
    
    %Plot L_t vs epoch
    figure
    plot(L_t);
    legend('L(\theta)');
    xlabel('epoch');  
    
    L_v = [];
    Acc_v = [];
    
    for i = 1:size(theta_all_cell,2)
        theta_curr = theta_all_cell{i};
        L_v(end+1) = getLoss(theta_curr,X_v,Y_v,num_c);
        acc = getAccuracy(Y_v,predictLabel(theta_curr,X_v,num_c));
        Acc_v(end+1) = acc;
    end
    
    fprintf('2.3.3 \n Num_epochs = %i \n Final Loss = %i \n', last_epoch_t, L_v(end));

    
    
    %Plot L vs epoch
    figure
    plot(L_t);
    hold on;
    plot(L_v);
    legend('L_{train}','L_{val}');
    xlabel('epoch');
    hold off;
    
    %Plot Acc vs epoch
    figure
    plot(Acc_t);
    hold on;
    plot(Acc_v);
    legend('Acc_{train}','Acc_{val}');
    xlabel('epoch');
    hold off;
    figure;
    conf_t = confusionchart(Y_t,predictLabel(theta_curr,X_t,num_c),'FontSize',20);
    conf_t.Title = 'Training Data Confusion Matrix';
    figure;
    conf_v = confusionchart(Y_v,predictLabel(theta_curr,X_v,num_c),'FontSize',20);
    conf_v.Title = 'Val Data Confusion Matrix';
    

end


function q2(X_t,Y_t)
%3,0.1
    delta = 0.00001;
    eta_0 = 3;
    eta_1 = 0.1;
    max_epoch = 1000;
    batch_size = 16;
    num_c = 4;
    
    theta_init  = zeros(size(X_t,1),num_c-1);

    [theta_opti,last_epoch,L] = stochGradDesc(theta_init,X_t,Y_t,max_epoch,eta_0,eta_1,delta,batch_size,num_c);
    
    fprintf('2.3.2 \n Num_epochs = %i \n Final Loss = %i \n', last_epoch, L(end));
    
    %Plot L vs epoch
    figure
    plot(L);
    legend('L(\theta)');
    xlabel('epoch');
    

end



% 
% function q1(X_t,Y_t)
% 
%     delta = 0.00001;
%     eta_1 = 1;
%     eta_0 = 0.1;
%     max_epoch = 1000;
%     batch_size = 16;
%     num_c = 4;
%     
%     theta_init  = zeros(size(X_t,1),num_c-1);
% 
%     [theta_opti,last_epoch,L] = stochGradDesc(theta_init,X_t,Y_t,max_epoch,eta_0,eta_1,delta,batch_size,num_c);
%     
%     fprintf('2.3.1 \n Num_epochs = %i \n Final Loss = %i \n', last_epoch, L(end));
%     
%     %Plot L vs epoch
%     figure
%     plot(L);
%     legend('L(\theta)');
%     xlabel('epoch');   
%     
% 
% end

function acc = getAccuracy(Y_p,Y)
    Foo = Y_p == Y;
    acc = sum(Y_p == Y)/size(Y,2);
end



function Y_p = predictLabel(theta,X_b,num_c)
    P = zeros(num_c,size(X_b,2));
    
    for ci = 1:num_c
       C = ci*ones(1,size(X_b,2));
       p = getProb(theta,X_b,C,num_c);
       P(ci,:) = p;    
    end
%     sum(P,1)
    [max_p,Y_p] = max(P);

end

function[X,ID_X,Y,ID_Y] = loadData(Xfile,Yfile)

    [X,ID_X] = loadXData(Xfile);
    [Y,ID_Y] = loadYData(Yfile);
    
    %match Ids
    [p,q] = ismember(ID_Y,ID_X);
    ID_X = ID_X(q);
    X = X(:,q);
end

function [X_t,ID] = loadXData(filename)
    fid = fopen(filename,'r');
    X_t = [];
    ID = [];
    while ~feof(fid)
        tline = fgetl(fid);
        %use strplit to split the line, first cell contains the id, rest 512 cells contain the 512-d feature
        C = strsplit(tline);
        id = C(1,1);
        x = str2num(strjoin(C(1,2:end)));
        
        ID = [ID;id];
        X_t = [X_t;x];
        %disp(tline) 
    end
    X_t = X_t';
    ID = ID';
    
    fclose(fid);
    
end

function [Y,ID] = loadYData(filename)

    data = readtable(filename);
    Y = data{:,2};
    ID = data{:,1};

    Y = Y';
    ID = ID';
end

%Normalise, Pad and add extra features
function X =  nPAddFeat(X)

%     X = normalize(X,2);
%     X = [X;mean(X);mean(X.^2)]; 0.48
%     

%     X = [X;max(X);sum(X);sum(X.^2)]; %0.485
    X = normalize(X,2);
    
    X = [ X;ones(1,size(X,2)) ];    
end




%C is row vec
function P = getProb(theta,X_b,C,num_c)

    Dr = 1 + sum(exp(theta'*X_b));
    
    [k,n] = size(X_b);
    k=k-1;
    
    P = zeros(1,n);   
    
%     Nr_1 = C == num_c-1;
%     
%     theta = [theta,zeros(size(X_b,1),1)];
%     Nr_2 = exp(sum(theta(:,C).*X_b)).*(C ~= num_c-1);
%     
%     Nr = Nr_1 + Nr_2;
%     
%     P = Nr./Dr;
    
    
    
    for i = 1:n
        curr_class_label = C(i);
        if C(i) == num_c
           p = 1/Dr(i);
        else
            foo = theta(:,curr_class_label)'*X_b(:,i);
            p = exp(foo)/Dr(i);
        end
        P(i) = p;
    end
end


function L = getLoss(theta,X_b,Y,num_c)
    [k,n] = size(X_b);
    k = k-1; %exclude the bias term
    
    P = getProb(theta,X_b,Y,num_c);
    L = (-1/n)*sum(log(P));
end

function pd = getPartialDiffSum(theta,X_b,Y,c,num_c)

    [k,n] = size(X_b);
    k = k-1;

    term_1 = Y == c;
    
    term_2 = getProb(theta,X_b,c*ones(size(Y)),num_c);
    
    pd = repmat( term_1 - term_2 ,size(X_b,1),1).* X_b; %repmat done to multiple factor with all elem of X^i
    pd = (-1/n)*sum(pd,2);
    
    
end

function theta_new = gradDesc(theta,X_b,Y,eta,num_c)
    
    [k,n] = size(X_b);
    k = k-1;

    %partial diff%
    %TODO%
    PD = zeros(size(theta));
    
    %foreach theta_c
    for c = 1:num_c-1
        pd = getPartialDiffSum(theta,X_b,Y,c,num_c);
        PD(:,c) = pd;
    end
    
    theta_new = theta - eta*PD;
end


function [theta_grad,epoch,L,Acc,theta_all_cell] = stochGradDesc(theta,X_b,Y,max_epoch,eta_0,eta_1,delta,batch_size,num_c)
    
    [k,n] = size(X_b);
    k = k-1; %remove bias term
    
    %theta_all = [];
    L = [];
    Acc = [];

    theta_all_cell = {};
    
    
    for epoch = 1:max_epoch

        
        eta = eta_0/(eta_1 + epoch);
        if epoch == 1
            old_loss = getLoss(theta,X_b,Y,num_c);
        else
            old_loss = new_loss;
        end
        %disp(old_loss);

        index_perm = randperm(n);
%         index_perm = 1:n;
        
        %batching        
        num_batches = floor(n/batch_size);
        last_batch_size = mod(n,batch_size);
        if last_batch_size == 0
            last_batch_template = [];
        else
            last_batch_template = ones(1,last_batch_size);
        end
        batches = mat2cell(index_perm,1,[ batch_size*ones(1,num_batches),last_batch_template]);
        
        for bi = 1:size(batches,2)
            %fprintf("Running for epoch=%i batch_num=%i\n",epoch,bi);
            
            B = cell2mat(batches(1,bi));
            %update theta for these batch-corresponding values
            theta = gradDesc(theta,X_b(:,B),Y(:,B),eta,num_c);
        end
        
%         calc accuracy
        Y_p = predictLabel(theta,X_b,num_c);
        acc = getAccuracy(Y_p,Y);
        Acc = [Acc,acc];        
        
        new_loss = getLoss(theta,X_b,Y,num_c);
        L = [L;new_loss];
        
        theta_all_cell{end+1} = theta;


        if ( (new_loss / old_loss) > 1 - delta && new_loss < old_loss)
            break;
        end
    end
    
    theta_grad = theta;
    
end