%% Battery voltage prediction across diverse fast-charging protocols over entire lifespan
% B: Modeling
% @ZHANG Shuzhi **, CHEN Shouxuan, GAO Xiang, DING Run, XI Yuhang, CAO Ganglin, ZHANG Xiongwen 
% ** - Corresponding Author
% 
% This program aims to address voltage prediction during various multistage constant current (MCC) fast-charging protocols
% Use voltage at step k-1 to predict voltage at step k
% This code is the second part of the whole program
% Base-error joint modeling offline
% Gaussian process regression (GPR) for base modeling
% Regression tree (RT) for error modeling
% Training data of base model: the first-cycle data of the first cell corresponding to 3.7C(31%)-5.9C
% Training data of error model: several typical-cycle data of the first cell of all MCC fast-charging protocols

clc;clear;
rng(1)


%% Base modeling: the first-cycle data of the first cell corresponding to 3.7C(31%)-5.9C
% for simplification, we only retain the first CC stage and constant voltage (CV) stage
% the rest part is reconstructed using linear function
load('.\splitedData_A1\conditionNames.mat')
baseCondition =conditionNames{1,1};
fname = strcat('.\splitedData_A1\',baseCondition,'.mat');
load(fname)
baseDataTmp = data;
batteryGroupTmp=fieldnames(baseDataTmp);
batteryGroupTmp(1:4,:)=[];
data_base = baseDataTmp.(batteryGroupTmp{1,1}).(strcat('cycle', num2str(1)));
cc1_end = ischange(data_base.I,'mean');
cc1_end_tmp = find(cc1_end(10:end)==1,1);
cc1_end = 9+cc1_end_tmp-1;
cv_start=find(data_base.I<=1.5 & data_base.V >= 3.6,1);
t_division = [data_base.t(cc1_end), data_base.t(cv_start)]; % record the end(start) time of the first cc stage (cv stage)
V_division = [data_base.V(cc1_end), data_base.V(cv_start)]; % record the end(start) voltage of the first CC stage (CV stage)
linear_fit = polyfit(t_division, V_division, 1); % curve fit
V_fit = polyval(linear_fit, 302:1:1124); % voltage reconstruction
V_recons = [data_base.V(1:cc1_end); V_fit'; data_base.V(cv_start:end)]; % obtain the whole reconstructed voltage
V_real = data_base.V;
window_size = 1; % window size for voltage prediction, here it is 1 (can also be set to 10, 50, etc)
L = size(V_recons, 1); % data length
V_input_base = []; % input data for base modeling
V_output_base = []; % output data for base modeling
for i = 1:L-window_size
    V_input_base = [V_input_base; V_recons(i:i + window_size - 1)];
    V_output_base = [V_output_base; V_recons(i + window_size)];
end
save('Voltage_comparison.mat', "V_recons", "V_real")

% Base modeling via GPR
t_base = tic;
Mdl_base = fitrgp(V_input_base, V_output_base, 'KernelFunction', 'exponential');
t_base = toc(t_base);
save('Mdl_base.mat', 'Mdl_base','t_base'); % save base model

% verification on training data
trainResults_base = V_input_base(1:window_size); % the first window_size voltage is available
while size(trainResults_base, 1) < L % base voltage prediction
    trainResults_base = [trainResults_base; ...
        predict(Mdl_base, trainResults_base(end - window_size + 1: end))];
end
trainResults_base = trainResults_base(window_size + 1: end); % delete the first window_size voltage
save('BaseMdl_trainingResult.mat', "V_output_base", "trainResults_base")

% visualization
figure
subplot(211)
plot(V_output_base, 'DisplayName', 'Real')
hold on
plot(trainResults_base, '-.', 'DisplayName', 'trainResult')
hold off
legend('location', 'southeast')
subplot(212)
plot(V_output_base - trainResults_base, '.')


%% Generate training data for error modeling: several typical-cycle data of the first cell of all MCC fast-charging protocols
Error_input = []; % input data of Error model: base voltage, capacity_max, typical parameters of MCC fast-charging protocols
Error_output = []; % output data of Error model: prediction error of base voltage
Vreal = []; % real voltage

% several typical-cycle data of the first cell of 3.7C(31%)-5.9C
for condition = 1:numel(conditionNames)
    currentCondition = conditionNames{condition,1};
    fname = strcat('.\splitedData_A1\',currentCondition,'.mat');
    load(fname)
    clear fname
    conditionData = data;
    batteryGroup = fieldnames(conditionData);
    batteryGroup(1:4,:) = [];
    batteryName = batteryGroup{1,1};
    currentBatteryData=conditionData.(batteryName);

    fprintf('%s\t\t--%s\n',currentCondition,batteryName)
    if isnan(currentBatteryData.cycleLife)
        currentBatteryData.cycleLife =(numel(fieldnames(currentBatteryData))-1);
    end
    for cycleNum=1:100:currentBatteryData.cycleLife-1
        cycleName = strcat('cycle',num2str(cycleNum));
        data_error = currentBatteryData.(cycleName).V;
        L = length(data_error);%data length
        V_real = data_error(1 + window_size : end); % partial real voltage (exclude the first window_size voltage)
        Vreal =[Vreal;V_real];
        V_base = data_error(1:window_size);% the first window size voltage is available
        while size(V_base,1)<L% base voltage prediction
            V_base = [V_base; predict(Mdl_base, V_base(end - window_size + 1: end))];
        end
        V_base = V_base(window_size + 1: end);
        error = V_real - V_base;
        plot(V_base, error, 'DisplayName', num2str(cycleNum)); % visualization
        hold on
        legend('-DynamicLegend')
        Qmax = currentBatteryData.(cycleName).Qc(end); % record capacity_max
        Error_input = [Error_input; [V_base, Qmax*ones(L-window_size, 1), conditionData.rate1*ones(L-window_size, 1),...
            conditionData.socChange*ones(L-window_size, 1), conditionData.rate2*ones(L-window_size, 1)]]; % record input data (3.7,
        Error_output = [Error_output; error]; % record output data

        errrr.(currentCondition).(batteryName).(cycleName)(:, 1) = V_real;
        errrr.(currentCondition).(batteryName).(cycleName)(:, 2) = V_base;
        errrr.(currentCondition).(batteryName).(cycleName)(:, 3) = error;

    end
end
save('BaseMdl_result.mat', 'errrr')

%% Error modeling via RT
t_error = tic;
Mdl_error = fitrtree(Error_input, Error_output);
t_error = toc(t_error);
save('Mdl_error.mat', 'Mdl_error'); % save error model

% verification on training data
trainResult_error = predict(Mdl_error, Error_input);
% visualization
figure
subplot(211)
plot(Error_output, 'DisplayName', 'Real')
hold on
plot(trainResult_error, '-.', 'DisplayName', 'trainResult')
hold off
legend('location', 'southeast')
subplot(212)
plot(Error_output - trainResult_error, '. ')


%% The final verification on training data (base + error)
for condition =1:numel(conditionNames)
    currentCondition=conditionNames{condition,1};
    fname = strcat('.\splitedData_A1\',currentCondition,'.mat');
    load(fname)
    clear fname
    conditionData =data;
    batteryGroup=fieldnames(conditionData);
    batteryGroup(1:4,:)=[];
    batteryName = batteryGroup{1,1};
    fprintf('%s\n\t\t',currentCondition, batteryName)
    currentBatteryData=conditionData.(batteryName);
    RMSE = []; % record RMSE
    MAPE = []; % record MAPE
    if isnan(currentBatteryData.cycleLife)
        currentBatteryData.cycleLife =(numel(fieldnames(currentBatteryData))-1);
    end
    for cycleNum=1:100:currentBatteryData.cycleLife-1
        cycleName = strcat('cycle',num2str(cycleNum));
        currentData = currentBatteryData.(cycleName);
        data_error = currentData.V;
        L = length(data_error);%data length
        V_real = data_error(1 + window_size : end); % partial real voltage (exclude the first window_size voltage)
        V_base = data_error(1:window_size); % the first window_size voltage is available

        V_pre = V_base; % initialize predicted voltage
        while size(V_base, 1) < L
            V_base = [V_base; predict(Mdl_base, V_base(end - window_size+1: end)')]; % predict base voltage
        end
        V_base  = V_base(window_size+1: end);
        Qmax = currentBatteryData.(cycleName).Qc(end);

        Error_input = [V_base, Qmax*ones(L-window_size, 1), conditionData.rate1*ones(L-window_size, 1), ...
            conditionData.socChange*ones(L-window_size, 1), conditionData.rate2*ones(L-window_size, 1)]; % record input data (3.7, 31 & 5.9 are protocol parameters)
        trainResult_error = predict(Mdl_error, Error_input);

        V_pre = V_base+trainResult_error; % delete the first window_size voltage

        trainingResult.(currentCondition).(batteryName).(cycleName)(:, 1) = V_pre;
        trainingResult.(currentCondition).(batteryName).(cycleName)(:, 2) = V_real;
        trainingResult.(currentCondition).(batteryName).(cycleName)(:, 3) = trainResult_error;

        rmse = sqrt(mean((V_real - V_pre).^2));
        mape = mean(abs(V_real - V_pre) ./ V_real * 100);% calculate MAPE
        RMSE = [RMSE; rmse];
        MAPE = [MAPE; mape];
    end
    trainingResult.(currentCondition).(batteryName).RMSE = RMSE;
    trainingResult.(currentCondition).(batteryName).MAPE = MAPE;

end
save('trainingResult.mat', 'trainingResult'); % save prediction results

%% Verification on test data: other cells of all MCC fast-charging protocols
% save real voltage, predicted voltage, RMSE and MAPE
Results = [];
for condition =1:numel(conditionNames)
    currentCondition=conditionNames{condition,1};
    fname = strcat('.\splitedData_A1\',currentCondition,'.mat');
    load(fname)
    clear fname
    conditionData =data;
    batteryGroup=fieldnames(conditionData);
    batteryGroup(1:4,:)=[];
    for battery = 2:numel(batteryGroup)% exclude the first cell
        batteryName = batteryGroup{battery,1};
        fprintf('%s\n\t\t', batteryName)
        currentBatteryData=conditionData.(batteryName);
        c = 1;
        if isnan(currentBatteryData.cycleLife)
            currentBatteryData.cycleLife =(numel(fieldnames(currentBatteryData))-1);
        end
        C = zeros(currentBatteryData.cycleLife-1,1); % record cycle index
        Qreal = zeros(currentBatteryData.cycleLife-1,1); % record real capacity_max
        Qest = zeros(currentBatteryData.cycleLife-1,1); % record estimated capacity_max (assume there is ±2% relative error)
        RE = zeros(currentBatteryData.cycleLife-1,1); % record the relative error between real and estimated capacity_max
        RMSE = zeros(currentBatteryData.cycleLife-1,1); % record RMSE
        MAPE = zeros(currentBatteryData.cycleLife-1,1); % record MAPE
        
        cycleNameGroup = fieldnames(currentBatteryData);
        for cycleNum=1:currentBatteryData.cycleLife-1
            cycleName = strcat('cycle',num2str(cycleNum));
            if isempty(find(strcmp(cycleName, cycleNameGroup), 1))
                continue
            end
            currentData = currentBatteryData.(cycleName);
            data_cycle = currentData.V;
            L = length(data_cycle);%data length
            V_real = data_cycle(1 + window_size : end); % partial real voltage (exclude the first window_size voltage)
            c = cycleNum;
            q_real = currentData.Qc(end);% real capacity_max
            q_est = q_real * (1+(rand*4-2)/100);% estimated capacity_max
            re = (q_est-q_real)/q_real*100; % the relative error
            C(c) = cycleNum;
            Qreal(c) = q_real;
            Qest(c) = q_est;
            RE(c) = re;
            V_base = zeros(L, 1); % initialize base voltage
            V_base(1:window_size) = data_cycle(1:window_size); % the first window_size voltage is available
            V_pre = V_base; % initialize predicted voltage
            for l = window_size + 1 : L
                V_base(l) = predict(Mdl_base, V_base(l - window_size: l - 1)'); % predict base voltage
                V_error = predict(Mdl_error, [V_base(l), q_est, conditionData.rate1,...
                    conditionData.socChange, conditionData.rate2]); % predict error between real and base voltage
                V_pre(l) = V_base(l) + V_error; % calculate final prediction of real voltage
            end
            V_pre = V_pre(window_size + 1 : end); % delete the first window_size voltage
            rmse = sqrt(mean((V_real - V_pre).^2));
            mape = mean(abs(V_real - V_pre) ./ V_real * 100);% calculate MAPE
            RMSE(c) = rmse;
            MAPE(c) = mape;
            c = c+1;
            Results.(currentCondition).(batteryName).(cycleName) = [V_real, V_pre]; % save real voltage and predicted voltage
        end
        Results.(currentCondition).(batteryName).Qmax = [C Qreal Qest RE];
        Results.(currentCondition).(batteryName).Statistics = [C RMSE MAPE];
    end
end
save('Final_Results.mat', 'Results'); % save prediction results
%% Battery voltage prediction across diverse fast-charging protocols over entire lifespan
