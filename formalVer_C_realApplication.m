%% Battery voltage prediction across diverse fast-charging protocols over entire lifespan
% C: Real application
% @ZHANG Shuzhi **, CHEN Shouxuan, GAO Xiang, DING Run, XI Yuhang, CAO Ganglin, ZHANG Xiongwen 
% ** - Corresponding Author
% 
% This program aims to address voltage prediction during various multistage constant current (MCC) fast-charging protocols
% Use voltage at step k-1 to predict voltage at step k
% This code is the last part of the whole program
% Base-error joint modeling online utilization
% We generate initial voltage of SOC=0 according to the probability density function (PDF) fitted by training data
% Use the generated initial voltage to accurately predict the whole voltage curve
% The predicted voltage is referred through the comparison between measured voltage sequence and the predicted whole voltage curve

clc;clear;
rng(1)


%% Obtain the PDF and cumulative distribution function (CDF) of the initial voltage of SOC=0 by training data
load('.\splitedData_A1\conditionNames.mat')
V = []; % record the initial voltage of SOC=0
Q = []; % record capacity max
C1 = []; % record fast-charging protocol parameter
Qtrans = []; % record fast-charging protocol parameter
C2 = []; % record fast-charging protocol parameter

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
    cycleNameGroup = fieldnames(currentBatteryData);

    for cycleNum=1:1:currentBatteryData.cycleLife-1
        cycleName = strcat('cycle',num2str(cycleNum));
        if isempty(find(strcmp(cycleName, cycleNameGroup), 1))
            continue
        end
        data_tmp = currentBatteryData.(cycleName);
        V = [V; data_tmp.V(1)]; % initial voltage of SOC=0
        Q = [Q; data_tmp.Qc(end)]; % capacity max
        C1 = [C1; conditionData.rate1]; % fast-charging protocol parameter
        Qtrans = [Qtrans; conditionData.socChange]; % fast-charging protocol parameter
        C2 = [C2; conditionData.rate2]; % fast-charging protocol parameter
    end

end

correlation = [corr(Q, V), corr(C1, V), corr(Qtrans, V), corr(C2, V)]; %correlation analysis
[f, xi] = ksdensity(V); % estimate PDF of the initial voltage of SOC=0
cdf = cumsum(f) * (xi(2) - xi(1));
cdf = [0, cdf] / cdf(end); % calculate CDF
save('Voltage_distribution.mat', "cdf", "f", "xi","V")
%% Base-error joint modeling online utilization
% save real voltage, predicted voltage, RMSE and MAPE
load('Mdl_base.mat'); % load base model
load('Mdl_error.mat'); % load error model
Results = [];
for condition = 2:numel(conditionNames)
    currentCondition = conditionNames{condition,1};
    fname = strcat('.\splitedData_A1\',currentCondition,'.mat');
    load(fname)
    clear fname
    conditionData = data;
    batteryGroup = fieldnames(conditionData);
    batteryGroup(1:4,:) = [];
    for batteryNum = 2:size(batteryGroup,1)
        tic
        batteryName = batteryGroup{batteryNum,1};
        currentBatteryData=conditionData.(batteryName);
        fprintf('%s\t\t--%s\n',currentCondition,batteryName)
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
        c = 1;
        for cycleNum=1:1:currentBatteryData.cycleLife-1
            cycleName = strcat('cycle',num2str(cycleNum));
            if isempty(find(strcmp(cycleName, cycleNameGroup), 1))
                continue
            end
            data_tmp = currentBatteryData.(cycleName);
            V_real = data_tmp.V; % complete real voltage
            L = length(V_real); % data length
            q_real = data_tmp.Qc(end);
            q_est = q_real * (1 + (rand * 4 - 2)/100); % estimated capacity max
            re = (q_est - q_real)/q_real*100; % the relative error
            C(c) = cycleNum;
            Qreal(c) = q_real;
            Qest(c) = q_est;
            RE(c) = re;
            V_base = interp1(cdf, [2, xi], rand, 'linear', 'extrap'); % initialize voltage corresponding to SOC=0
            for i = 2:L % voltage prediction
                V_base(i) = predict(Mdl_base, V_base(i-1))  ;% predict base voltage
            end
            V_base = V_base';
            V_error = predict(Mdl_error, [V_base, ones(L, 1)*q_est, ones(L, 1)*conditionData.rate1, ...
                ones(L, 1)*conditionData.socChange, ones(L, 1)*conditionData.rate2]); % predict voltage error
            V_pre = V_base + V_error; % calculate the final prediction of voltage

            % only partial voltage sequence is available in real application
            % we assume the start point = 20%, 40% and 80% respectively
            % accumulate 10s data for comparison
            percentage = 0.05;

            start_point = floor(L*percentage);
            accumulate_length = 600;
            V_partial = V_real(start_point : start_point + accumulate_length - 1); % available partial voltage sequence
            comparison_error = []; % RMSE between voltage sequence and the predicted whole voltage curve
            diff_com = []; % difference in RMSE at adjacent sample
            for l = 1:L - accumulate_length + 1
                comparison_error = [comparison_error; ...
                    sqrt(mean((V_pre(l:l + accumulate_length - 1) - V_partial).^2))]; % calculate RMSE
            end
            [~, location] = min(comparison_error); % locate the most similar sequence in the predicted whole voltage curve
            Results.(currentCondition).(batteryName).(cycleName).Vcompare_pre = ...
                [(0:1:length(V_pre) - 1)', V_pre]; % record the predicted whole voltage curve
            Results.(currentCondition).(batteryName).(cycleName).Vcompare_partial = ...
                [(location:1:location + accumulate_length - 1)', V_partial];
            % the V_pre with index after the determined location is the predicted voltage
            % evaluate the prediction performance
            % notice that the comparison data length will be different
            if start_point >= location
                del_data_length = start_point - location;
                rmse = sqrt(mean((V_real(start_point + accumulate_length:end) - ...
                    V_pre(location + accumulate_length:end - del_data_length)).^2)); % calculate RMSE
                mape = mean(abs(V_real(start_point + accumulate_length:end) - ...
                    V_pre(location + accumulate_length:end - del_data_length)) ./ ...
                    V_real(start_point + accumulate_length:end) * 100); % calculate MAPE
                Results.(currentCondition).(batteryName).(cycleName).V_pre=...
                     [V_real(start_point + accumulate_length:end), V_pre(location + accumulate_length:end - del_data_length)];
            else
                del_data_length = location - start_point;
                rmse = sqrt(mean((V_real(start_point + accumulate_length:end - del_data_length) - ...
                    V_pre(location + accumulate_length:end)).^2)); % calculate RMSE
                mape = mean(abs(V_real(start_point + accumulate_length:end - del_data_length) - ...
                    V_pre(location + accumulate_length:end)) ./ ...
                    V_real(start_point + accumulate_length:end - del_data_length) * 100); % calculate MAPE
                Results.(currentCondition).(batteryName).(cycleName).V_pre=...
                     [V_real(start_point + accumulate_length:end - del_data_length), V_pre(location + accumulate_length:end)];
            end
            RMSE(c) = rmse;
            MAPE(c) = mape;
            c = c+1;
        end
        Results.(currentCondition).(batteryName).Qmax = [C Qreal Qest RE];
        Results.(currentCondition).(batteryName).Statistics = [C RMSE MAPE];
    end
end
save(strcat('Final_Results_Initialization4V_percentage_', num2str(100*percentage),'.mat'), 'Results');
%% Battery voltage prediction across diverse fast-charging protocols over entire lifespan
