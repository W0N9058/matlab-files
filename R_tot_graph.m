% 데이터 불러오기
param_5degC = load('param_2RC_5degC.mat');
param_15degC = load('param_2RC_15degC.mat');
param_25degC = load('param_2RC_25degC.mat');
param_35degC = load('param_2RC_35degC.mat');
param_45degC = load('param_2RC_45degC.mat');

% 그래프 크기 지정
figure('Position', [100 100 800 500]);
hold on;

% 온도와 데이터 이름 지정
temps = [5 15 25 35 45];
param_data = {param_5degC, param_15degC, param_25degC, param_35degC, param_45degC};

% 온도별 색상 지정
colors = {'b', 'r', 'g', 'm', 'k'};

% 온도 별로 그리기
for i = 1:length(temps)
    % 지금 온도 데이터 받아오기
    current_param = param_data{i};
    
    % R_tot 계산하기 (R_tot = R0 + R1 + R2)
    R_tot = current_param.R0 + current_param.R1 + current_param.R2;
    
    % R_tot 단위 mΩ으로 바꾸기
    R_tot = R_tot * 1000

    % SOC 정렬
    [SOC_sorted, sort_idx] = sort(current_param.SOC_LUT);
    % R_tot 정렬
    R_tot_sorted = R_tot(sort_idx);
    
    % 그래프 점 찍기
    plot(SOC_sorted, R_tot_sorted, [colors{i} '-o'], ...
        'LineWidth', 1.5, ...
        'MarkerSize', 6, ...
        'DisplayName', [num2str(temps(i)) '°C']);
end

% 그래프 표시 조절
grid on;
xlabel('SOC (%)', 'FontSize', 12);
ylabel('R_{tot} (mΩ)', 'FontSize', 12);
title('Total Resistance vs. State of Charge at Different Temperatures', 'FontSize', 14);
legend('Location', 'best');
set(gca, 'FontSize', 11);
box on;
xlim([0 100]);