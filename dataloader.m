% MAT 파일 로드
% load("battery_properties.csv");
% load("dUdT(entropy coefficient).csv")
load('param_2RC_5degC.mat');
load('param_2RC_15degC.mat');
load('param_2RC_25degC.mat');
load('param_2RC_35degC.mat');
load('param_2RC_45degC.mat');
% load("Capacity.txt")

%%
param_5degC = load('param_2RC_5degC.mat');
param_15degC = load('param_2RC_15degC.mat');
param_25degC = load('param_2RC_25degC.mat');
param_35degC = load('param_2RC_35degC.mat');
param_45degC = load('param_2RC_45degC.mat');


%%
% Bus 요소 정의
elems(1) = Simulink.BusElement;
elems(1).Name = 'Em';
elems(1).Dimensions = size(Em);

elems(2) = Simulink.BusElement;
elems(2).Name = 'R0';
elems(2).Dimensions = size(R0);

elems(3) = Simulink.BusElement;
elems(3).Name = 'R1';
elems(3).Dimensions = size(R1);

elems(4) = Simulink.BusElement;
elems(4).Name = 'tau1';
elems(4).Dimensions = size(tau1);

elems(5) = Simulink.BusElement;
elems(5).Name = 'SOC_Em_LUT';
elems(5).Dimensions = size(SOC_Em_LUT);

% Bus 객체 생성
BatteryBus = Simulink.Bus;
BatteryBus.Elements = elems;

% Bus 객체를 저장하여 Simulink에서 참조 가능하게 설정
assignin('base', 'BatteryBus', BatteryBus);