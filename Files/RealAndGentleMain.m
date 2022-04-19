clc;clear;
load('breast-cancer-wisconsin.mat');
X = X';
Y = Y';
%Y = Y*2 - 1;
tempLabel = find(Y == 2);
Y(tempLabel)= 1;
tempLabel2 = find(Y == 4);
Y(tempLabel2)= -1;

MaxIter = 200; % boosting iterations

TrainData   = X(:,1:2:end);
TrainLabels = Y(1:2:end);
ControlData   = X(:,2:2:end);
ControlLabels = Y(2:2:end);

% initializing matrices for storing step error
RAB_control_error = zeros(1, MaxIter);
MAB_control_error = zeros(1, MaxIter);
GAB_control_error = zeros(1, MaxIter);

% constructing weak learner
weak_learner = tree_node_w(3); % pass the number of tree splits to the constructor

%initializing learners and weights matices
GLearners = [];
GWeights = [];
RLearners = [];
RWeights = [];
NuLearners = [];
NuWeights = [];

%iterativly running the training
for i = 1 : MaxIter
    clc;
    MaxIter
    i
    %training gentle adaboost
    [GLearners GWeights] = GentleAdaBoost(weak_learner, TrainData, TrainLabels, 1, GWeights, GLearners);
    %evaluating control error
    GControl = sign(Classify(GLearners, GWeights, ControlData));
    GAB_control_error(i) = GAB_control_error(i) + sum(GControl ~= ControlLabels) / length(ControlLabels);

    %training real adaboost
    [RLearners RWeights] = RealAdaBoost(weak_learner, TrainData, TrainLabels, 1, RWeights, RLearners);
    %evaluating control error
    RControl = sign(Classify(RLearners, RWeights, ControlData));
    RAB_control_error(i) = RAB_control_error(i) + sum(RControl ~= ControlLabels) / length(ControlLabels);

end

% Step4: displaying graphs
figure, plot(GAB_control_error);
hold on;
plot(RAB_control_error, 'g');
hold off;

legend('Gentle AdaBoost', 'Real AdaBoost');
xlabel('Iterations');
ylabel('Test Error');
(sum(ControlLabels == RControl) / length(ControlLabels)) * 100
(sum(ControlLabels == GControl) / length(ControlLabels)) * 100
