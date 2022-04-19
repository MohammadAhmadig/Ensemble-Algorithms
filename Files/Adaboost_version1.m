
clear;close all;clc;
javaaddpath('\weka.jar');
T=30;
% Load data set
load('sonar.mat');

% tempLabel = find(Y == 2);
% Y(tempLabel)= -1;
% tempLabel2 = find(Y == 4);
% Y(tempLabel2)= 1;

samples_size = length(Y);
temptest = zeros(T,floor(samples_size/5));
indices = K_Fold(samples_size,5);
ACC = zeros(5,1);

for fold=1:5
    fold
    % set train and test data with this fiveFold
    Train = [X(indices(:,1:fold-1),:) ; X(indices(:,fold+1:end),:)];
    Train_labels = [Y(indices(:,1:fold-1),:) ; Y(indices(:,fold+1:end),:)];
    Test = X(indices(:,fold),:);
    Test_labels = Y(indices(:,fold),:);
    Train = [Train Train_labels];
    Test = [Test Test_labels];
    pre_Train = Train;
    
    save pre_train.txt Train -ascii
    pre_ArffTrain = convertToArff('pre_train.txt');
    
    weigths = ones(4*floor(samples_size/5) , 1);
    weigths = weigths * (1/(4*floor(samples_size/5)));
    
    classifiers = {};
    indices2 = randsample(4*floor(samples_size/5),4*floor(samples_size/5),true,weigths);
    for i = 1 : T
        
        Train = Train(indices2,:);
        
        %Save train and test data
        save train.txt Train -ascii
        save test.txt Test -ascii
        ArffTrain = convertToArff('train.txt');
        ArffTest = convertToArff('test.txt');
        % Train a DecisionStump classifier
        classifier = weka.classifiers.trees.DecisionStump();
        classifier.buildClassifier(ArffTrain);
        classifiers{i} = classifier;
        
        % Classify train instances
        numInst = pre_ArffTrain.numInstances();
        for k=1:numInst
            
            temp = classifiers{i}.classifyInstance(pre_ArffTrain.instance(k-1));
            estimatedTrainLabels(k,1) = str2num(char(pre_ArffTrain.classAttribute().value((temp)))); % Predicted pre_train labels
            
        end
        e1 = sum(weigths(find(estimatedTrainLabels ~= pre_Train(:,end))));
        alpha(i) = (1/2) * log((1-e1)/e1);
        new_weigths = weigths .* exp(-alpha(i) * (pre_Train(:,end).* estimatedTrainLabels));
        %new_weigths = real(new_weigths);
        
        indices2 = randsample(4*floor(samples_size/5),4*floor(samples_size/5),true,new_weigths);
        weigths = new_weigths;
    end
    
    % Compute accuracy of each fold
    sum_vector =zeros(size(Test,1),1);
    for i = 1: T
        numInst = ArffTest.numInstances();
        for k=1:numInst
            
            temp = classifiers{i}.classifyInstance(ArffTest.instance(k-1));
            estimatedTestLabels(k,i) = str2num(char(ArffTest.classAttribute().value((temp)))); % Predicted Test labels
            
            
        end
        sum_vector = sum_vector + (alpha(i) * estimatedTestLabels(:,i));
    end
    testLabels = Test(:,end);
    est_labesl = sign(sum_vector);
    ACC(fold) = (sum(est_labesl == testLabels) / length(testLabels)) * 100;
    
end

mean(ACC)
std(ACC)
