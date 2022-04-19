
clear;close all;clc;
javaaddpath('\weka.jar');
T=50;
% Load dataset
load('diabetes.mat');
samples_size = length(Y);
temptest = zeros(T,floor(samples_size/5));

indices = K_Fold(samples_size,5);
ACC = zeros(5,1);

for fold=1:5
    fold
    % set train and test data with this 5Fold
    Train = [X(indices(:,1:fold-1),:) ; X(indices(:,fold+1:end),:)];
    Train_labels = [Y(indices(:,1:fold-1),:) ; Y(indices(:,fold+1:end),:)];
    Test = X(indices(:,fold),:);
    Test_labels = Y(indices(:,fold),:);
    Train = [Train Train_labels];
    Test = [Test Test_labels];
    FirstTrain = Train;

for i = 1 : T
    Train = FirstTrain;
    classifiers = {};
    indices2 = randi(4*floor(samples_size/5),4*floor(samples_size/5),1);
    Train = Train(indices2,:);
    
    %Save train and test data
    save train.txt Train -ascii
    save test.txt Test -ascii
    ArffTrain = convertToArff('train.txt');
    ArffTest = convertToArff('test.txt');
    
    % Train a J48 classifier
    classifier = weka.classifiers.trees.J48();
    classifier.buildClassifier(ArffTrain);
    classifiers{i} = classifier;
    
    % Classify test instances
    numInst = ArffTest.numInstances();
    for k=1:numInst
        
        temp = classifiers{i}.classifyInstance(ArffTest.instance(k-1));
        estimatedTestLabels(k,1) = str2num(char(ArffTest.classAttribute().value((temp)))); % Predicted labels
        
        
    end
    temptest(i,:) = estimatedTestLabels';
    
end
est_labesl = mode(temptest,1);
est_labesl = est_labesl';
% Compute accuracy of each fold
testLabels = Test(:,end);
ACC(fold) = (sum(est_labesl == testLabels) / length(testLabels)) * 100;

end
mean(ACC)
std(ACC)
