clear;close all;clc;
javaaddpath('\weka.jar');
T=30;
% Load bupa dataset
load('diabetes.mat');

uniq_label = unique(Y);
samples_size = length(Y);
fold_size = 5;
indices = K_Fold(samples_size,fold_size);
ACC = zeros(fold_size,1);

for fold=1:fold_size
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
    weigths = ones(4*floor(samples_size/fold_size) , 1);
    weigths = weigths * (1/(4*floor(samples_size/fold_size)));
    
    classifiers = {};
    Pt = weigths / (sum(weigths));
    indices2 = randsample(4*floor(samples_size/5),4*floor(samples_size/5),true,Pt);
    T=30;
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
        e1 = sum(Pt(find(estimatedTrainLabels ~= pre_Train(:,end))))
        if(e1 > 0.5)
            T = i-1;
            break;
        end
        Beta(i) = e1 / (1 - e1);
        new_weigths = weigths .* (power(Beta(i) , (1 - (pre_Train(:,end) ~= estimatedTrainLabels))));
        Pt = new_weigths / (sum(new_weigths));
        indices2 = randsample(4*floor(samples_size/5),4*floor(samples_size/5),true,Pt);
        weigths = new_weigths;
    end
    % Compute accuracy of each fold
    sum_vector =zeros(size(Test,1),length(uniq_label));
    for i = 1: T
        numInst = ArffTest.numInstances();
        for k=1:numInst
            
            temp = classifiers{i}.classifyInstance(ArffTest.instance(k-1));
            estimatedTestLabels(k,i) = str2num(char(ArffTest.classAttribute().value((temp)))); % Predicted Test labels
            
        end
        for j = 1: length(uniq_label)
            tmp = estimatedTestLabels(:,i) == (ones(size(Test,1),1) * uniq_label(j)) ;
            sum_vector(:,j) = sum_vector(:,j) + (log(1/Beta(i)) * tmp );% * estimatedTestLabels(:,i));
        end
    end
    testLabels = Test(:,end);
    est_labesl = zeros(size(Test,1),1);
    [~,ind]=max(sum_vector,[],2);
    for j = 1:size(Test,1)
        est_labesl(j) = uniq_label(ind(j));
    end
    ACC(fold) = (sum(est_labesl == testLabels) / length(testLabels)) * 100;
    
end

mean(ACC)
std(ACC)
