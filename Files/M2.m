clear;clc;
load('glass.mat');
% N = length(X); % X training labels
% W = 1/N * ones(N,1); %Weights initialization
% M = 10; % Number of boosting iterations
% 
% for m=1:M
%     C = 10; 
%     
%     %cmd = ['-c ', num2str(C), ' -w ', num2str(W')];
%     model = fitcsvm(X, Y,'Weights',W);
%     [Xout, acc, ~] = predict(model,X);
%     
%     err = sum(.5 * W .* acc * N)/sum(W);
%     alpha = log( (1-err)/err );
%     
%     % update the weight
%     W = W.*exp( - alpha.*Xout.*X );
%     W = W/norm(W);
%     
% end

%mdl = fitensemble(X, Y,'Bag',100,'Tree','Type','Classification');
mdl = fitensemble(X, Y, 'AdaBoostM2', 50, 'Tree');
class = predict(mdl,X);

(sum(class == Y) / length(class)) * 100
