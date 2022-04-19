clear;close all;clc;

data = load('bupa.data');
X = data(:,1:end-1);
Y = data(:,end);
save('bupa.mat')
