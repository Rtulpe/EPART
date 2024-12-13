% mainscript is rather short this time

% primary component count
comp_count = 40; 

[tvec tlab tstv tstl] = readSets(); 

% shift labels by one to use labels directly as indices
tlab += 1;
tstl += 1;

% check number of samples in each class
% labels = unique(tlab)';
% [labels; sum(tlab == labels); sum(tstl == labels)]

% RT: Add to report
%      1      2      3      4      5      6      7      8      9     10
%   5923   6742   5958   6131   5842   5421   5918   6265   5851   5949
%    980   1135   1032   1010    982    892    958   1028    974   1009

% compute and perform PCA transformation
[mu trmx] = prepTransform(tvec, comp_count);
tvec = pcaTransform(tvec, mu, trmx);
tstv = pcaTransform(tstv, mu, trmx);

% Now experiment with the learning rate
% It make sense to use "easy" (0 vs. 1) and "difficult" (4 vs. 9) cases.
%
%
% YOUR CODE GOES HERE - experiments with learning rates in the perceptron function
allzeros = tvec(tlab == 1, :);
allones = tvec(tlab == 2, :);

allfours = tvec(tlab == 5, :);
allnines = tvec(tlab == 10, :);

rep_cnt = 5;
lr_result = zeros(rep_cnt, 4);

for i = 1: rep_cnt
  [sp mispos misneg] = perceptron(allzeros, allones);
  lr_result(i, 1) = (mispos + misneg) / (rows(allzeros) + rows(allones));

  [sp mispos misneg] = perceptron(allfours, allnines);
  lr_result(i, 2) = (mispos + misneg) / (rows(allfours) + rows(allnines));

  [sp mispos misneg] = perceptron_fixed(allzeros, allones);
  lr_result(i, 3) = (mispos + misneg) / (rows(allzeros) + rows(allones));

  [sp mispos misneg] = perceptron_fixed(allfours, allnines);
  lr_result(i, 4) = (mispos + misneg) / (rows(allfours) + rows(allnines));
end

%lr_result =
%   4.5006e-03   4.1218e-02   1.5002e-03   4.0285e-02
%   4.7375e-03   4.2320e-02   1.7371e-03   4.0539e-02
%   4.7375e-03   4.1896e-02   1.5792e-03   4.1218e-02
%   4.5795e-03   4.1727e-02   1.5002e-03   4.1981e-02
%   2.1319e-03   4.2320e-02   4.7375e-04   4.0539e-02

% I will use fixed lr perceptron

% training of the whole ensemble
[ovo ovo40errors] = trainOVOensemble(tvec, tlab, @perceptron_fixed);
save ovo40errors.txt ovo40errors

% check your ensemble on train set
disp('OVO40 Train')
clab = unamvoting(tvec, ovo);
cfmx = confMx(tlab, clab)
compErrors(cfmx)

% Training   0.914700   0.055683   0.029617

% repeat on test set
disp('OVO40 Test')
clab = unamvoting(tstv, ovo);
cfmx = confMx(tstl, clab)
compErrors(cfmx)

% Test  0.918000   0.055500   0.026500

%
% YOUR CODE GOES HERE

% Train and test the OVR ensemble
[ovr ovr40errors] = trainOVRensemble(tvec, tlab, @perceptron_fixed);
save ovr40errors.txt ovr40errors

disp('OVR40 Train')
clab = unamvoting(tvec, ovr);
cfmx = confMx(tlab, clab)
compErrors(cfmx)

% ovr training set cfs              %testing set cfs
% 0.733967   0.044417   0.221617    0.739200   0.042500   0.218300

disp('OVR40 Test')
clab = unamvoting(tstv, ovr);
cfmx = confMx(tstl, clab)
compErrors(cfmx)

% expand features
trainExp = expandFeatures(tvec);
testExp = expandFeatures(tstv);

% Train and test the OVO ensemble on the expanded features
if exist('ovoExp.txt', 'file') == 0
  [ovoExp ovo860errors] = trainOVOensemble(trainExp, tlab, @perceptron_fixed);
  save ovo860errors.txt ovo860errors
  save ovoExp.txt ovoExp
end
load ovoExp.txt;
load ovo860errors.txt;

% train
disp('OVO860 Train')
clab = unamvoting(trainExp, ovoExp);
cfmx = confMx(tlab, clab)
compErrors(cfmx)
% test
disp('OVO860 Test')
clab = unamvoting(testExp, ovoExp);
cfmx = confMx(tstl, clab)
compErrors(cfmx)

%    ovoExp training set cfs                %ovoExp testing set cfs
%    9.9662e-01   2.2667e-03   1.1167e-03


% Train and test the OVR ensemble on the expanded features
if exist('ovrExp.txt', 'file') == 0
  [ovrExp ovr860errors] = trainOVRensemble(trainExp, tlab, @perceptron_fixed);
  save ovr860errors.txt ovr860errors
  save ovrExp.txt ovrExp
end
load ovrExp.txt;
load ovr860errors.txt;

% train
disp('OVR860 Train')
clab = unamvoting(trainExp, ovrExp);
cfmx = confMx(tlab, clab)
compErrors(cfmx)
% test
disp('OVR860 Test')
clab = unamvoting(testExp, ovrExp);
cfmx = confMx(tstl, clab)
compErrors(cfmx)

%    ovrExp training set cfs                %ovrExp testing set cfs
%    9.6613e-01   5.1667e-03   2.8700e-02   9.4390e-01   9.6000e-03   4.6500e-02

% Think about improving your classifier further :)
% RT: Possible improvement using LDA instead of PCA
tveclda = ldaTransform(tvec, tlab, mu, comp_count);
tstvlda = ldaTransform(tstv, tstl, mu, comp_count);

disp('LDA OVO40 Train')
clab = unamvoting(tveclda, ovo);
cfmx = confMx(tlablda, clab)
compErrors(cfmx)