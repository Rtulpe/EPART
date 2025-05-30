function [ovosp sperrors] = trainOVOensemble(tset, tlab, htrain)
% Trains a set of linear classifiers (one versus one class)
% on the training set using trainSelect function
% tset - training set samples
% tlab - labels of the samples in the training set
% htrain - handle to proper function computing separating plane
% ovosp - one versus one class linear classifiers matrix
%   the first column contains positive class label
%   the second column contains negative class label
%   columns (3:end) contain separating plane coefficients
% sperrors - upper triangular matrix containing error coefficients of
% individual linear classifiers on the training set

  labels = unique(tlab);
  sperrors = zeros(rows(labels));
  % nchoosek produces all possible unique pairs of labels
  % that's exactly what we need for ovo classifier
  pairs = nchoosek(labels, 2);
  ovosp = zeros(rows(pairs), 2 + 1 + columns(tset));
  
  for i=1:rows(pairs)
	% store labels in the first two columns
    ovosp(i, 1:2) = pairs(i, :);
	
	% select samples of two digits from the training set
    posSamples = tset(tlab == pairs(i,1), :);
    negSamples = tset(tlab == pairs(i,2), :);
	
	% train 5 classifiers and select the best one
    [sp misp misn] = trainSelect(posSamples, negSamples, 5, htrain);
	
	% what to do with errors? - store in sperrros
    sperrors(pairs(i, 1), pairs(i, 2)) = (misp + misn) / (rows(posSamples) + rows(negSamples));
	% it would be wise to add additional output argument
	% to return error coefficients
	
    % store the separating plane coefficients (this is our classifier)
	% in ovo matrix
    ovosp(i, 3:end) = sp; 
  end
end
