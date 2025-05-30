function [hl ol] = crann(cfeat, chn, cclass)
% generates hidden and output ANN weight matrices
% cfeat - number of features 
% chn - number of neurons in the hidden layer
% cclass - number of neurons in the outpur layer (= number of classes)

% hl - hidden layer weight matrix
% ol - output layer weight matrix

% ACHTUNG: we assume that constant value IS NOT INCLUDED in the feature count

	hl = (rand(cfeat + 1, chn) - 0.5) / sqrt(cfeat + 1);
	ol = (rand(chn + 1, cclass) - 0.5) / sqrt(chn + 1);
