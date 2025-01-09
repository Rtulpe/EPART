function [errcfs] = voteUni(clslabels, tstl)
% simple unanimity classifier
% clslabels - matrix containing elementary classifiers results
%	each row contains NN labels for one test element
% tstl - test set labels (ground truth)
% errcf - error coefficients of the classifier

	treshold = columns(clslabels);
	reject = max(tstl) + 1;
	[lab f] = mode(clslabels');
	lab(f < treshold) = reject;

	confmx = confMx(tstl, lab);
	errcfs = compErrors(confmx);
