function [errcfs] = votePlr(clslabels, tstl)
% simple plurality classifier - class with greates number of votes wins (if unique)
% clslabels - matrix containing elementary classifiers results
%	each row contains NN labels for one test element
% tstl - test set labels (ground truth)
% errcf - error coefficients of the classifier

	clsCount = columns(clslabels);
	treshold = floor(clsCount/2) + 1;
	reject = max(tstl) + 1;

	[lab f] = mode(clslabels');
	for i=1:size(clslabels,1)
		if (f(i) < treshold)
			res = clslabels(i,:);
			res(res == lab(i)) = [];
			[l2 f2] = mode(res);
			if f2 == f(i)
				lab(i) = reject;
			end
		end
	end

	confmx = confMx(tstl, lab);
	errcfs = compErrors(confmx);
