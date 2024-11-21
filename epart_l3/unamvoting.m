function clab = unamvoting(tset, clsmx)
% Simple unanimity voting function 
% 	tset - matrix containing test data; one row represents one sample
% 	clsmx - voting committee matrix
% Output:
%	clab - classification result 

	% class processing
	if clsmx(1, 2) == -1
		labels = clsmx(:, 1);
	else
		labels = unique(clsmx(:, [1 2]));
	endif
	
	reject = max(labels) + 1;

	% cast votes of classifiers
	votes = voting(tset, clsmx);

	if clsmx(1,2) == -1
		sumvotes = 1; % unanimity voting in one vs. rest scheme
		[mv clab] = max(votes, [], 2);

		% reject decision 
		clab(sum(votes, 2) ~= sumvotes) = reject;
	else
		maxvotes = rows(labels) - 1; % unanimity voting in one vs. one scheme

		[mv clab] = max(votes, [], 2);

		% reject decision 
		clab(mv ~= maxvotes) = reject;
	endif

end
