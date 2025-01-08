function [hidlw outlw terr] = backprop(tset, tslb, inihidlw, inioutlw, lr)
% derivative of sigmoid activation function
% tset - training set (every row represents a sample)
% tslb - column vector of labels 
% inihidlw - initial hidden layer weight matrix
% inioutlw - initial output layer weight matrix
% lr - learning rate

% hidlw - hidden layer weight matrix
% outlw - output layer weight matrix
% terr - total squared error of the ANN

% 1. Set output matrices to initial values
	hidlw = inihidlw;
	outlw = inioutlw;
	
% 2. Set total error to 0
	terr = 0;

% for each sample in the training set
	for i=1:rows(tset)

		% 3. Set desired output of the ANN (it depends on actf you use!)
		dout = -ones(1, columns(outlw));
		dout(tslb(i)) = 1;

		% 4. Propagate input forward through the ANN
		hidden = actf([tset(i, :) 1] * hidlw);
		output = actf([hidden 1] * outlw);

		% 5. Adjust total error
		terr += sumsq(dout - output);

		% 6. Compute delta error of the output layer
		outdelta = (dout - output) .* actdf(output);

		% 7. Compute delta error of the hidden layer
		hiddelta = (outdelta * outlw(1:end-1, :)').* actdf(hidden);

		% 8. Update output layer weights
		outlw += [hidden 1]' * outdelta * lr;

		% 9. Update hidden layer weights
		hidlw += [tset(i, :) 1]' * hiddelta * lr;
	end
