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

		% 4. Propagate input forward through the ANN

		% 5. Adjust total error

		% 6. Compute delta error of the output layer

		% 7. Compute delta error of the hidden layer

		% 8. Update output layer weights

		% 9. Update hidden layer weights
	end
