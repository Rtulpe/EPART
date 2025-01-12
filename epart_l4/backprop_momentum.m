function [hidlw, outlw, terr] = backprop_momentum(tset, tslb, inihidlw, inioutlw, lr, momentum)
% derivative of sigmoid activation function
% tset - training set (every row represents a sample)
% tslb - column vector of labels 
% inihidlw - initial hidden layer weight matrix
% inioutlw - initial output layer weight matrix
% lr - learning rate
% momentum - momentum factor

% hidlw - hidden layer weight matrix
% outlw - output layer weight matrix
% terr - total squared error of the ANN

% 1. Set output matrices to initial values
    hidlw = inihidlw;
    outlw = inioutlw;

% 2. Set total error to 0
    terr = 0;

% Initialize momentum values
    delta_hidl_prev = zeros(rows(inihidlw), columns(inihidlw));
    delta_outl_prev = zeros(rows(inioutlw), columns(inioutlw));

% for each sample in the training set
    for i = 1:rows(tset)

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
        hiddelta = (outdelta * outlw(1:end-1, :)') .* actdf(hidden);

        % 8. Compute gradients
        delta_outl = lr * [hidden 1]' * outdelta;
        delta_hidl = lr * [tset(i, :) 1]' * hiddelta;

        % 9. Update output layer weights with momentum
        outlw += delta_outl + momentum * delta_outl_prev;

        % 10. Update hidden layer weights with momentum
        hidlw += delta_hidl + momentum * delta_hidl_prev;

        % 11. Update momentum values
        delta_outl_prev = delta_outl;
        delta_hidl_prev = delta_hidl;
    end
end
