function ldaSet = ldaTransform(tvec, tlab, mu, comp_count)
    % tvec - matrix containing vectors to be transformed
    % tlab - labels of the samples
    % mu - mean value of the training set
    % comp_count - count of LDA components in the final space
    % ldaSet - output set transformed to LDA space
    
    % Number of samples and features
    [num_samples, num_features] = size(tvec);
    classes = unique(tlab);
    num_classes = length(classes);
    
    % Initialize scatter matrices
    S_W = zeros(num_features, num_features);
    S_B = zeros(num_features, num_features);

    % Calculate S_W
    for c = 1:num_classes
        class_samples = tvec(tlab == classes(c), :);
        class_mean = mean(class_samples);
        S_W = S_W + (class_samples - class_mean)' * (class_samples - class_mean);
    end

    % Calculate S_B
    for c = 1:num_classes
        class_samples = tvec(tlab == classes(c), :);
        class_mean = mean(class_samples);
        n_c = size(class_samples, 1);
        mean_diff = class_mean - mu;
        S_B = S_B + n_c * (mean_diff' * mean_diff);
    end

    % Get eigenvectors and eigenvalues
    [V, D] = eig(pinv(S_W) * S_B);

    % Sort eigenvalues and eigenvectors
    [eigvals, indices] = sort(diag(D), 'descend');
    V_sorted = V(:, indices);
    
    % Select the top 'comp_count' eigenvectors (dimensions to retain)
    ldaMatrix = V_sorted(:, 1:comp_count);
    
    % Transform the data to the LDA
    ldaSet = tvec * ldaMatrix;
end
