% RT: Function to normalize data, based on standardization
function norm_data = normalize_data(data)
	norm_data = data;
	for i = 2:columns(data)
		norm_data(:, i) = (data(:, i) - mean(data(:, i))) / std(data(:, i));
	end
end