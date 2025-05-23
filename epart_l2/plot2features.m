function plot2features(tset, f1, f2, name="")
% Plots tset samples on a 2-dimensional diagram
%	using features f1 and f2
% tset - training set; the first column contains class label
% f1 - index of the first feature (mapped to horizontal axis)
% f2 - index of the second feature (mapped to vertical axis)

	% plotting parameters for different classes
	%   restriction to 8 classes seems reasonable
	pattern(1,:) = "ks";
	pattern(2,:) = "rd";
	pattern(3,:) = "mv";
	pattern(4,:) = "b^";
	pattern(5,:) = "kd";
	pattern(6,:) = "r^";
	pattern(7,:) = "mo";
	pattern(8,:) = "bd";

	res = tset(:, [f1, f2]);

	% extraction of all unique labels used in tset
	labels = unique(tset(:,1));
	if size(labels, 1) > 8
		labels = labels(1:8);
	end

	% create diagram and switch to content preserving mode
	figure;
	hold on;

  	% RT: MY CODE
  	% I have added some legend, to make more sense of what is going on
  	legendEntries = cell(size(labels, 1), 1);

	for i=1:size(labels,1)
		idx = tset(:,1) == labels(i);
		plot(res(idx,1), res(idx,2), pattern(i,:));

    % Store the label text for the legend
    legendEntries{i} = ['Class ', num2str(labels(i))];
	end

  	% Add the legend using the stored legend entries
    legend(legendEntries, 'Location', 'northeast');
	% RT: Adding title to the plot, if provided
	if ~isempty(name)
        title(name);
    end

	hold off;
end
