% tiny data file to verify pdf functions
load pdf_test.txt
size(pdf_test)

% how many classes are there?
labels = unique(pdf_test(:,1))

% how many samples are in each class?
[labels'; sum(pdf_test(:,1) == labels')]
		  % ^^^ how does this expression work?

% what's the layout of the samples?
% will it work?
plot2features(pdf_test, 2, 3);

% check if statistics package is present
normpdf(0, 0, 1)
% it can work directly - nothing to be done
% it can be installed but not loaded - pkg load statistics
% it can be not installed at all - use __normpdf function provided instead


pdfindep_para = para_indep(pdf_test)
% para_indep indep is already implemented; it should give:

% pdfindep_para =
%  scalar structure containing the fields:
%    labels =
%       1
%       2
%    mu =
%       0.7970000   0.8200000
%      -0.0090000   0.0270000
%    sig =
%       0.21772   0.19172
%       0.19087   0.27179

% now you have to implement pdf_indep and then verify it

pi_pdf = pdf_indep(pdf_test([2 7 12 17],2:end), pdfindep_para)

%pi_pdf =
%  1.4700e+000  4.5476e-007
%  3.4621e+000  4.9711e-005
%  6.7800e-011  2.7920e-001
%  5.6610e-008  1.8097e+000

% multivariate normal distribution - parameters ...

pdfmulti_para = para_multi(pdf_test)

%pdfmulti_para =
%  scalar structure containing the fields:
%    labels =
%       1
%       2
%    mu =
%       0.7970000   0.8200000
%      -0.0090000   0.0270000
%    sig =
%    ans(:,:,1) =
%       0.047401   0.018222
%       0.018222   0.036756
%    ans(:,:,2) =
%       0.036432  -0.033186
%      -0.033186   0.073868

% ... and probability density function (use mvnpdf in pdf_multi)

pm_pdf = pdf_multi(pdf_test([2 7 12 17],2:end), pdfmulti_para)

%pm_pdf =
%  7.9450e-001  6.5308e-017
%  3.9535e+000  3.8239e-013
%  1.6357e-009  8.6220e-001
%  4.5833e-006  2.8928e+000

% parameters for Parzen window approximation
pdfparzen_para = para_parzen(pdf_test, 0.5)
									 % ^^^ window width

%pdfparzen_para =
%  scalar structure containing the fields:
%    labels =
%       1
%       2
%    samples =
%    {
%      [1,1] =
%         1.10000   0.95000
%         0.98000   0.61000
% .....
%         0.69000   0.93000
%         0.79000   1.01000
%      [2,1] =
%        -0.010000   0.380000
%         0.250000  -0.440000
% .....
%        -0.110000   0.030000
%         0.120000  -0.090000
%    }
%    parzenw =  0.50000

% now you have to implement pdf_parzen and then verify it

pp_pdf = pdf_parzen(pdf_test([2 7 12 17],2:end), pdfparzen_para)

%pp_pdf =
%  9.7779e-001  6.1499e-008
%  2.1351e+000  4.2542e-006
%  9.4059e-010  9.8823e-001
%  2.0439e-006  1.9815e+000


% now you can start work with cards!
[train test] = load_cardsuits_data();

% Our first look at the data
size(train)
size(test)
labels = unique(train(:,1))
unique(test(:,1))
[labels'; sum(train(:,1) == labels')]

% Point 1
%
% the first task after loading the data is checking
% training set for outliers; to this end we usually compute
% simple statistics: mean, median, std,
% and/or plot histogram of individual feature: hist
% and/or plot two features at a time: plot2features

[mean(train); median(train)]
hist(train(:,1));
plot2features(train, 4, 6);
					%^^^^ just an example

% to identify outliers you can use two output argument versions
% of min and max functions

[mv midx] = min(train);

% because the minimum or maximum values can be determined always,
% it's worth to look at neighbors of the suspected sample in the training set

% RT: Disabling the following code, it offsets the actual results
% /////////////////////////////////////////////////
% let's assume that sample 41 is suspected
%midx = 41
%train(midx-1:midx+1, :)
% it seems that these three rows are very similar to each other...
% that's because 41 is evidently not an outlier index

% if you're sure the midx sample should be removed:
%size(train)
%train(midx, :) = [];
%size(train)
% /////////////////////////////////////////////////

% the procedure of searching for and removing outliers must be repeated
% until no outliers exist in the training set

% RT: MY CODE, Finding and removing outliers
% First, plotting 2 features at a time to find outliers



% First, checking with min() and max() functions to find outliers
[minv_tr, midx_tr] = min(train)
[minv_ts, midx_ts] = min(test)
[maxv_tr, midx_tr] = max(train)
[maxv_ts, midx_ts] = max(test)

% Indexes that are outliers for both sets:
% 642, 186
% Removing them and checking again
train(642, :) = [];
train(186, :) = [];
test(642, :) = [];
test(186, :) = [];

[minv_tr, midx_tr] = min(train)
[minv_ts, midx_ts] = min(test)
[maxv_tr, midx_tr] = max(train)
[maxv_ts, midx_ts] = max(test)

% RT: Seems to be clean now

% after removing outliers, you can deal with the selection of TWO features for classification
% in this case, it is enough to look at the graphs of two features and choose the ones that
% give relatively well separated classes

% RT: Plotting all possible combinations of 2 features to find the best pair:
for i = 2:columns(test)
	for j = i+1:columns(test)
		name = sprintf("Features %d and %d", i, j);
		plot2features(test, i, j, name);
	end
end

% RT: After examination, I have selected 2 and 4 as the best features for classification

% after selecting features reduce both sets:
train = train(:, [1 2 4]);
test = test(:, [1 2 4]);

% POINT 2

pdfindep_para = para_indep(train);
pdfmulti_para = para_multi(train);
pdfparzen_para = para_parzen(train, 0.001);
% this window width should be included in your report!

% Point 2 results
base_ercf = zeros(1,3);
base_ercf(1) = mean(bayescls(test(:,2:end), @pdf_indep, pdfindep_para) != test(:,1));
base_ercf(2) = mean(bayescls(test(:,2:end), @pdf_multi, pdfmulti_para) != test(:,1));
base_ercf(3) = mean(bayescls(test(:,2:end), @pdf_parzen, pdfparzen_para) != test(:,1));
base_ercf

% before moving to point 3 it would be wise to
% implement and test reduce function
% let's start with small test set - just 2 classes

rdlab = unique(pdf_test(:,1));
reduced = reduce(pdf_test, [0.8 0.4]);
[rdlab'; sum(reduced(:,1) == rdlab')]

% ans =
%     1    2
%     8    4


% POINT 3

% In the next point, the reduce function will be useful, which reduces the number of samples
% in the individual classes (in this case, the reduction will be the same in all classes -
% OF THE TRAINING SET)
% Because reduce has to draw samples randomly, the experiment should be repeated 5 times
% In the report, please provide only the mean value and the standard deviation
% of the error coefficient

parts = [0.1 0.25 0.5];
rep_cnt = 5; % at least

% YOUR CODE GOES HERE
%

mean_ercf = zeros(length(parts), 3);
std_ercf = zeros(length(parts), 3);
local_ercf = zeros(3, rep_cnt);

for partid = 1:columns(parts)
	part_local = parts(partid);
	for rep = 1:rep_cnt
		red_train = reduce(train, part_local * ones(1, 8));
		pdfindep_para = para_indep(red_train);
		pdfmulti_para = para_multi(red_train);
		pdfparzen_para = para_parzen(red_train, 0.001);

		local_ercf(1, rep) = mean(bayescls(test(:,2:end), @pdf_indep, pdfindep_para) != test(:,1));
		local_ercf(2, rep) = mean(bayescls(test(:,2:end), @pdf_multi, pdfmulti_para) != test(:,1));
		local_ercf(3, rep) = mean(bayescls(test(:,2:end), @pdf_parzen, pdfparzen_para) != test(:,1));
	end

	mean_ercf(partid, :) = mean(local_ercf, 2)';
	std_ercf(partid, :) = std(local_ercf, 0, 2)';
end

mean_ercf
std_ercf

% note that for given experiment you should reduce all classes in the training
% set with the same reduction coefficient; assuming that class_count is the
% number of different classes in the training set you can take 3/4 random samples
% of each class with:
% 	reduced_train = reduce(train, 0.75 * ones(1, class_count))
%

% POINT 4
% Point 4 concerns only Parzen window classifier (on the full training set)

parzen_widths = [0.0001, 0.0005, 0.001, 0.005, 0.01];
parzen_res = zeros(1, columns(parzen_widths));

% YOUR CODE GOES HERE
%

[parzen_widths; parzen_res]
% Plots are sometimes better than numerical results
semilogx(parzen_widths, parzen_res)

% POINT 5
% In point 5 you should reduce TEST SET
%

apriori = [0.165 0.085 0.085 0.165 0.165 0.085 0.085 0.165];
parts = [1.0 0.5 0.5 1.0 1.0 0.5 0.5 1.0];

% YOUR CODE GOES HERE
%


% POINT 6
% In point 6 we should consider data normalization

std(train(:,2:end))

% Should we normalize?
% If YES remember to normalize BOTH training and testing sets

% YOUR CODE GOES HERE
%
