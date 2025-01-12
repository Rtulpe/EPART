[tvec tlab tstv tstl] = readSets(); 
[unique(tlab)'; unique(tstl)']
tlab += 1;
tstl += 1;

[unique(tlab)'; sum(tlab == unique(tlab)')]
[unique(tstl)'; sum(tstl == unique(tstl)')]

% if you consider normalization of data 
% this is proper place to implement it

% reference network architecture & training parameters
noHiddenNeurons = 100;
noEpochs = 50;
learningRate = 0.005;

rand()

% saving state of (pseudo)random number generator
% loading state of (pseudo)random number generator
% RT: Minor change to not comment out the code :)
if exist('rndstate.txt ', 'file') == 0
  load rndstate.txt 
  rand("state", rndstate);
else
  rndstate = rand("state");
  save rndstate.txt rndstate
end

[hlnn olnn] = crann(columns(tvec), noHiddenNeurons, 10);
trainError = zeros(1, noEpochs);
testError = zeros(1, noEpochs);
trReport = [];
for epoch=1:noEpochs
	tic();
	[hlnn olnn terr] = backprop(tvec, tlab, hlnn, olnn, learningRate);
	clsRes = anncls(tvec, hlnn, olnn);
	cfmx = confMx(tlab, clsRes);
	errcf = compErrors(cfmx);
	trainError(epoch) = errcf(2);

	clsRes = anncls(tstv, hlnn, olnn);
	cfmx = confMx(tstl, clsRes);
	errcf2 = compErrors(cfmx);
	testError(epoch) = errcf2(2);
	epochTime = toc();
  %RT: Keep things short
  fprintf('Epoch: %d, Time: %.2f, Total Error: %.1f, Train Error: %.4f, Test Error: %.4f\n', epoch, epochTime, terr, trainError(epoch), testError(epoch));
	trReport = [trReport; epoch epochTime trainError(epoch) testError(epoch)];
	fflush(stdout);
end

% you'll be wise to paramterise this filename not to overwrite
% the reference result
save rep_h100_e50_lr0005.txt trReport 

plot(1:50, trainError, 'b', 1:50, testError, 'r')
xlabel('epoch');
ylabel('error');
title ("training and testing error during backprop");
legend ("train error", "test error");
set(findall(gcf,'-property','FontSize'),'FontSize',12);

    % 1.00000   30.67761    0.24450    0.25180
    % 2.00000   30.69140    0.18698    0.19960
    % 3.00000   29.72420    0.16920    0.18040
    % 4.00000   30.18758    0.15950    0.17210
    % 5.00000   30.83003    0.15330    0.16730
    % 6.00000   29.80827    0.14855    0.16370
    % 7.00000   30.14833    0.14465    0.16060
    % 8.00000   30.53052    0.14105    0.15740
    % 9.00000   30.27049    0.13807    0.15500
   % 10.00000   30.29409    0.13545    0.15190
   % 11.00000   30.55205    0.13315    0.14940
   % 12.00000   29.05374    0.13075    0.14740
   % 13.00000   30.06374    0.12840    0.14600
   % 14.00000   30.37319    0.12680    0.14480
   % 15.00000   29.43519    0.12515    0.14270
   % 16.00000   29.95211    0.12365    0.14240
   % 17.00000   30.37092    0.12255    0.14180
   % 18.00000   29.13097    0.12130    0.13950
   % 19.00000   30.35579    0.12017    0.13950
   % 20.00000   30.50302    0.11885    0.13780
   % 21.00000   29.32516    0.11757    0.13700
   % 22.00000   30.19691    0.11617    0.13550
   % 23.00000   30.75535    0.11545    0.13490
   % 24.00000   29.27134    0.11452    0.13420
   % 25.00000   30.04758    0.11352    0.13380
   % 26.00000   30.95289    0.11250    0.13340
   % 27.00000   30.46662    0.11183    0.13350
   % 28.00000   29.35743    0.11083    0.13260
   % 29.00000   29.42741    0.10990    0.13220
   % 30.00000   29.25062    0.10902    0.13130
   % 31.00000   29.44622    0.10815    0.13080
   % 32.00000   29.98084    0.10735    0.13020
   % 33.00000   29.62857    0.10663    0.12970
   % 34.00000   29.84644    0.10580    0.12910
   % 35.00000   29.68380    0.10498    0.12870
   % 36.00000   30.40955    0.10432    0.12860
   % 37.00000   30.35389    0.10360    0.12830
   % 38.00000   30.32788    0.10293    0.12810
   % 39.00000   30.45069    0.10208    0.12770
   % 40.00000   29.86320    0.10145    0.12710
   % 41.00000   30.44679    0.10092    0.12650
   % 42.00000   29.40610    0.10035    0.12600
   % 43.000000   30.466921    0.099750    0.126200
   % 44.000000   30.034881    0.099283    0.126200
   % 45.000000   29.259302    0.098717    0.126300
   % 46.000000   30.532743    0.098233    0.126200
   % 47.000000   29.432242    0.097667    0.126200
   % 48.000000   29.210704    0.097200    0.125900
   % 49.000000   29.294924    0.096750    0.126100
   % 50.000000   29.364464    0.096167    0.125800
