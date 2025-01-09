vnames = {
'valid_1.txt',
'valid_2.txt',
'valid_3.txt',
'valid_4.txt',
'valid_5.txt',
'valid_6.txt',
'valid_7.txt'};

tnames = {
'test_1.txt',
'test_2.txt',
'test_3.txt',
'test_4.txt',
'test_5.txt',
'test_6.txt',
'test_7.txt'};

vtab = loadCNNOutputs(vnames);
load validlab.txt
validlab = validlab + 1;

ttab = loadCNNOutputs(tnames);
load testlab.txt
testlab = testlab + 1;

iqltytest = [];
iqltyvald = [];
for i=1:columns(vtab)
	iqltyvald = [iqltyvald mean(vtab(:,i)!=validlab)];
	iqltytest = [iqltytest mean(ttab(:,i)!=testlab)];
end
[1-iqltyvald; 1-iqltytest]


vecUni = voteUni(vtab, validlab);
vecMaj = voteMaj(vtab, validlab);
vecPlr = votePlr(vtab, validlab);

tecUni = voteUni(ttab, testlab);
tecMaj = voteMaj(ttab, testlab);
tecPlr = votePlr(ttab, testlab);

resVald = [vecUni; vecMaj; vecPlr];
resTest = [tecUni; tecMaj; tecPlr];

resVald = [resVald fobj(resVald)]
resTest = [resTest fobj(resTest)]
