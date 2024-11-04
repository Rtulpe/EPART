% RT: Function to check 1NN classification quality
function quality = check_1nn(train, test)
	quality = 0;
	for testid = 1:rows(test)
		test_instance = test(testid, 2:end);

		predicted = cls1nn(test_instance, train);

		if predicted == test(testid, 1)
			quality += 1;
		end
	end

	quality = quality / rows(test);
endfunction