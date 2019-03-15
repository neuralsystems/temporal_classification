function cumulativeBinAccuracy = multiClass_classification_accuracy(spikeCountMatArrayTest, spikeCountMatArrayTrain)
% cumulativeBinAccuracy = multiClass_classification_accuracy(spikeCountMatArrayTest, spikeCountMatArrayTrain)
% Performs a supervised classification based on template matching (using euclidean distances)
% spikeCountMatArrayTest and spikeCountMatArrayTrain are cell arrays of length numClasses (i.e. number of clusters).
% Each element of the arrays is spikeBinMatrix, with numTrials rows and numBins columns.
% spikeCountMatArrayTrain is optional. If it is not provided, half of the trials in spikeCountMatArrayTest are used for training 
% and the remaining half for testing
%The input spikeCountMatArrayTest is a cell-array of matrices, with one matrix for each stimulus (so if you have 3 odors, the length of the cellarray would be 3). If you have 10 trials per odor, and if you are using 5 time-bins, then each matrix will be of size 10x5, with element (i,j) of the matrix containing the number of spikes in the j'th bin of the i'th trial.
% Reference: Gupta and Stopfer, Current Biology, 2014

numClasses = length(spikeCountMatArrayTest);
for c = 1:numClasses
	tmpNumTrials(c) = size(spikeCountMatArrayTest{c},1);
end	
numTrials = min(tmpNumTrials);

if (nargin ==2)  % if separate training matrices are provided, use them 
	numClassestmp = length(spikeCountMatArrayTrain);
	if numClassestmp ~= numClasses
		error('number of classes in spikeCountMatArrayTrain and spikeCountMatArrayTest is not equal');
	end
	cumulativeBinAccuracy = computeTemplateAccuracyHalf(spikeCountMatArrayTrain, spikeCountMatArrayTest, numTrials);
	
else % if separate training matrices are not provided, use half of the test trials for training and half for testing
	halfN = floor(numTrials/2);	
	
	trainArray = {}; testArray = {};
	for c = 1:numClasses
		trainArray{c} = spikeCountMatArrayTest{c}(1:halfN,:);
		testArray{c} = spikeCountMatArrayTest{c}(halfN+1:numTrials,:);
	end	
	cumulativeBinAccuracy_A = computeTemplateAccuracyHalf(trainArray, testArray, numTrials-halfN);

	trainArray = {}; testArray = {};
	for c = 1:numClasses
		testArray{c} = spikeCountMatArrayTest{c}(1:halfN,:);
		trainArray{c} = spikeCountMatArrayTest{c}(halfN+1:numTrials,:);
	end	
	cumulativeBinAccuracy_B = computeTemplateAccuracyHalf(trainArray, testArray, halfN);

	cumulativeBinAccuracy = mean([cumulativeBinAccuracy_A; cumulativeBinAccuracy_B ],1);		
end

end


%% trainsMatArray and testMatsArray are cell arrays of length numClasses (i.e. number of clusters).
% Each element of the arrays is spikeBinMatrix, with numTrials rows and numBins columns. 
% numTestTrials is the actual number of test trials that will be used (useful if there are different number of test trials)
function cumuBinAccu = computeTemplateAccuracyHalf(trainMatsArray, testMatsArray, numTestTrials)
	numClasses = length(trainMatsArray);
	for c = 1:numClasses
		centers{c} = mean(trainMatsArray{c},1);
	end	
	numBins = size(trainMatsArray{1},2);		
	
	cumuBinAccuPerTestClass = zeros(numClasses,numBins);	
	for testClass = 1:numClasses
		cumuBinAccuPerTrial = zeros(numTestTrials,numBins);	
		for trial = 1:numTestTrials  % loop over every test trial
			testVector = testMatsArray{testClass}(trial,:);

			% compute accuracy for length number of bins used
			for j = 1: numBins		
				distance = zeros(1,numClasses);
				for c = 1:numClasses
					distance(c) = pdist([testVector(1:j); centers{c}(1:j)]); % pdist computes the euclidean distance
				end
				if min(distance) == distance(testClass) % this means classification can work. now find out if there are other classes that are equally close
					countEquallyClose = 0;
					for tmpc = 1:numClasses
						if distance(tmpc) == distance(testClass)
							countEquallyClose = countEquallyClose+1;
						end
					end
					accu = 1/countEquallyClose;  % if this test vector is equally close to 3 templates including the correct one, classification accuracy for this would be 1/3
				else  % this means classification would be poor
					accu = 0; 
				end				
				cumuBinAccuPerTrial(trial,j) = accu;
			end					
		end
		cumuBinAccuPerTestClass(testClass,:) = mean(cumuBinAccuPerTrial,1);
	end
	cumuBinAccu = mean(cumuBinAccuPerTestClass,1);
end

