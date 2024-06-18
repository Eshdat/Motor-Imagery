%Final project

close all
clear all
clc
%turn on display of new figures
set(0, 'DefaultFigureVisible', 'on');


load('motor_imagery_train_data.mat')
TrainingData = P_C_S.data;
[nTrials, nSample, nChanel]= size(TrainingData);
FS= P_C_S.samplingfrequency;  % sampling rate

BASELINE = 1;
DURATION = nSample / FS;
C3=1;
C4=2;

imaginStart = 2.25;
time_vec= 0: 1/FS : nSample/FS-1/FS;
imagnationIncides = imaginStart * FS: nSample;


imagiryDataC3 = TrainingData(:,imagnationIncides,C3);
imagiryDataC4 = TrainingData(:,imagnationIncides,C4);

right_incides = find(P_C_S.attribute(4,:) == 1);
right_trial_C3 = TrainingData(right_incides,:,C3);   
right_trial_C4 = TrainingData(right_incides,:,C4);   

left_incides = find(P_C_S.attribute(3,:) == 1);
left_trial_C3 = TrainingData(left_incides,:,C3);   
left_trial_C4 = TrainingData(left_incides,:,C4);    

baseTrialC3 = avg_across_baseline(TrainingData(:,:,C3), FS);
baseTrialC4 = avg_across_baseline(TrainingData(:,:,C4), FS);
% Fourier Trandform Parameters
windowSize = FS;
overLap =  floor(0.9 * windowSize); 
nfft = nSample; 

%% Visualization
trial2represent = 20;
random_indices= unique(randperm(nTrials/2,trial2represent));

figure('Color', 'w', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.7, 0.35], 'WindowState', 'maximized');
sgtitle('EEG signal on 20 different trials- right hand')
PlotRandomTrials(random_indices,time_vec, TrainingData, right_incides)

figure('Color', 'w', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.7, 0.35], 'WindowState', 'maximized');
sgtitle('EEG signal on 20 different trials- left hand')
PlotRandomTrials(random_indices,time_vec, TrainingData, left_incides)
%% Power Spectrom Welch method

[pwelchRightC4, f] = AvaragePwelch(right_trial_C4(:,imagnationIncides)',windowSize, overLap, nfft , FS);
[pwelchRightC3, ~] = AvaragePwelch(right_trial_C3(:,imagnationIncides)',windowSize, overLap, nfft , FS);
[pwelchLeftC4, ~] = AvaragePwelch(left_trial_C4(:,imagnationIncides)',windowSize, overLap, nfft , FS);
[pwelchLeftC3, ~] = AvaragePwelch(left_trial_C3(:,imagnationIncides)',windowSize, overLap, nfft , FS);



figure('Color', 'w', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.7, 0.35], 'WindowState', 'maximized');
subplot(2 ,1 ,1)

plot(f, pwelchLeftC3(:,1))
hold on
upper_bound = pwelchLeftC3(:, 1) + pwelchLeftC3(:, 2);
lower_bound = pwelchLeftC3(:, 1) - pwelchLeftC3(:, 2);
fill([f; flipud(f)], [lower_bound; flipud(upper_bound)], 'm', 'FaceAlpha', 0.3);
plot(f, pwelchRightC3(:,1))
upper_bound = pwelchRightC3(:, 1) + pwelchRightC3(:, 2);
lower_bound = pwelchRightC3(:, 1) - pwelchRightC3(:, 2);
fill([f; flipud(f)], [lower_bound; flipud(upper_bound)], 'g', 'FaceAlpha', 0.3);
title("C3")
legend("Left hand", 'std left hand', "Right hand", 'std right hand')
xlabel("frequency [Hz]")
ylabel("Power")


subplot(2 ,1 ,2)
plot(f, pwelchLeftC4(:,1))
hold on
upper_bound = pwelchLeftC4(:, 1) + pwelchLeftC4(:, 2);
lower_bound = pwelchLeftC4(:, 1) - pwelchLeftC4(:, 2);
fill([f; flipud(f)], [lower_bound; flipud(upper_bound)], 'm', 'FaceAlpha', 0.3);
plot(f, pwelchRightC4(:,1))
upper_bound = pwelchRightC4(:, 1) + pwelchRightC4(:, 2);
lower_bound = pwelchRightC4(:, 1) - pwelchRightC4(:, 2);
fill([f; flipud(f)], [lower_bound; flipud(upper_bound)], 'g', 'FaceAlpha', 0.3);
title("C4")
legend("Left hand", 'std left hand', "Right hand", 'std right hand')
xlabel("frequency [Hz]")
ylabel("Power")
title("C4")
sgtitle("Power Spectrum welch method")


%% powerspectrum with substruction of the Base line

f = 0:0.1:40;
%Average across trials  spectra.

avgRightC3 = squeeze(mean(right_trial_C3, 1));  
avgRightC4 = squeeze(mean(right_trial_C4, 1));
avgLeftC3 = squeeze(mean(left_trial_C3 , 1));
avgLeftC4 = squeeze(mean(left_trial_C4 , 1));

[powerC3Right, fspec, tspec] = spectrogram(avgRightC3, windowSize, overLap, f ,FS, 'yaxis');
powerC3Left = spectrogram(avgLeftC3, windowSize,overLap, f ,FS, 'yaxis');
powerC4Right = spectrogram(avgRightC4, windowSize, overLap, f,FS, 'yaxis');
powerC4Left = spectrogram(avgLeftC4, windowSize,overLap, f ,FS, 'yaxis');




%finding the index corresponding to 1 [sec] (rounded up)
baseLinePeriod = ceil((BASELINE/ DURATION) * length(tspec));

%avarging the base lineover the time domain 
meanBaseLinec3right = mean(powerC3Right(:,1:baseLinePeriod), 2);
meanBaseLinec4right = mean(powerC4Right(:,1:baseLinePeriod), 2);
meanBaseLinec3left = mean(powerC3Left(:,1:baseLinePeriod), 2);
meanBaseLinec4left = mean(powerC4Left(:,1:baseLinePeriod), 2);


powerC3Left = powerC3Left - meanBaseLinec3left;
powerC4Left = powerC4Left - meanBaseLinec4left;
powerC3Right = powerC3Right - meanBaseLinec3right;
powerC4Right = powerC4Right - meanBaseLinec4right;


% Convert to dB

powerC3RightdB = 10* log10(abs(powerC3Right));
powerC3LeftdB = 10* log10(abs(powerC3Left));
powerC4RightdB = 10* log10(abs(powerC4Right));
powerC4LeftdB = 10* log10(abs(powerC4Left));

% Plotting
figure('Color', 'w', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.7, 0.35], 'WindowState', 'maximized');

sgtitle("Spectogram with division by base line")
subplot(2,2,1)
plotSpectogram(tspec, fspec, powerC3RightdB, -15,15);
title('C3 - Right');

subplot(2,2,2)
plotSpectogram(tspec, fspec, powerC4RightdB,-15,15);

title('C4 - Right');

subplot(2,2,3)
plotSpectogram(tspec, fspec, powerC3LeftdB, -15,15);

title('C3 - Left');


subplot(2,2,4)
plotSpectogram(tspec, fspec, powerC4LeftdB,-15,15);
title('C4 - Left');

%% powerspectrum with substruction of the conditions

powerC3 = powerC3Right - powerC3Left;
powerC4 = powerC4Right - powerC4Left;

% powerC3 = convert2dB(powerC3, windowSize);
% powerC4 = convert2dB(powerC4, windowSize);
powerC3 = 10* log10(abs(powerC3));
powerC4 = 10* log10(abs(powerC4));
% Plotting
figure('Color', 'w', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.7, 0.35], 'WindowState', 'maximized');

sgtitle("Spectogram with substruction of right - left")
subplot(2,1,1)
plotSpectogram(tspec, fspec, powerC3, -10, 20);
title('C3');

subplot(2,1,2)
plotSpectogram(tspec, fspec, powerC4,-10,20);
title('C4')
%% features
features =  featuresExtraction(baseTrialC3, baseTrialC4, FS, ...
        right_incides, left_incides, imagnationIncides, nTrials, windowSize,overLap);
normalizedData = zscore(features');


%% pca 

pcaCoeff = pca(normalizedData);
pcaData = (normalizedData * pcaCoeff)';

figure('Color', 'w', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.7, 0.35], 'WindowState', 'maximized')
subplot(1,2,1)
scatter(pcaData(1,right_incides), pcaData(2,right_incides), 'filled','magenta')
hold on 
scatter(pcaData(1,left_incides), pcaData(2,left_incides),'filled', 'g')
legend('Right', 'Left')
title("Pca data 2D")
xlabel("Pca first component")
ylabel("Pca second component")

subplot(1,2,2)
scatter3(pcaData(1,right_incides), pcaData(2,right_incides), pcaData(3,right_incides),'filled','r')
hold on 
scatter3(pcaData(1,left_incides), pcaData(2,left_incides), pcaData(3,left_incides),'filled', 'b')
legend('Right', 'Left')
title("Pca data 3D")
xlabel("Pca first component")
ylabel("Pca second component")
zlabel("Pca third component")


%% NCA
labels = P_C_S.attribute(4, :);  % labels, 1 = right, 0 = left
NCA = fscnca(normalizedData,labels);

figure('Color', 'w', 'Units', 'normalized', 'Position', [0.05, 0.05, 0.7, 0.35], 'WindowState', 'maximized')
plot(NCA.FeatureWeights,'ro')
grid on
xlabel('Feature index')
ylabel('Feature weight')
title("Feature Importance Ranking using Neighborhood Components Analysis (NCA)")

% make an ordered deatures list
% Create a table with feature names and their corresponding scores
feature_scores = NCA.FeatureWeights;
feature_names = {'powerBandC3_1', 'powerBandC3_2', 'powerBandC3_3','powerBandC3_4', 'powerBandC3_5','powerBandC4_1', 'powerBandC4_2', 'powerBandC4_3', 'powerBandC4_4', 'powerBandC4_5','powerBandC4_6', 'covariance', 'spectral_entropy_c3', 'spectral_entropy_c4'};
count = (1:numel(feature_names))';
feature_table = table(count, feature_names', feature_scores, 'VariableNames', {'Num', 'Feature', 'Importance'});

% Sort the table in descending order of importance scores
sorted_table = sortrows(feature_table, 'Importance', 'descend');
sorted_table.Num = (1:numel(feature_names))';
disp(sorted_table);

%% Training the classifier
feature_ind = find(feature_scores>= 0.1);
num_featurs = 5;
best_features = normalizedData(:,feature_ind(1:num_featurs));

% Define the number of folds
k = 5;
labaled_features = horzcat(best_features, labels');

% Randomly shuffle the data (if not already shuffled)
random_indices = randperm(nTrials);
shuffled_labeled_features = labaled_features(random_indices, :);


% Split the data into k folds
cv = cvpartition(size(best_features, 1), 'KFold', k);

% Initialize arrays to store accuracy values
trainAccuracy = zeros(k, 1);
valAccuracy = zeros(k, 1);

% Perform k-fold cross-validation
for i = 1:k
    trainIndices = training(cv, i);
    valIndices = test(cv, i);
    % Split the data into training and validation sets
    trainData = shuffled_labeled_features(trainIndices, 1:end -1);
    trainLabels = shuffled_labeled_features(trainIndices, end);
    valData = shuffled_labeled_features(valIndices, 1:end -1);
    valLabels = shuffled_labeled_features(valIndices, end);

    % Train the LDA classifier
    [ldaClassifier, err] = classify(valData, trainData, trainLabels);

    
    % Evaluate the classifier on the training set
    trainAccuracy(i) = (1 - err) * 100;
    
    % Evaluate the classifier on the validation set
    valAccuracy(i) = sum(ldaClassifier == valLabels) / length(valLabels);
end

% Calculate average and standard deviation of accuracy
avgTrainAccuracy = mean(trainAccuracy);
stdTrainAccuracy = std(trainAccuracy);
avgValAccuracy = mean(valAccuracy) * 100;
stdValAccuracy = std(valAccuracy) * 100;

% Report the results
disp(['LDA Training Accuracy: ', num2str(avgTrainAccuracy), '±', num2str(stdTrainAccuracy), '%']);
disp(['LDA Validation Accuracy: ', num2str(avgValAccuracy), '±', num2str(stdValAccuracy), '%']);


%%
% Train and evaluate SVM with RBF kernel

svmClassifier = fitcsvm(trainData(:,1:num_featurs), trainLabels, 'KernelFunction', 'RBF', 'KernelScale','auto');
trainPredLabels = predict(svmClassifier, trainData(:,1:num_featurs));
trainAccuracy = sum(trainPredLabels == trainLabels) / numel(trainLabels) *100;
valPredLabels = predict(svmClassifier, valData(:,1:num_featurs));
valAccuracy = sum(valPredLabels == valLabels) / numel(valLabels) *100;
disp(['RBF Training Accuracy: ', num2str(trainAccuracy), '±', num2str(stdTrainAccuracy), '%']);
disp(['RBF Validation Accuracy: ', num2str(valAccuracy), '±', num2str(stdValAccuracy), '%']);


%% TEST classifier
testData = load('motor_imagery_test_data.mat').data;
imagiryDataC3 = testData(:,imagnationIncides,C3);
imagiryDataC4 = testData(:,imagnationIncides,C4);
baseTrialC3 = avg_across_baseline(testData(:,:,C3), FS);
baseTrialC4 = avg_across_baseline(testData(:,:,C4), FS);
trials = size(testData,1);
% we randomly chose labels beacuse we didn't have labels for the test file.
indices = randperm(trials);
hold on
%turn off display of new figures
set(0, 'DefaultFigureVisible', 'off');

testFeature =  featuresExtraction(baseTrialC3, baseTrialC4, FS, ...
        indices(1:trials/2), indices((trials/2) + 1:end) , imagnationIncides , trials, windowSize,overLap);
hold off
testFeature = zscore(testFeature);
testFeature = testFeature(feature_ind,:);
trainData =  labaled_features(:, 1:end -1);
trainLabels =  labaled_features(:, end);
% train the lda classifier on the entire training set
[ldaClassifier, err] = classify(testFeature(1:num_featurs,:)', trainData, trainLabels);
disp(['Training Accuracy: ', num2str((1 - err) * 100)]);


