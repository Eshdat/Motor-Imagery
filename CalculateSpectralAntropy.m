function [spectralAntropy] = CalculateSpectralAntropy(imagiryData, frequencyRange, nTrials,  windowSize,overLap,FS)
    % The function calculate the spectral Antropy of the data during
    % imagery period in the relavent frequencyRange
    %Inputs - 
    % imagiryData - the original data in the imagery period
    % frequencyRange - the relavent frequency range
    % nTrials - the amount of trails, FS - the sampling rate.
    % windowSize - the size of a single
    % window for the pwelch function 
    % overLap - the size of the over laps for the pwelch function 
    % Outputs : The spectral antropy
    powerSpectrum = zeros(nTrials, length(frequencyRange));
    
    for trial = 1:nTrials
        % Compute the power spectrum for each trial separately
        welch = pwelch(imagiryData(trial,:), windowSize, overLap, frequencyRange, FS);
        powerSpectrum(trial,:) = welch;
    end
    spectralAntropy = -sum(powerSpectrum .* log2(powerSpectrum),2);
end