function plotSpectogram(time_vec, freq_vec, spectrogam, min, max)
% Plot the average spectrogram
% Inputs:
%   time_vec - a vector of time points which correspond to the spectrogram
%   freq_vec - a vector of frequencies which correspond to the spectrogram
%   avg_spectrogram - the average spectrogram
%   min - min value for color bar
%   max - max value for color bar


    imagesc(time_vec, freq_vec, spectrogam);
    axis xy;
    colormap(jet);
    xlabel("Time [sec]");
    ylabel("Frequency [Hz]");
    c = colorbar;
    clim([min max])
    c.Label.String = "Power / frequency [dB / Hz]";
    c.Label.Rotation = 90;
    c.Label.Position = [2.5, 0, 0];
    c.Label.FontSize = 11;


end