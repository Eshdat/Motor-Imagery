function [power_band] = calculate_power_histogram(data, right_incides, left_incides,freq_band, time_range,fs)
    % this function take time range & frequency band, calculate it's power
    % band to the releveant electrode and create a histogram that
    % represents the power for each hand imagination.
    
    % data = histogram data in Hz
    % right_incides, left_incides- incides of each hand
    % freq_band= the chosen informative frequency band
    % time_range= the relevant time window
    % FS-sampling frequency
    
    %calculate time range
    time_range_start= time_range(1)*fs ;
    time_range_end = time_range(end)*fs;
    
    %calculate the powerband for the given frequency band
    relavent_data = data(:,floor(time_range_start):floor(time_range_end))';
    power_band = bandpower(relavent_data,fs, freq_band);
    
    % %convert to dB
    power_band = 10*log10(power_band);
    % plot
    create_histogram(power_band, right_incides, left_incides, 30)
    %labels
    title_text = sprintf('Power Distribution, in Frequency Band: %d - %d, in time range %.2f- %.2f',freq_band(1), freq_band(2) ,time_range(1), time_range(2));
    title(title_text);
end