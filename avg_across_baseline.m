function[normalized_data] =avg_across_baseline(electrode_vec,FS)
% A function that averages the data's first second of
% a certain electrode and then normalizes the entire data according to the first second average
%inputs:
%electrode_vec= C3/C4 data
%FS= 128
%outputs
%normalized_data= the electrode's data according to the first second average
    mean_first_sec= squeeze(mean(electrode_vec(:, 1:FS), 2));
    normalized_data= electrode_vec - mean_first_sec;
end