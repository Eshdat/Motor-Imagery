function PlotRandomTrials(incides,time_vec, TrainingData, side_vec)
% a function that create a figure with 20 subplots that each one represents EEG signal of a different trial
% inputs:
% incides-20 random indices (between 1-64)
% time_vec- x axis
% TrainingData- the data with voltage per time
% side_vec- vector with indices of right/ left functioning

trial2represent= 20;
 for trial = 1:trial2represent
     % take the information of the chosen trial from electrode c3(represents right hand) 
     iTrial= side_vec(incides(trial));
     y_vec_right = TrainingData(iTrial,:,1); 
     % take the information of the chosen trial from electrode c4(represents left hand) 
     y_vec_left = TrainingData(iTrial,:,2);
    
     subplot(4, 5, trial)
     plot(time_vec, y_vec_right,"Color",'red')
     hold on
     plot(time_vec, y_vec_left,"Color",'green')
    
    %labels
    xlabel("Time [sec]")
    ylabel("Voltage [mV]")
    title(sprintf('trial# %d',iTrial))
 end
 leg= legend("C3", "C4");
 set(leg, 'Position',[0,0.8,0.1,0.1],'Orientation','vertical')
 
 hold off
end
