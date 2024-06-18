function create_histogram(data,right_indexex,left_indexes, num_col)
    % A function that receives data and builds a 
    % histogram with reference to the indices of the right/left hand
    % inputs:
    %data= Common data for both hands
    %right_indexex= the right hand imagination indexes (64)
    %left_indexes= the left hand imagination indexes (64)
    %num_col=  number of bins 
    
    [N_right, edges_right]= histcounts(data(right_indexex), num_col);
    edges_right= edges_right(1: end-1);
    
    [N_left, edges_left]= histcounts(data(left_indexes), num_col);
    edges_left= edges_left(1: end-1);
    
    bar(edges_right, N_right, 'FaceAlpha',0.7)
    hold on
    bar(edges_left, N_left, 'FaceAlpha',0.7)
    
    %labels
    xlabel('Power (dB)');
    ylabel('Count');
    legend('right', 'left');
    hold off
end