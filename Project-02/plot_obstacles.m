function plot_obstacles(ObstaclesData, ObstaclesLength)
    index_first = 1;
    
    for i = 1:size(ObstaclesLength, 2)
        index_last = index_first + ObstaclesLength(i) - 1;
        
        plot(ObstaclesData(1, index_first:index_last), ...
             ObstaclesData(2, index_first:index_last))
    
        index_first = index_last + 1;
    end
end
