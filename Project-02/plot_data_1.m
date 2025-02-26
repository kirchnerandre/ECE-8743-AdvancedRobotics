function plot_data_1(PositionBegin, PositionEnd, ObstaclesData, ObstaclesLength)
    plot(PositionBegin(1), PositionBegin(2), 'r+', "LineWidth", 2, "MarkerSize", 5)
    plot(PositionEnd  (1), PositionEnd  (2), 'r*', "LineWidth", 2, "MarkerSize", 5)

    index_first = 1;

    for i = 1:size(ObstaclesLength, 2)
        index_last = index_first + ObstaclesLength(i) - 1;

        plot(ObstaclesData(1, index_first:index_last), ...
             ObstaclesData(2, index_first:index_last))

        index_first = index_last + 1;
    end
end
