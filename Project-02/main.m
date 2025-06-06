function [ Time Distance ] = main(PositionBegin, ...
                                  PositionEnd, ...
                                  SensorRange, ...
                                  StepSize, ...
                                  Clearance, ...
                                  Filename)
    left_bottom     = [min(PositionBegin(1), PositionEnd(1)) ...
                       min(PositionBegin(2), PositionEnd(2))];
    right_top       = [max(PositionBegin(1), PositionEnd(1)) ...
                       max(PositionBegin(2), PositionEnd(2))];

    [ obstacles_data, obstacles_length ] = create_obstacles(Filename);

    figure;
    axis([ left_bottom(1) - SensorRange, right_top(1) + SensorRange, ...
           left_bottom(2) - SensorRange, right_top(2) + SensorRange ]);
    axis equal
    set(gca,'XLimMode','manual');
    set(gca,'YLimMode','manual');
    hold on

    plot_obstacles(obstacles_data, obstacles_length);

    plot(PositionBegin(1), PositionBegin(2), 'r+', "LineWidth", 2, "MarkerSize", 5)
    plot(PositionEnd(1),   PositionEnd(2),   'r*', "LineWidth", 2, "MarkerSize", 5)

    position_begin  = PositionBegin;
    steps           = 0;

    tic

    while true
        radar_data = compute_radar(obstacles_data, ...
                                   obstacles_length, ...
                                   position_begin, ...
                                   SensorRange);

        radar_data = compute_clearance(radar_data, Clearance);

        [ position_middle ...
          position_begin ] = compute_step(position_begin, ...
                                          PositionEnd, ...
                                          radar_data, ...
                                          StepSize);

        plot_robot(position_begin)

        plot_path(position_begin, position_middle, PositionEnd)

        plot_radar_range(position_begin, SensorRange, 1.0)

        plot_radar_detection(position_begin, radar_data, 1.0)

        distance = sqrt((PositionEnd(1) - position_begin(1)) ^ 2 ...
                      + (PositionEnd(2) - position_begin(2)) ^ 2);

        if distance < StepSize
            break
        elseif steps > 1000
            steps = 0;
            break;
        end

        steps = steps + 1;

        pause(0.01)
    end

    Time        = toc;
    Distance    = steps * StepSize;
end
