
close all
clear all
clc

global SENSOR_RANGE
global LEN_STEP
global PT_START
global PT_GOAL

SENSOR_RANGE    = 0.50;
LEN_STEP        = 0.01;
PT_START        = [ 3; 3 ];
PT_GOAL         = [ 5; 6 ];

left_bottom = [min(PT_START(1), PT_GOAL(1)) min(PT_START(2), PT_GOAL(2))];
right_top   = [max(PT_START(1), PT_GOAL(1)) max(PT_START(2), PT_GOAL(2))];

[ obstacles_data, obstacles_length ] = create_obstacles(PT_START, PT_GOAL);

figure('units', 'normalized', 'outerposition', [0 0 1 1])
hold on
axis([left_bottom(1) right_top(1) left_bottom(2) right_top(2)]);
axis equal

plot_data_1(PT_START, PT_GOAL, obstacles_data, obstacles_length)

tic

x_path = [];
y_path = [];
x_path = [x_path PT_START(1)];
y_path = [y_path PT_START(2)];

pt_current  = PT_START;
pt_previous = pt_current;

mode = 0;   % mode = 0, do motion-to-goal
            % mode = 1, do boundary-following

while true
    curve = compute_curve(obstacles, PT_GOAL, SENSOR_RANGE);

    robot_plot = plot(pt_current(1), ...
                      pt_current(2), ...
                      'bo', ...
                      "LineWidth",  2, ...
                      "MarkerSize", 5)

    path_plot = plot(x_path, ...
                     y_path, ...
                     'b');

    current_goal_plot = line([pt_current(1) PT_GOAL(1)], ...
                             [pt_current(2) PT_GOAL(2)], ...
                             'Color',     'green', ...
                             'LineStyle', '--');

    circle_plot = drawcircle(pt_current, SENSOR_RANGE)
    
    figure
    plot_data(pt_current, PT_GOAL, curve)

    if ~isExist             % No blocking obstacle
        pt_dest = PT_GOAL;  % Drive towards goal point
    else                    % Existing blocking obstacle
        if checkIntersect(pt_current,PT_GOAL,endpoints)
            curve_plot = plot(curve(1,:),curve(2,:),'r.'); hold on
            endpt_plot = plot(endpoints(1,:),endpoints(2,:),'ro'); hold on
        end
        pt_dest = decideOi(pt_current,...   % Drive towards Oi
            PT_GOAL,endpoints);             %
    end
    
%     disp(['Go to endpoint: (',...
%             num2str(pt_dest(1)),',',num2str(pt_dest(2)),')']);
    if ~mode
        %%  motion-to-goal
        pt_current = straight2next(pt_current,pt_dest,LEN_STEP);
        
        mode = checkalong(pt_previous,...   % Check the condition 
            pt_current,PT_GOAL,...          % to start boundary following.
            curve,LEN_STEP);                % 
        if mode                             % mode from 0 to 1.
            disp('End motion-to-goal, start boundary-following...');
            angle_previous = atan2(pt_dest(2)-pt_current(2),...
                pt_dest(1)-pt_current(1));
        end
    else
        %%  boundary-following   
        [pt_current,angle_next] = follow_boundary(...    
            angle_previous,...                  % Follow the most recent
            pt_current,pt_dest,LEN_STEP,obstacles) % direction. Initially
                                                % from motion-to-goal
        angle_previous = angle_next;            %
                                                % 
        mode = checkoff(pt_current,...          % Check the condition 
            curve,PT_GOAL,SENSOR_RANGE,...      %
            obstacles,LEN_STEP,endpoints);      % to terminate boundary
                                                % following.
        if ~mode                                % mode from 1 to 0.
            disp('End boundary-following, start motion-to-goal...');
        end
    end
    
    x_path = [x_path pt_current(1)];
    y_path = [y_path pt_current(2)];
    
    pause(0.01);
    if pt_current ~= PT_GOAL
        delete(path_plot);  
        delete(robot_plot);
        delete(circle_plot);
        delete(current_goal_plot);
        if isExist && ~isempty(curve_plot)
            delete(curve_plot);
            delete(endpt_plot);
        end
    else
        break;
    end
end

disp('Reach goal point!');
toc