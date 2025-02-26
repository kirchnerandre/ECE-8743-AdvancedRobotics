
function [PositionNext, AngleNext] = follow_boundary(AnglePrevious, ...
                                                     PositionCurrent, ...
                                                     PositionDestiny, ...
                                                     StepLength, ...
                                                     Obstacles)
    PositionNext        = zeros(2,1);
    angle_most_current  = AnglePrevious;
    min_dist            = Inf;
    n_curve             = size(Obstacles,2);
    pt_closest          = Obstacles(:,1);
    
    for i = 1:n_curve
        if norm(PositionCurrent-Obstacles(:,i)) < min_dist
            min_dist    = norm(PositionCurrent-Obstacles(:,i));
            pt_closest  = Obstacles(:,i);
        end
    end
    
    angle_closest = atan2(pt_closest(2) - PositionCurrent(2), ...
                          pt_closest(1) - PositionCurrent(1));
    
    if angle_closest > angle_most_current
        AngleNext = angle_closest - pi/2;
    else
        AngleNext = angle_closest + pi/2;
    end
    
    PositionNext(1) = PositionCurrent(1) + cos(AngleNext) * StepLength;
    PositionNext(2) = PositionCurrent(2) + sin(AngleNext) * StepLength;
end
