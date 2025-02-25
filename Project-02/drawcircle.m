%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   TANGENTBUG.M
%   ECE8743 Advanced Robotics
%   Date:   Spring 2024
%   Description:    Implement TangentBug path planning algorithm.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function circle_plot = drawcircle(centre,radius)
%drawcircle Draw a circle around centre with radius

theta = linspace(0,2*pi);
x = radius*cos(theta) + centre(1);
y = radius*sin(theta) + centre(2);
circle_plot = plot(x,y,'m');

end

