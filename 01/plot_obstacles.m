function plot_obstacles(Obstacles)
    for i = 1:size(Obstacles, 3)
        numVert = size(Obstacles, 3);
        pgon    = polyshape(Obstacles(2:numVert+1,:,i));
        plot(pgon);
    end
end
