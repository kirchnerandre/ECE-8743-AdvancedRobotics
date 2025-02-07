function plot_obstacles(Obstacles)
    for i = 1:size(Obstacles, 3)
        numVert = Obstacles(1, 1, i);
        pgon    = polyshape(Obstacles(2:numVert+1,:,i));
        plot(pgon);
    end
end
