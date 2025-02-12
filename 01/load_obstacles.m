function Obstacles = load_obstacles(File)
    obstacles = load(File);
    [m n ]    = size(obstacles);
    Obstacles = zeros(n / 2, 2, m);

    for i = 1:m
        for j = 0:(n/2 - 1)
            Obstacles(j + 1, 1, i) = obstacles(i, 2 * j + 1);
            Obstacles(j + 1, 2, i) = obstacles(i, 2 * j + 2);
        end
    end
end
