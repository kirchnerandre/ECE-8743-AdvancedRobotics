function Obstacles = load_obstacles(File)
    %
    % File should contain the list of vertices of all obstacles in the
    % following format.
    %
    % Size_1 0 Vertex_11_X Vertex_11_Y ... Vertex_1N_X Vertex_1N_Y
    % ...
    % Size_M 0 Vertex_M1_X Vertex_M1_Y ... 0           0
    %
    % Each line has all the data of a single obstacle, where the first
    % number is the number of vertices of the obstacle, the 2nd number
    % should be ZERO followed by pairs of numbers that represent the X and
    % Y coordinates of each vertex.
    %
    % All rows should have the same length, and they should be padded with
    % ZEROs if needed.
    %

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
