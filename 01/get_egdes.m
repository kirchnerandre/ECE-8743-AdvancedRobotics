function Edges = get_egdes(Vertices)
    edges_size  = factorial(size(Vertices,1)) ...
                / factorial(size(Vertices,1) - 2) ...
                / 2;
    Edges       = zeros(cast(edges_size, 'uint32'), 3);
    index       = 0;

    for i = 1:size(Vertices, 1)
        for j = (i + 1):size(Vertices, 1)
            index = index + 1;
            Edges(index, :) = [i j 0];
        end
    end
end
