function Edges = get_tangents_non(Vertices, Obstacles)
    length  = sum(Obstacles(1, 1, :)) ...
            + sum(Obstacles(1, 1, :)) ...
            + sum(Obstacles(1, 1, :));

    Edges   = zeros(length, 3);
    initial = 1;
    final   = size(Vertices, 1);
    index   = 0;
    offset  = 1;

    for i = 1:size(Obstacles, 3)
        for j = 1:Obstacles(1, 1, i)
            index           = index + 1;
            Edges(index, :) = [ initial (j + offset) 0 ];
        end

        offset = offset + Obstacles(1, 1, i);
    end

    offset  = 1;

    for i = 1:size(Obstacles, 3)
        for j = 1:Obstacles(1, 1, i)
            index           = index + 1;
            Edges(index, :) = [ final   (j + offset) 0 ];
        end

        offset = offset + Obstacles(1, 1, i);
    end

    offset  = 1;

    for i = 1:size(Obstacles, 3)
        for j = 1:Obstacles(1, 1, i)
            index               = index + 1;

            if j ~= 1
                Edges(index, :) = [ (j + offset - 1) (j + offset)                  0 ];
            else
                Edges(index, :) = [ (1 + offset)     (Obstacles(1, 1, i) + offset) 0 ];
            end
        end

        offset = offset + Obstacles(1, 1, i);
    end
end
