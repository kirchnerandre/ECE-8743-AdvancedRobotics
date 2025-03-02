function plot_path(PositionBegin, PositionIntermediate, PositionFinal)
    persistent path_ab
    persistent path_bc

    if ~isempty(path_ab)
        delete(path_ab)
    end

    if ~isempty(path_bc)
        delete(path_bc)
    end

    path_ab = line([PositionBegin(1) PositionIntermediate(1)], ...
                   [PositionBegin(2) PositionIntermediate(2)], ...
                   'Color',     'green', ...
                   'LineStyle', '--');

    path_bc = line([PositionIntermediate(1) PositionFinal(1)], ...
                   [PositionIntermediate(2) PositionFinal(2)], ...
                   'Color',     'green', ...
                   'LineStyle', '--');
end
