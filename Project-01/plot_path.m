function plot_path(Path, Vertices)
    for i = 1:(size(Path, 2) - 1)
        plot([Vertices(Path(i), 1) Vertices(Path(i + 1), 1)], ...
             [Vertices(Path(i), 2) Vertices(Path(i + 1), 2)], ...
             'k', ...
             'LineWidth', 3)
    end
end
