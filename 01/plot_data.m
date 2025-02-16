function plot_data(Path, Edges, Vertices, Obstacles, Axis, Diameter)
    vertex_initial  = Vertices(1,   :);
    vertex_final    = Vertices(end, :);

    figure();
    axis(Axis);
    axis square;
    hold on

    plot_circle(vertex_initial(1), vertex_initial(2), Diameter, 'facecolor', 'green')
    plot_circle(vertex_final  (1), vertex_final  (2), Diameter, 'facecolor', 'yellow')

    plot_obstacles(Obstacles)

    plot_edges(Edges, Vertices)
    plot_path (Path,  Vertices)
end
