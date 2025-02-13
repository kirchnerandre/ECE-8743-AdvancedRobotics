function [ Distance Path Edges Vertices ] = visibility_graph_tangent(VertexInitial, VertexFinal, Obstacles, XMin, XMax, YMin, YMax)
    Distance = 0;
    Path = [];

    Vertices = get_vertices(VertexInitial, VertexFinal, Obstacles);

    Edges = get_edges_tangents(Vertices, Obstacles, XMin, XMax, YMin, YMax);

    
end
