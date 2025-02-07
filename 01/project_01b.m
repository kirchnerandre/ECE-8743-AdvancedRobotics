
close all
clear all
clc

tic

vertex_initial  = [4,  4 ];
vertex_final    = [90, 85];
n               = 0;
circleSize      = 2;

obstacles(:,:,1) = [4 0; 20 10; 20 30; 60 30; 60 10];
obstacles(:,:,2) = [4 0; 70 10; 70 50; 90 50; 90 10];
obstacles(:,:,3) = [4 0; 10 40; 10 60; 50 60; 50 40];
obstacles(:,:,4) = [4 0; 20 70; 20 90; 80 90; 80 70];

figure()

axis([0 100 0 100])
axis square
hold on

circles(vertex_initial(1), vertex_initial(2), circleSize, 'facecolor', 'green')
circles(vertex_final  (1), vertex_final  (2), circleSize, 'facecolor', 'yellow')

n = size(obstacles, 3);

plot_obstacles(obstacles)

vertices    = get_vertices(vertex_initial, vertex_final, obstacles)
edges       = get_egdes(vertices, obstacles)

plot_edges(edges, vertices);

path = get_path(vertices, edges);

plot_path(path, vertices)


figure()

axis([0 100 0 100])
axis square
hold on

plot_obstacles(obstacles)

circles(vertex_initial(1), vertex_initial(2), circleSize, 'facecolor', 'green')
circles(vertex_final  (1), vertex_final  (2), circleSize, 'facecolor', 'yellow')

obstaclesx = cell(n,1);
obstaclesy = cell(n,1);

for i = 1:n
    numVert         = obstacles(1,1,i);
    curObs          = obstacles(2:numVert+1,:,i);
    obstaclesx(i)   = {curObs(:,1)'};
    obstaclesy(i)   = {curObs(:,2)'};
end

keys    = {1};
values  = {vertex_initial};
index   = 1;

for i = 0:n-1
    for j = 2:obstacles(1,1,i+1)+1
        keys  (1,index+1) = {index+1};
        values(1,index+1) = {obstacles(j,:,i+1)};
        index = index + 1;
    end
end

keys  (end+1) = {index};
values(end+1) = {vertex_final};

Map     = containers.Map(keys, values);
len     = Map.values;
length  = size(keys);
G       = graph();

for i = 1:length(2)
    for j = (i + 1):length(2)
        p1      = Map(keys{i});
        p2      = Map(keys{j});
        dist    = EuclDist(p1, p2);
        G       = addedge(G,keys{i}, keys{j}, dist);
    end
end

visEd   = G.Edges;
sizeEd  = size(G.Edges);

for i=1:sizeEd(1)
    x = visEd(i,1);
    xx = x{1,1};
    p1 = Map(xx(1,1));
    p2 = Map(xx(1,2));
    
    xplot = [p1(1,1), p2(1,1)];
    yplot = [p1(1,2), p2(1,2)];
    
    numPoints = max(abs((p1(1,1)-p2(1,1))/0.25), abs((p1(1,2)-p2(1,2))/0.25));
    
    xpoints = linspace(p1(1,1),p2(1,1), numPoints);
    ypoints = linspace(p1(1,2),p2(1,2), numPoints);
    
    notInObs = true;
    
    for k = 1:n
        [in,on] = inpolygon(xpoints, ypoints,obstaclesx{k},obstaclesy{k});
        
        if max(xor(in,on))
            notInObs = false;
            break;
        end
    end
        
    if notInObs
        plot(xplot, yplot, 'b')
    else
        G = rmedge(G,xx(1),xx(2));
    end
end

path        = shortestpath(G, keys{1}, keys{length(2)});
pathSize    = size(path);
totalDis    = 0;
path

for i=1:pathSize(2)-1
    p1 = Map(path(i));
    p2 = Map(path(i+1));

    totalDis = totalDis + EuclDist(p1,p2);

    xpoints = [p1(1,1), p2(1,1)];
    ypoints = [p1(1,2), p2(1,2)];
    hold on
    plot(xpoints, ypoints, 'k', 'LineWidth', 3)
    title('Visibility Graph')
end

time = toc;

disp("Environment 1 " + ": Time = " + time + " sec, Distance = " + totalDis);

function plot_obstacles(Obstacles)
    for i = 1:size(Obstacles, 3)
        numVert = size(Obstacles, 3)
        pgon    = polyshape(Obstacles(2:numVert+1,:,i));
        plot(pgon);
    end
end

function plot_edges(Edges, Vertices)
    for i = 1:size(Edges, 1)
        plot([Vertices(Edges(i, 1), 1) Vertices(Edges(i, 2), 1)], ...
             [Vertices(Edges(i, 1), 2) Vertices(Edges(i, 2), 2)], 'b')
    end
end

function plot_path(Path, Vertices)
    for i = 1:(size(Path, 2) - 1)
        plot([Vertices(Path(i), 1) Vertices(Path(i + 1), 1)], ...
             [Vertices(Path(i), 2) Vertices(Path(i + 1), 2)], 'k', 'LineWidth', 3)
    end
end

function Vertices = get_vertices(VertexInitial, VertexFinal, Obstacles)
    vertices_size = 2; % VertexInitial + VertexFinal 
    
    for i = 1:size(Obstacles, 3)
        vertices_size = vertices_size + Obstacles(1, 1, i);
    end
    
    Vertices = zeros(vertices_size, 2);
    
    Vertices(vertices_size, :) = VertexFinal;
    
    for i = 1:size(Obstacles, 3)
        for j = 2:Obstacles(1, 1, i)+1
            vertices_size = vertices_size - 1;
            Vertices(vertices_size, :) = Obstacles(j, :, i);
        end
    end
    
    Vertices(1, :) = VertexInitial;
end

function Edges = get_egdes(Vertices, Obstacles)
    edges_size  = factorial(size(Vertices, 1)) / factorial(size(Vertices, 1) - 2) / 2;
    Edges       = zeros(edges_size, 3);
    index       = 0;

    for i = 1:size(Vertices, 1)
        for j = (i + 1):size(Vertices, 1)
            points = 4 * max(abs(Vertices(i, 1) - Vertices(j, 1)), ...
                             abs(Vertices(i, 2) - Vertices(j, 2)));

            points_x = linspace(Vertices(i, 1), Vertices(j, 1), points);
            points_y = linspace(Vertices(i, 2), Vertices(j, 2), points);

            valid = true;

            for k = 1:size(Obstacles, 3)
                [ in, on ] = inpolygon(points_x, points_y, Obstacles(2:end, 1, k), Obstacles(2:end, 2, k));

                if max(xor(in, on))
                    valid = false;
                    break
                end
            end

            if valid
                index = index + 1;
                Edges(index, :) = [i j sqrt((Vertices(i, 1) - Vertices(j, 1)) ^ 2 + (Vertices(i, 2) - Vertices(j, 2)) ^ 2)];
            end
        end
    end

    Edges = Edges(1:index, :);
end

function Path = get_path(Vertices, Edges)
    sources = zeros(2 * size(Edges, 1), 1);
    targets = zeros(2 * size(Edges, 1), 1);
    weights = zeros(2 * size(Edges, 1), 1);

    for i = 1:size(Edges, 1)
        sources(2 * i - 1)  = Edges(i, 1);
        targets(2 * i - 1)  = Edges(i, 2);
        weights(2 * i - 1)  = Edges(i, 3);

        sources(2 * i)      = Edges(i, 2);
        targets(2 * i)      = Edges(i, 1);
        weights(2 * i)      = Edges(i, 3);
    end

    G = graph(sources, targets, weights);

    Path = shortestpath(G, 1, size(Vertices, 1));
end
