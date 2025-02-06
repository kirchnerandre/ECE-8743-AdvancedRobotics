
close all
clear all
clc

tic

start           = [4,4];
finish          = [90, 85];
n               = 0;
circleSize      = 2;

figure()
axis([0 100 0 100])
axis square
hold on

obs(:,:,1) = [4 0; 20 10; 20 30; 60 30; 60 10];
obs(:,:,2) = [4 0; 70 10; 70 50; 90 50; 90 10];
obs(:,:,3) = [4 0; 10 40; 10 60; 50 60; 50 40];
obs(:,:,4) = [4 0; 20 70; 20 90; 80 90; 80 70];

plot_obstacles(obs)

circles(start (1), start (2), circleSize, 'facecolor', 'green')
circles(finish(1), finish(2), circleSize, 'facecolor', 'yellow')

vertices = get_vertices(start, finish, obs);
edges    = get_egdes(vertices);

edges = [ edges(1, :) ]; % debugging
plot_edges(edges, vertices);

edges    = clean_edges(edges, vertices, obs);

plot_edges(edges, vertices);
 
n       = size(obs,3);
obsx    = cell(n,1);
obsy    = cell(n,1);

for i = 1:n
    numVert = obs(1,1,i);
    curObs  = obs(2:numVert+1,:,i);
    obsx(i) = {curObs(:,1)'};
    obsy(i) = {curObs(:,2)'};
end

keys    = {1};
values  = {start};
index   = 1;

for i = 0:n-1
    for j = 2:obs(1,1,i+1)+1
        keys  (1,index+1)   = {index+1};
        values(1,index+1)   = {obs(j,:,i+1)};
        index               = index + 1;
    end
end

keys  (end+1)   = {index};
values(end+1)   = {finish};

Map             = containers.Map(keys, values);
len             = Map.values;

length          = size(keys);
G               = graph();

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
    x           = visEd(i,1);
    xx          = x{1,1};
    p1          = Map(xx(1,1));
    p2          = Map(xx(1,2));

    xplot       = [p1(1,1), p2(1,1)];
    yplot       = [p1(1,2), p2(1,2)];

    numPoints   = max(abs((p1(1,1)-p2(1,1))/0.25), abs((p1(1,2)-p2(1,2))/0.25));

    xpoints     = linspace(p1(1,1),p2(1,1), numPoints);
    ypoints     = linspace(p1(1,2),p2(1,2), numPoints);

    notInObs    = true;

    for k = 1:n
        [in,on] = inpolygon(xpoints, ypoints,obsx{k},obsy{k});

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
    p1          = Map(path(i));
    p2          = Map(path(i+1));
    totalDis    = totalDis + EuclDist(p1,p2);

    xpoints     = [p1(1,1), p2(1,1)];
    ypoints     = [p1(1,2), p2(1,2)];

    hold on
    plot(xpoints, ypoints, 'k', 'LineWidth', 3)
    title('Visibility Graph')
end

time    = toc;
disp("Environment " + 2 + ": Time = " + time + " sec, Distance = " + totalDis);
xlabel({"Environment " + 2; "Time = " + time + " sec"; "Distance = " + totalDis});

function plot_obstacles(Obstacles)
    for i = 1:size(Obstacles, 3)
        numVert = Obstacles(1, 1, i);
        polygon = polyshape(Obstacles(2:numVert+1, : ,i));
        plot(polygon);
    end
end

function Vertices = get_vertices(Start, Finish, Obstacles)
    Vertices = [];
    Vertices = [ Vertices; Start];

    for i = 1:size(Obstacles, 3)
        for j = 2:Obstacles(1, 1, i)+1
            Vertices = [ Vertices; Obstacles(j, :, i) ];
        end
    end

    Vertices = [ Vertices; Finish ];
end

function Egdes = get_egdes(Vertices)
    Egdes = [];

    for i = 1:size(Vertices, 1)
        for j = i+1:size(Vertices, 1)
            Egdes = [ Egdes; i j sqrt((Vertices(i, 1) - Vertices(j, 1)) ^ 2 ...
                                    + (Vertices(i, 2) - Vertices(j, 2)) ^ 2) ];
        end
    end
end

function Result = hit_line_segment(AB, CD)
    Result = false;

    m_ab = (AB(1, 2) - AB(2, 2)) / (AB(1, 1) - AB(2, 1));
    n_ab =  AB(1, 2) - AB(1, 1) * m_ab;

    m_cd = (CD(1, 2) - CD(2, 2)) / (CD(1, 1) - CD(2, 1));
    n_cd =  CD(1, 2) - CD(1, 1) * m_cd;

    if (AB(1, 1) == AB(2, 1)) && (CD(1, 1) == CD(2, 1))
        if AB(1, 1) ~= CD(1, 1)
            Result = false;
        else
            if ((min(CD(1, 2), CD(2, 2)) < AB(1, 2)) && (AB(1, 2) < max(CD(1, 2), CD(2, 2)))) ...
            || ((min(CD(1, 2), CD(2, 2)) < AB(2, 2)) && (AB(2, 2) < max(CD(1, 2), CD(2, 2)))) ...
            || ((min(AB(1, 2), AB(2, 2)) < CD(1, 2)) && (CD(1, 2) < max(AB(1, 2), AB(2, 2)))) ...
            || ((min(AB(1, 2), AB(2, 2)) < CD(2, 2)) && (CD(2, 2) < max(AB(1, 2), AB(2, 2))))
                Result = true;
            end
        end
    elseif AB(1, 1) == AB(2, 1)
        y = m_cd * A(1, 1) + n_cd;

        if (min(AB(1, 2), AB(2, 2)) < y) && (y < max(AB(1, 2), AB(2, 2)))
            Result = true;
        end
    elseif CD(1, 1) == CD(2, 1)
        y = m_ab * CD(1, 1) + n_ab;

        if (min(CD(1, 2), CD(2, 2)) < y) && (y < max(CD(1, 2), CD(2, 2)))
            Result = true;
        end
    else
        x = - (n_ab - n_cd) / (m_ab - m_cd);
    
        if (min(AB(1, 1), AB(2, 1)) < x) && (x < max(AB(1, 1), AB(2, 1))) ...
        && (min(CD(1, 1), CD(2, 1)) < x) && (x < max(CD(1, 1), CD(2, 1)))
            Result = true;
        end
    end
end

function Result = hit_triangle(AB, P0, P1, P2)
    Result = false;

    if hit_line_segment(AB, [P0; P1]) == true
        Result = true;
    elseif hit_line_segment(AB, [P1; P2]) == true
        Result = true;
    elseif hit_line_segment(AB, [P2; P0]) == true
        Result = true;
    else
        dot_0a_01 = (AB(1, 1) - P0(1)) * (P1(1) - P0(1)) + (AB(1, 2) - P0(2)) * (P1(2) - P0(2));
        dot_1a_12 = (AB(1, 1) - P1(1)) * (P2(1) - P1(1)) + (AB(1, 2) - P1(2)) * (P2(2) - P1(2));
        dot_2a_20 = (AB(1, 1) - P2(1)) * (P0(1) - P2(1)) + (AB(1, 2) - P2(2)) * (P0(2) - P2(2));
    
        dot_0b_01 = (AB(2, 1) - P0(1)) * (P1(1) - P0(1)) + (AB(2, 2) - P0(2)) * (P1(2) - P0(2));
        dot_1b_12 = (AB(2, 1) - P1(1)) * (P2(1) - P1(1)) + (AB(2, 2) - P1(2)) * (P2(2) - P1(2));
        dot_2b_20 = (AB(2, 1) - P2(1)) * (P0(1) - P2(1)) + (AB(2, 2) - P2(2)) * (P0(2) - P2(2));
    
        if (dot_0a_01 > 0) && (dot_1a_12 > 0) && (dot_2a_20 > 0)
            Result = true;
        elseif (dot_0b_01 > 0) && (dot_1b_12 > 0) && (dot_2b_20 > 0)
            Result = true;
        end
    end
end

function Result = hit_obstacle(AB, Obstacle)
    Result  = false;
    p0      = Obstacle(2, :);

    for i = 3:Obstacle(1, 1)
        p1 = Obstacle(i,   :);
        p2 = Obstacle(i+1, :);

        if hit_triangle(AB, p0, p1, p2) == true
            Result = true;
            break;
        end
    end
end

function Result = hit_obstacles(AB, Obstacles)
    for i = 1:size(Obstacles(3))
        if hit_obstacle(AB, Obstacles(:, :, i)) == true
            Result = true;
            return
        end
    end

    Result = false;
end

function Edges = clean_edges(Edges, Vertices, Obstacles)
    edges = [];

    for i = 1:size(Edges, 1)
        if hit_obstacles([Vertices(Edges(i, 1), :); Vertices(Edges(i, 2), :)], Obstacles) == false
            edges = [edges; Edges(i, :)];
        end
    end

    Edges = edges;
end

function plot_edges(Edges, Vertices)
    for i = 1:size(Edges, 1)
        xs = [ Vertices(Edges(i, 1), 1) Vertices(Edges(i, 2), 1) ];
        ys = [ Vertices(Edges(i, 1), 2) Vertices(Edges(i, 2), 2) ];
        plot(xs, ys);
    end
end
