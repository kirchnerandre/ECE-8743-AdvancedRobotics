
close all
clear all
clc

tic

start       = [4,  4 ];
goal        = [90, 85];
n           = 0;
circleSize  = 2;

figure()

axis([0 100 0 100])
axis square
hold on

obs(:,:,1) = [4 0; 20 10; 20 40; 60 40; 60 10];
obs(:,:,2) = [4 0; 50 60; 50 80; 70 80; 70 60];

n = size(obs,3);

for i = 1:n
    numVert = obs(1,1,i);
    pgon    = polyshape(obs(2:numVert+1,:,i));
    plot(pgon);
    
    if exist('origObs') == 1
        numVert = origObs(1,1,i);
        pgon = polyshape(origObs(2:numVert+1,:,i));
        plot(pgon);
    end
end

obsx = cell(n,1);
obsy = cell(n,1);

for i = 1:n
    numVert = obs(1,1,i);
    curObs = obs(2:numVert+1,:,i);
    obsx(i) = {curObs(:,1)'};
    obsy(i) = {curObs(:,2)'};
end

circles(start(1), start(2),circleSize, 'facecolor', 'green')
circles(goal(1),  goal(2), circleSize, 'facecolor', 'yellow')

keys    = {1};
values  = {start};
index   = 1;

for i = 0:n-1
    for j = 2:obs(1,1,i+1)+1
        keys  (1,index+1) = {index+1};
        values(1,index+1) = {obs(j,:,i+1)};
        index = index + 1;
    end
end

keys  (end+1) = {index};
values(end+1) = {goal};

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
xlabel({"Environment 1 "; "Time = " + time + " sec"; "Distance = " + totalDis});
