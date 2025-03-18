close all
clear all
clc

% map-0: [3; 3] -> [4.5; 6]
[ time distance ] = main([3; 3], [4.50; 6], 0.50, 0.01, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_11.jpg');

[ time distance ] = main([3; 3], [4.50; 6], 0.75, 0.01, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_12.jpg');

[ time distance ] = main([3; 3], [4.50; 6], 1.00, 0.01, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_13.jpg');

[ time distance ] = main([3; 3], [4.50; 6], 0.50, 0.05, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_14.jpg');

[ time distance ] = main([3; 3], [4.50; 6], 0.75, 0.05, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_15.jpg');

[ time distance ] = main([3; 3], [4.50; 6], 1.00, 0.05, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_16.jpg');

[ time distance ] = main([3; 3], [4.50; 6], 0.50, 0.10, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_17.jpg');

[ time distance ] = main([3; 3], [4.50; 6], 0.75, 0.10, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_18.jpg');

[ time distance ] = main([3; 3], [4.50; 6], 1.00, 0.10, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_19.jpg');

% map-0: [3; 6] -> [5; 3]
[ time distance ] = main([3; 6], [5; 3], 0.50, 0.01, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_21.jpg');

[ time distance ] = main([3; 6], [5; 3], 0.75, 0.01, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_22.jpg');

[ time distance ] = main([3; 6], [5; 3], 1.00, 0.01, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_23.jpg');

[ time distance ] = main([3; 6], [5; 3], 0.50, 0.05, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_24.jpg');

[ time distance ] = main([3; 6], [5; 3], 0.75, 0.05, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_25.jpg');

[ time distance ] = main([3; 6], [5; 3], 1.00, 0.05, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_26.jpg');

[ time distance ] = main([3; 6], [5; 3], 0.50, 0.10, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_27.jpg');

[ time distance ] = main([3; 6], [5; 3], 0.75, 0.10, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_28.jpg');

[ time distance ] = main([3; 6], [5; 3], 1.00, 0.10, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_29.jpg');

% map-0: [3; 3] -> [5; 6]
[ time distance ] = main([3; 3], [5; 6], 0.50, 0.01, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_31.jpg');

[ time distance ] = main([3; 3], [5; 6], 0.75, 0.01, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_32.jpg');

[ time distance ] = main([3; 3], [5; 6], 1.00, 0.01, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_33.jpg');

[ time distance ] = main([3; 3], [5; 6], 0.50, 0.05, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_34.jpg');

[ time distance ] = main([3; 3], [5; 6], 0.75, 0.05, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_35.jpg');

[ time distance ] = main([3; 3], [5; 6], 1.00, 0.05, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_36.jpg');

[ time distance ] = main([3; 3], [5; 6], 0.50, 0.10, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_37.jpg');

[ time distance ] = main([3; 3], [5; 6], 0.75, 0.10, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_38.jpg');

[ time distance ] = main([3; 3], [5; 6], 1.00, 0.10, 10, 'map_0.txt');
[ time distance ]
saveas(gcf, 'map_0_39.jpg');

% map-1: [16; 1] -> [1; 16]
[ time distance ] = main([1; 16], [16; 1], 2.50, 0.05, 10, 'map_1.txt');
[ time distance ]
saveas(gcf, 'map_1.jpg');

% map-2: [1; 1] -> [21; 21]
[ time distance ] = main([1; 1], [21; 21], 0.75, 0.05, 10, 'map_2.txt');
[ time distance ]
saveas(gcf, 'map_2.jpg');

% map-3: [1; 1] -> [21; 21]
[ time distance ] = main([1; 1], [21; 21], 0.75, 0.05, 10, 'map_3.txt');
[ time distance ]
saveas(gcf, 'map_3.jpg');
