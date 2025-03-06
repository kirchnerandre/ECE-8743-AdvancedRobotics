function plot_route(PositionBegin, PositionMiddle, PositionEnd)
    plot([PositionMiddle(1) PositionBegin(1)], [PositionMiddle(2) PositionBegin(2)]);
    plot([PositionMiddle(1) PositionEnd(1)  ], [PositionMiddle(2) PositionEnd(2)  ]);
end

