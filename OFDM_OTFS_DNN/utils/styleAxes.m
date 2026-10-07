function styleAxes(ax, xl, yl, ttl)
%STYLEAXES Consistent publication style: labels with units, grid, box, readable fonts.
if nargin >= 2 && ~isempty(xl),  xlabel(ax, xl);  end
if nargin >= 3 && ~isempty(yl),  ylabel(ax, yl);  end
if nargin >= 4 && ~isempty(ttl), title(ax, ttl);  end
set(ax, 'FontSize', 12, 'LineWidth', 1, 'TickDir', 'out');
grid(ax, 'on');  box(ax, 'on');
end
