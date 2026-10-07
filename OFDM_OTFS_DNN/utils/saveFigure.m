function saveFigure(fig, cfg, name)
%SAVEFIGURE Save a figure as high-resolution PNG and as MATLAB .fig in cfg.paths.plots.

if ~exist(cfg.paths.plots, 'dir'), mkdir(cfg.paths.plots); end
base = fullfile(cfg.paths.plots, name);
drawnow;
try
    exportgraphics(fig, [base '.png'], 'Resolution', 200);     % R2020a+
catch
    print(fig, [base '.png'], '-dpng', '-r200');
end
savefig(fig, [base '.fig']);
end
