function plot_clusters(clusters)
%plot_clusters - Plot Clusters
%   clusters - array of clusters to plot
    figure;
    t = linspace(0,2*pi,100);
    unit_circle = [cos(t); sin(t)]; % unit circle
    for i = 1:length(clusters)
        cluster = clusters(i);
        
        % Plot samples
        x = cluster.data(:,1);
        y = cluster.data(:,2);
        marker_shape = cluster.marker_shape;
        hold on, plot(x, y, marker_shape, 'markersize', 5, 'color', cluster.color);
        
        % Plot unit standard deviation contour
        % Real mean and real cov are used for this plot
        [V, D] = eig(cluster.real_cov);
        scaledV = V*sqrt(D); % scaled eigenvectors
        circle = bsxfun(@plus, scaledV*unit_circle, cluster.real_mean);
        hold on, plot(circle(1,:), circle(2,:), 'lineWidth', 2, 'color', cluster.color);
    end
end

