% This function classifies each square of the grid in the variable space
% using a MAP classifier.
%
% cluster: An array of structs, with each struct representing a class. Each
% struct contains the samples, mean, covariance and display properties of 
% its corresponding class.
%
% space: A n-by-n-by-2 matrix representing coordinates in the 2d section of
% the feature space in which classification will be performed. The first
% layer of the 3d matrix is a n-by-n array containing the x1 coordinates of
% each subsection in the section. The second layer contains the x2 
% coordinates.
%
% p: contains the prior probaility of the clusters
%
% classIndex: A n-by-n array where each element corresponds to a subsection
% in the space, and where the value of each element is the class index of 
% the corresponding subsection, which indicates the class of that 
% subsection identified by the classifier.

function classIndex = MAP_classifier(cluster,p,space )
    numClusters = length(cluster);
    N = size(space,1); % Assume space is a square
    points = reshape(space, N*N, 2);
    posterior = zeros(N*N, 1);
    for i = 1:numClusters
        mu = cluster(i).mean';
        S = cluster(i).cov;
        P = p(i);
        pointMinusMu = bsxfun(@minus, points, mu);
        ePowerTerm = -1/2*sum(pointMinusMu*inv(S).*pointMinusMu,2);
        numerator = P*exp(ePowerTerm);
        denominator = 2*pi*sqrt(det(S));
        posterior(:,i) = numerator/denominator;
    end
    [~, maxPosteriorClass] = max(posterior, [], 2);
    classIndex = reshape(maxPosteriorClass, N, N);
end

