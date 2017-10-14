function cluster  = generate_cluster(N, mu, cov)
% generate_cluster - Generates a cluster
%   N - number of points
%   mu - mean of the points
%   cov - covariance matrix
%
%   Derived from: http://math.stackexchange.com/questions/1039711/create-a-gaussian-distribution-with-a-customize-covariance-in-matlab
    randomMatrix = randn(N, size(cov,1));
    cholMatrix = chol(cov)';
    cluster = bsxfun(@plus, mu, cholMatrix*randomMatrix.').';
end

