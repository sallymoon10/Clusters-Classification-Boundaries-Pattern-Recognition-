function [x1,x2,space] = generateSpace(cluster,numPoints)
%GENERATESPACE Summary of this function goes here
%   Detailed explanation goes here
% dim of space would be (numPoints,numPoints)

n = length(cluster);
upper_limit = -Inf;
lower_limit = Inf;

for i = 1:n
    data = cluster(i).data;
    upper_limit = max(upper_limit,ceil(max(data,[],1)));
    lower_limit = min(lower_limit,floor(min(data,[],1)));
end
x1 = linspace(lower_limit(1)-1,upper_limit(1)+1,numPoints);
x2 = linspace(lower_limit(2)-1,upper_limit(2)+1,numPoints);
[space1,space2] = meshgrid(x1,x2);
space = cat(3,space1,space2);
end

