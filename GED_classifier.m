% This function classifies each square of the grid in the variable space
% using a GED classifier.
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
% classIndex: A n-by-n array where each element corresponds to a subsection
% in the space, and where the value of each element is the class index of 
% the corresponding subsection, which indicates the class of that 
% subsection identified by the classifier.

function classIndex = GED_classifier(cluster,space)

% n is the number of structs/classes in cluster
n = length(cluster);
% Array which holds the generalized Euclidean distance of each subsection
% in the space to each class in the 3rd dimension.
dg = zeros(size(space,1),size(space,2),n);

for i = 1:n
    mui = cluster(i).mean;
    S = cluster(i).cov;
    for x1 = 1:size(space,1)
        for x2 = 1:size(space,2)
            x = [space(x1,x2,1);space(x1,x2,2)]; % finds coordinates of this particular subsection
            dg(x1,x2,i) =  sqrt((x - mui)' * inv(S) * (x - mui));
        end
    end
end

% For each subsection in the space, finds the minimum generalized Euclidean
% distance to a class. The index of this minimum in the 3rd dimension of dg
% indicates which class the subsection was classified to.
[~,classIndex] = min(dg,[],3); 

end

