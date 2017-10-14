% This function classifies each square of the grid in the variable space
% using a NN classifier.
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

function classIndex = NN_classifier(cluster,space)
%NN_CLASSIFIER Summary of this function goes here
% classIndex the index of the classified class

n = length(cluster);
g = zeros(size(space,1),size(space,2),n) + inf;
for i = 1:n % classes
    Zi = cluster(i).data;
    for j = 1:size(Zi,1)
        zj = Zi(j,:)';   
        temp = -(zj(1) * space(:,:,1) + zj(2) * space(:,:,2)) + (0.5 * (zj' * zj));
        g(:,:,i) = min(g(:,:,i),temp);
    end
end

% For each subsection in the space, finds the minimum NN
% to a class. The index of this minimum in the 3rd dimension of dg
% indicates which class the subsection was classified to.
[~,classIndex] = min(g,[],3);

end

