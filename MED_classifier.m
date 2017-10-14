% This function classifies each square of the grid in the variable space
% using a MED classifier.
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

function classIndex = MED_classifier(cluster,space)
%MED_CLASSIFIER Summary of this function goes here
% Z is vector is all prototypes
% classIndex the index of the classified class

n = length(cluster);
g = zeros(size(space,1),size(space,2),n);
for i = 1:n
    zi = cluster(i).mean;
    g(:,:,i) = -(zi(1) * space(:,:,1) + zi(2) * space(:,:,2)) + (0.5 * (zi' * zi));
end

[~,classIndex] = min(g,[],3);

end

