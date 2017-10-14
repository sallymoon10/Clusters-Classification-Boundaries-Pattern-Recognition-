function [Pe,confusionMatrix] = ErrorAnalysis(cluster,space,classifier)
%ERRORRATE Summary of this function goes here
%   Detailed explanation goes here

n = length(cluster);
confusionMatrix = zeros(n,n);
error = 0;
counter = 0;
for i = 1:n % classes
    actual = i;
    xi = cluster(i).data';
    nSamples = size(xi,2);
    for isample = 1:nSamples;
        counter = counter + 1;
        temp1 = xi(1,isample) - space(:,:,1);
        temp2 = xi(2,isample) - space(:,:,2);
        difference = sqrt(temp1.^2 + temp2.^2);
        [~,indices] = min(difference,[],1);
        [~,index2] = min(min(difference),[],2);
        index1 = indices(1);
        predicted = classifier(index1,index2);
        confusionMatrix(actual,predicted) = confusionMatrix(actual,predicted) + 1;
        if (actual ~= predicted)
            error = error + 1;
        end
    end
end

Pe = error/counter;

end

