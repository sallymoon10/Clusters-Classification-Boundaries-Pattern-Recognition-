clear all;
close all;
clc;

% Declare variables
n_A = 200;
n_B = 200;
n_C = 100;
n_D = 200;
n_E = 150;

mean_A = [5; 10];
mean_B = [10; 15];
mean_C = [5; 10];
mean_D = [15; 10];
mean_E = [10; 5];

cov_A = [8 0; 0 4];
cov_B = [8 0; 0 4];
cov_C = [8 4; 4 40];
cov_D = [8 0; 0 8];
cov_E = [10 -5; -5 20];

% Generate Clusters
cluster_A = generate_cluster(n_A, mean_A, cov_A);
cluster_B = generate_cluster(n_B, mean_B, cov_B);
cluster_C = generate_cluster(n_C, mean_C, cov_C);
cluster_D = generate_cluster(n_D, mean_D, cov_D);
cluster_E = generate_cluster(n_E, mean_E, cov_E);

sample_mean_A = mean(cluster_A)';
sample_mean_B = mean(cluster_B)';
sample_mean_C = mean(cluster_C)';
sample_mean_D = mean(cluster_D)';
sample_mean_E = mean(cluster_E)';

sample_cov_A = cov(cluster_A);
sample_cov_B = cov(cluster_B);
sample_cov_C = cov(cluster_C);
sample_cov_D = cov(cluster_D);
sample_cov_E = cov(cluster_E);
%We need to remember to replace these function calls to generate_cluster
%with function calls for Bahareh's code for generating clusters, since it's
%easier to explain for the report (no use of chol function). After doing
%that, this comment can be deleted.

% I recommend to use chol function, as I have realized that these two
% methods does not give me similar weighting matrix. I don't know why.

% Plot Clusters
orange_rgb = [255/255 165/255 0];
purple_rgb = [128/255 0 128/255];
cluster_A_struct = struct('data', cluster_A, 'marker_shape', 'x', ...
                            'color', purple_rgb, 'mean', sample_mean_A, 'cov', sample_cov_A, ...
                            'real_mean',mean_A,'real_cov',cov_A);
cluster_B_struct = struct('data', cluster_B, 'marker_shape', 'o', ...
                            'color', orange_rgb, 'mean', sample_mean_B, 'cov', sample_cov_B, ...
                            'real_mean',mean_B,'real_cov',cov_B);
cluster_C_struct = struct('data', cluster_C, 'marker_shape', '*', ...
                            'color', purple_rgb, 'mean', sample_mean_C, 'cov', sample_cov_C, ...
                            'real_mean',mean_C,'real_cov',cov_C);
cluster_D_struct = struct('data', cluster_D, 'marker_shape', 's', ...
                            'color', orange_rgb, 'mean', sample_mean_D, 'cov', sample_cov_D, ...
                            'real_mean',mean_D,'real_cov',cov_D);
cluster_E_struct = struct('data', cluster_E, 'marker_shape', 'd', ...
                            'color', 'g', 'mean', sample_mean_E, 'cov', sample_cov_E, ...
                            'real_mean',mean_E,'real_cov',cov_E);
                        
clusters_AB = [cluster_A_struct cluster_B_struct];
clusters_CDE = [cluster_C_struct cluster_D_struct cluster_E_struct];

%% AB

% Classifiers
numPoints = 500;
[x1,x2,space] = generateSpace(clusters_AB,numPoints);   % generate space
classIndexMED = MED_classifier(clusters_AB,space);      % MED
classIndexGED = GED_classifier(clusters_AB,space);      % GED
p = [n_A;n_B]/(n_A+n_B);
classIndexMAP = MAP_classifier(clusters_AB,p,space);    % MAP
classIndexNN = NN_classifier(clusters_AB,space);        % NN
classIndexkNN = kNN_classifier(clusters_AB,space,5);    % kNN

MED_Pe_mean = 0;
GED_Pe_mean = 0;
MAP_Pe_mean = 0;
NN_Pe_mean = 0;
kNN_Pe_mean = 0;
for i = 1:10
% Test samples
cluster_Atest = generate_cluster(n_A, mean_A, cov_A);
cluster_Btest = generate_cluster(n_B, mean_B, cov_B);

sample_mean_Atest = mean(cluster_Atest)';
sample_mean_Btest = mean(cluster_Btest)';

sample_cov_Atest = cov(cluster_Atest);
sample_cov_Btest = cov(cluster_Btest);

cluster_A_structtest = struct('data', cluster_Atest, 'marker_shape', 'x', ...
                            'color', purple_rgb, 'mean', sample_mean_Atest, 'cov', sample_cov_Atest, ...
                            'real_mean',mean_A,'real_cov',cov_A);
cluster_B_structtest = struct('data', cluster_Btest, 'marker_shape', 'o', ...
                            'color', orange_rgb, 'mean', sample_mean_Btest, 'cov', sample_cov_Btest, ...
                            'real_mean',mean_B,'real_cov',cov_B);
                        
clusters_ABtest = [cluster_A_structtest cluster_B_structtest];

% Error Analysis
[MED_Pe,MED_confusionMatrix] = ErrorAnalysis(clusters_ABtest,space,classIndexMED);
[GED_Pe,GED_confusionMatrix] = ErrorAnalysis(clusters_ABtest,space,classIndexGED);
[MAP_Pe,MAP_confusionMatrix] = ErrorAnalysis(clusters_ABtest,space,classIndexMAP);
[NN_Pe,NN_confusionMatrix] = ErrorAnalysis(clusters_ABtest,space,classIndexNN);
[kNN_Pe,kNN_confusionMatrix] = ErrorAnalysis(clusters_ABtest,space,classIndexkNN);
MED_Pe_mean = MED_Pe_mean + MED_Pe;
GED_Pe_mean = GED_Pe_mean + GED_Pe;
MAP_Pe_mean = MAP_Pe_mean + MAP_Pe;
NN_Pe_mean = NN_Pe_mean + NN_Pe;
kNN_Pe_mean = kNN_Pe_mean + kNN_Pe;
end

MED_Pe_mean = MED_Pe_mean/10;
GED_Pe_mean = GED_Pe_mean/10;
MAP_Pe_mean = MAP_Pe_mean/10;
NN_Pe_mean = NN_Pe_mean/10;
kNN_Pe_mean = kNN_Pe_mean/10;
disp('CASE 1:')
disp(['Error MED: ',num2str(MED_Pe_mean)]), disp(MED_confusionMatrix)
disp(['Error GED: ',num2str(GED_Pe_mean)]), disp(GED_confusionMatrix)
disp(['Error MAP: ',num2str(MAP_Pe_mean)]), disp(MAP_confusionMatrix)
disp(['Error NN: ',num2str(NN_Pe_mean)]), disp(NN_confusionMatrix)
disp(['Error 5NN: ',num2str(kNN_Pe_mean)]), disp(kNN_confusionMatrix)

% Plot
plot_clusters(clusters_AB);
xlabel('x1'), ylabel('x2')
axis equal
title('CASE 1: Samples and the Unit Standard Deviation Contour')
[~,~,~,~] = legend({'A samples','A contour','B samples','B contour'});
xlim([min(x1) max(x1)])
ylim([min(x2) max(x2)])

plot_clusters(clusters_AB);
hold on
[~,MED] = contour(x1,x2,classIndexMED,'-r');
[~,GED] = contour(x1,x2,classIndexGED,'.b');
[~,MAP] = contour(x1,x2,classIndexMAP,'k');
xlabel('x1'), ylabel('x2')
axis equal
title('CASE 1')
[~,~,~,~] = legend({'A samples','A contour','B samples','B contour','MED','GED','MAP'});
xlim([min(x1) max(x1)])
ylim([min(x2) max(x2)])

plot_clusters(clusters_AB);
hold on
[~,NN] = contour(x1,x2,classIndexNN,'-r');
[~,kNN] = contour(x1,x2,classIndexkNN,'.b');
xlabel('x1'), ylabel('x2')
axis equal
title('CASE 1')
[~,~,~,~] = legend({'A samples','A contour','B samples','B contour','NN','kNN'});
xlim([min(x1) max(x1)])
ylim([min(x2) max(x2)])

%% CDE

% Classifiers
[x1,x2,space] = generateSpace(clusters_CDE,numPoints);   % generate space
classIndexMED = MED_classifier(clusters_CDE,space);      % MED
classIndexGED = GED_classifier(clusters_CDE,space);      % GED
p = [n_C;n_D;n_E]/(n_C+n_D+n_E);
classIndexMAP = MAP_classifier(clusters_CDE,p,space);    % MAP
classIndexNN = NN_classifier(clusters_CDE,space);        % NN
classIndexkNN = kNN_classifier(clusters_CDE,space,5);    % kNN

for i = 1:10
% Test samples
cluster_Ctest = generate_cluster(n_C, mean_C, cov_C);
cluster_Dtest = generate_cluster(n_D, mean_D, cov_D);
cluster_Etest = generate_cluster(n_E, mean_E, cov_E);

sample_mean_Ctest = mean(cluster_Ctest)';
sample_mean_Dtest = mean(cluster_Dtest)';
sample_mean_Etest = mean(cluster_Etest)';

sample_cov_Ctest = cov(cluster_Ctest);
sample_cov_Dtest = cov(cluster_Dtest);
sample_cov_Etest = cov(cluster_Etest);

cluster_C_structtest = struct('data', cluster_Ctest, 'marker_shape', 'x', ...
                            'color', purple_rgb, 'mean', sample_mean_Ctest, 'cov', sample_cov_Ctest, ...
                            'real_mean',mean_C,'real_cov',cov_C);
cluster_D_structtest = struct('data', cluster_Dtest, 'marker_shape', 'o', ...
                            'color', orange_rgb, 'mean', sample_mean_Dtest, 'cov', sample_cov_Dtest, ...
                            'real_mean',mean_D,'real_cov',cov_D);
cluster_E_structtest = struct('data', cluster_Etest, 'marker_shape', 'o', ...
                            'color', 'g', 'mean', sample_mean_Etest, 'cov', sample_cov_Etest, ...
                            'real_mean',mean_E,'real_cov',cov_E);
                                          
clusters_CDEtest = [cluster_C_structtest cluster_D_structtest cluster_E_structtest];

% Error Analysis
[MED_Pe,MED_confusionMatrix] = ErrorAnalysis(clusters_CDEtest,space,classIndexMED);
[GED_Pe,GED_confusionMatrix] = ErrorAnalysis(clusters_CDEtest,space,classIndexGED);
[MAP_Pe,MAP_confusionMatrix] = ErrorAnalysis(clusters_CDEtest,space,classIndexMAP);
[NN_Pe,NN_confusionMatrix] = ErrorAnalysis(clusters_CDEtest,space,classIndexNN);
[kNN_Pe,kNN_confusionMatrix] = ErrorAnalysis(clusters_CDEtest,space,classIndexkNN);
MED_Pe_mean = MED_Pe_mean + MED_Pe;
GED_Pe_mean = GED_Pe_mean + GED_Pe;
MAP_Pe_mean = MAP_Pe_mean + MAP_Pe;
NN_Pe_mean = NN_Pe_mean + NN_Pe;
kNN_Pe_mean = kNN_Pe_mean + kNN_Pe;
end

MED_Pe_mean = MED_Pe_mean/10;
GED_Pe_mean = GED_Pe_mean/10;
MAP_Pe_mean = MAP_Pe_mean/10;
NN_Pe_mean = NN_Pe_mean/10;
kNN_Pe_mean = kNN_Pe_mean/10;
disp('CASE 2:')
disp(['Error MED: ',num2str(MED_Pe_mean)]), disp(MED_confusionMatrix)
disp(['Error GED: ',num2str(GED_Pe_mean)]), disp(GED_confusionMatrix)
disp(['Error MAP: ',num2str(MAP_Pe_mean)]), disp(MAP_confusionMatrix)
disp(['Error NN: ',num2str(NN_Pe_mean)]), disp(NN_confusionMatrix)
disp(['Error 5NN: ',num2str(kNN_Pe_mean)]), disp(kNN_confusionMatrix)

% Plot
plot_clusters(clusters_CDE);
hold on
xlabel('x1'), ylabel('x2')
axis equal
title('CASE 2: Samples and the Unit Standard Deviation Contour')
[~,~,~,~] = legend({'C samples','C contour','D samples','D contour','E samples','E contour'});
xlim([min(x1) max(x1)])
ylim([min(x2) max(x2)])

plot_clusters(clusters_CDE);
hold on
[~,MED] = contour(x1,x2,classIndexMED,'-r');
[~,GED] = contour(x1,x2,classIndexGED,'.b');
[~,MAP] = contour(x1,x2,classIndexMAP,'k');
xlabel('x1'), ylabel('x2')
axis equal
title('CASE 2')
[~,~,~,~] = legend({'C samples','C contour','D samples','D contour','E samples','E contour','MED','GED','MAP'});
xlim([min(x1) max(x1)])
ylim([min(x2) max(x2)])

plot_clusters(clusters_CDE);
hold on
[~,NN] = contour(x1,x2,classIndexNN,'-r');
[~,kN] = contour(x1,x2,classIndexkNN,'.b');
xlabel('x1'), ylabel('x2')
axis equal
title('CASE 2')
[~,~,~,~] = legend({'C samples','C contour','D samples','D contour','E samples','E contour','NN','kNN'});
xlim([min(x1) max(x1)])
ylim([min(x2) max(x2)])