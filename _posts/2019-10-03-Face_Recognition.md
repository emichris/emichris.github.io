---
title: Facial Recognition Using Principal Component Analysis
tags: [face recognition, PCA]
image: "http://cvlab.cse.msu.edu/images/teasers/feature_transfer_longtail.png"
---

**Principal Component Analysis (PCA)** is a statistical/linear algebra method that uses orthogonal transformations to decompose a piece of data with potentially correlated components into a linearly uncorrelated set of data containing principal components. In doing PCA, a new coordintate system is found such that the greatest variance by any projection of the data lies on the first coordinate, the second greatest variance on the second coordinate, and so on... 

The Karhunen-Loeve Transform (KLT) explained [here](http://fourier.eng.hmc.edu/e161/lectures/klt/node3.html) establishes the basis for data reduction using PCA. In this project, we demonstrate PCA as a method for data/model reduction measuring recognition rate against number of principal components. 

The face data used for this project can be found at [link](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html). There are 40 different people with 10 pictures for each person. For some of the poeple, images were taken at different times, lighting conditions, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses). The project was done in Matlab. 



```
% Christian Emiyah
% EEGR 675 - Computer Vision
 
%clear all; close all; clc
 
% %% Create Test Data
% for i = 1:40
%     s = ['att_faces\s', int2str(i)];
%     d = ['test\s', int2str(i)]; mkdir (d);
%     for j = 1:2
%         S = dir(fullfile(s,'*.pgm')); % get list of all files
%         k = randi(numel(S), 1, 1);
%         F = fullfile(s, S(k).name);
%         movefile(F, d);
%     end
% end
 
%% Read in Training images
W = [];
for i = 1 : 40
    s = ['att_faces\s', int2str(i)];
    test_dir = dir(fullfile(s,'*.pgm')); % pattern to match filenames.
    for k = 1:numel(test_dir)
        F = fullfile(s, test_dir(k).name);
        img = imresize(imread(F), [32, 32]);
        [r c] = size(img);              % Get size of image
        temp = reshape(img', r*c, 1);   % Reshaping 2D images into 1D image vectors
        W = [W temp];
    end
end
```

![Faces](img/Faces.png)


```
%% Computing the mean, m
m = mean(W, 2);
imgcount = size(W, 2);
 
%% Subtract mean image from images
Z = [];
for i = 1 : imgcount
    temp = double(W(:,i)) - m;
    Z = [Z temp];
end
 
%%
% Find Eigen values and vectors
C = cov(Z);
[S,U] = eig(C);
 
%% Will Measure Recognition as Accuracy using varying number of principal components
num_eig = size(S, 2); acc = [];
 
for n = 0:num_eig-1
    n_eig_vec = S(:, [num_eig - n: num_eig]);
    V = [ ];
    for e = 1:n+1
        v = Z * n_eig_vec(:, e);
        V = [V v/norm(v, 2)];
    end
  %% 
    Y = V'*Z; %eigenfaces
    projectimg = Y;
   
    % Tst
    Y_test = [ ]; Y_test_pred = [ ];
%%
    for i = 1 : 40 % number of classes
        s = ['test\s', int2str(i)];
        S_dir = dir(fullfile(s,'*.pgm'));
        for j = 1:numel(S_dir)
            Y_test = [Y_test int32(i)]; %get true label
            F = fullfile(s,S_dir(j).name);
            test_image = imresize(imread(F), [32, 32]);
            [r c] = size(test_image);
            temp = reshape(test_image', r*c, 1); % creating (MxN)x1 image vector from the 2D image
            temp = double(temp) - m; % mean subtracted vector
            projtestimg = V'*temp;
 
            % Find matching image
            euclide_dist = [ ];
            for k = 1 : size(S, 2)
                temp = (norm(projtestimg-projectimg(:,k)))^2;
                euclide_dist = [euclide_dist temp];
            end
            [euclide_dist_min index] = min(euclide_dist);
            u = index - mod(index-1, 8);
            idx = (1 + int32(u)/8);
            Y_test_pred = [Y_test_pred idx];
        end
    end
 
%% Measure Recognition Rate
    Accuracy = int16(Y_test == Y_test_pred);
    score = sum(Accuracy)/size(Y_test, 2);
    acc = [acc score];
end
%% Plot results 
figure()
p = plot(1:320, acc, '-r');
p.LineWidth = 1.5; ax = gca; ax.Box = 'off'; ax.LineWidth = 1.5;
title('Performance Face Recognition Using PCA', 'FontWeight', 'bold')
xlabel('Number of Principal Components', 'FontWeight', 'bold')
ylabel('Recognition Rate', 'FontWeight', 'bold')
```

![PCA_Recognition_Rate.png](img/PCA_Recognition_Rate.png)

**The system achieved a recognition rate more than 90% with less than 10% (32) principal components. This drastic fall in the number of components (features) needed to model the system makes Principal Component Analysis a highly coveted tool in computer vision (which can be extended to machine learning).**
