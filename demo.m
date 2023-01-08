% Demo of FDAGF on dataset yaleA

clc; 
clear;

datapath = './';
dataname = 'yaleA_3view';
load([dataname,'.mat']); 
fprintf('Dataset:%s\t',dataname);

addpath(genpath('./'));

metric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};

y=Y;
nv=length(X); % view
ns=length(unique(y)); % k
anchor_traverse = 4; % R

% Parameter Setting
alpha = 1000; lmd = 10.^1; %% yaleA_3view

numanchor = (1:anchor_traverse)*ns;
Xnor = cell(nv,1);
H = cell(nv,length(numanchor));
for ni=1:nv
    nor = mapstd(X{ni}',0,1); % Normalize X
    % initialize A
    for nj=1:length(numanchor)
        rand('twister',5489);
        [~, anc] = litekmeans(nor',numanchor(nj),'MaxIter', 100,'Replicates',10);
        H{ni,nj} = anc';
    end
    Xnor{ni} = nor;
end

[ids] = algorithm(Xnor,y,H,alpha,lmd,numanchor);
% Performance evaluation
result=Clustering8Measure(ids,y); % 'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'

fprintf('alpha=%d \tlmd=%d\nacc:%5.4f\tnmi:%5.4f\tpur:%5.4f\tFs:%5.4f\t\n',[...
         alpha lmd result(1) result(2) result(3) result(4)]);