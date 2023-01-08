function [labels] = algorithm(X,y,A,alpha,lmd,numanchor)
% X: di * n
% Z: m * n
nv=length(X); % view num
num=length(y); % sample num
k=length(unique(y)); % class num: k
choice = length(numanchor); % R
beta = 1/choice * ones(nv,choice); % beta initialization

S = cell(nv,choice);
Zi = cell(nv,choice);
Sbar = [];
xaz = zeros(nv,choice);

flag = 1;
iter = 0;
obj = [];
maxIter=50;

while flag
    iter = iter + 1;
    % updatet Z
    for iv = 1:nv
        for ir = 1:choice
            m = numanchor(ir);
            Z = zeros(m, num);
            Q = ( A{iv,ir}' * X{iv} ) / (1 + alpha);
            for ji=1:num
                idx = 1:m;
                Z(idx, ji) = EProjSimplex_new(Q(idx,ji));
            end
            Zi{iv,ir}=Z;
        end
    end
    
    % update A
    for iv = 1:nv
        for ir = 1:choice
            XZ = X{iv} * Zi{iv,ir}';
            [AU,~,AV] = svd(XZ, 'econ');
            A{iv,ir} = AU * AV';
        end
    end
    
    % update beta
    for iv = 1:nv
        for ir  = 1:choice
            xaz(iv,ir) = norm(X{iv}-A{iv,ir}*Zi{iv,ir},'fro')^2 + alpha * norm(Zi{iv,ir},'fro')^2;
        end
        proj = - 0.5/lmd * xaz(iv,:);
        beta(iv,:) = EProjSimplex_new(proj);
    end

    % obj
    term1=0;
    term2=0;
    term3=0;
    sumobj = 0;
    for iv=1:nv
        for ir = 1:choice
            term1 = norm(X{iv}-A{iv,ir}*Zi{iv,ir},'fro')^2;
            term2 = norm(Zi{iv,ir},'fro')^2;
            term3 = beta(iv,ir)^2;
            sumobj = sumobj + beta(iv,ir) * (term1 + alpha * term2) + lmd * term3;
        end
    end
    obj(end+1) = sumobj;
    % convergence
    if (iter>9) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-5 || iter>maxIter || obj(iter) < 1e-10)
        flag=0;
    end
end
% norm Z ans Sbar
for iv=1:nv
    for ir=1:choice
        rowsum = sum(Zi{iv,ir},2);
        rowsum_sq = rowsum.^(-1/2);
        rowsum_sq(rowsum_sq==inf)=0;
        sigma_sq = diag(rowsum_sq);
        Zbar = sigma_sq * Zi{iv,ir}; % m*n
        Sbar = cat(1, Sbar, sqrt(beta(iv,ir)) * Zbar); 
    end
end


[U,~,~] = mySVD(Sbar',k); 

rand('twister',5489)
labels=litekmeans(U, k, 'MaxIter', 100,'Replicates',10);
