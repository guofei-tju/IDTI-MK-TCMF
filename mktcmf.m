function [LapA,beta_1,beta_2] = mktcmf(Klist_1,Klist_2,y,lamda_T,lamda_L,lamda_1,lamda_2,lamda_w,k1,k2,interMax,isPro)
%Multiple Kernel-based Three Collaborative Matrix Factorization (MK-TCMF)
%uestc cs, bioinformatics. 
LapA=[];
KK=[];
[n,m] = size(y);
number_k1 = size(Klist_1,3);
beta_1 = ones(number_k1,1)/number_k1;
number_k2 = size(Klist_2,3);
beta_2 = ones(number_k2,1)/number_k2;

phi_1 = zeros(number_k1,number_k1);
phi_2 = zeros(number_k2,number_k2);

for ii=1:number_k1
	for jj=1:number_k1
		mm1 = Klist_1(:,:,ii);
		mm2 = Klist_1(:,:,jj);
		phi_1(ii,jj) =  trace(mm1*mm2');
	end
end

for ii=1:number_k2
	for jj=1:number_k2
		mm1 = Klist_2(:,:,ii);
		mm2 = Klist_2(:,:,jj);
		phi_2(ii,jj) =  trace(mm1*mm2');
	end
end

l_1=ones(number_k1,1);
l_2=ones(number_k2,1);

zeta_1 = [];
zeta_2 = [];

	W1 = combine_kernels(beta_1, Klist_1);
	W2 = combine_kernels(beta_2, Klist_2);
if isPro==1
y = preprocess_Y(y,W1,W2,5,0.7);
end

[U1,S_k,V1] = svds(W1,k1);
G1 = U1*(S_k^0.5);  


[U2,S_k,V2] = svds(W2,k2);
G2 = U2*(S_k^0.5); 

A = G1;
B = G2;
k_r1 = eye(k1);
k_r2 = eye(k2);
Theta = zeros(k1,k2);

for o=1:interMax

inv_BB = pinv(B'*B);
t_inv_B = pinv(B');


	a = A'*A;
	b = lamda_T*inv_BB; 
	c = A'*y*t_inv_B;
	Theta = sylvester(a,b,c);
	
	A = (y*B*Theta' + lamda_1*W1*A)/(Theta*B'*B*Theta' + lamda_L*k_r1 + lamda_1*A'*A);
	
	B = (y'*A*Theta + lamda_2*W2*B)/(Theta'*A'*A*Theta + lamda_L*k_r2 + lamda_2*B'*B);
	
	
	zeta_1 = computer_zeta(A,Klist_1);
	temp_1=[];temp_2=[];
	temp_1 = (l_1'*((phi_1 + lamda_w*eye(number_k1))\zeta_1)) - 1;
	temp_2 = (l_1'*((phi_1 + lamda_w*eye(number_k1))\l_1));
	beta_1 = (phi_1 + lamda_w*eye(number_k1))\(zeta_1 - (temp_1/temp_2)*l_1);
	
	
	
	zeta_2 = computer_zeta(B,Klist_2);
	temp_1=[];temp_2=[];
	temp_1 = (l_2'*((phi_2 + lamda_w*eye(number_k2))\zeta_2)) - 1;
	temp_2 = (l_2'*((phi_2 + lamda_w*eye(number_k2))\l_2));
	beta_2 = (phi_2 + lamda_w*eye(number_k2))\(zeta_2 - (temp_1/temp_2)*l_2);
	
	
	W1 = combine_kernels(beta_1, Klist_1);
	W2 = combine_kernels(beta_2, Klist_2);
	

	%objc_v = computing_err(y,A,B,W1,W2,Theta)
	
end
	%reconstruct Y*
	
LapA = A*Theta*B';

end


function zeta_vector = computer_zeta(AB,KK)

	zeta_vector = zeros(size(KK,3),1);

	for ss=1:size(KK,3)
		zeta_vector(ss) = trace(AB'*KK(:,:,ss)*AB);
	end


end


function obj_v = computing_err(y,A,B,S1,S2,Theta)
		obj_1 = y-A*Theta*B';
	
	obj_2 = S1-A*A';
	obj_3 = S2-B*B';
	obj_v = norm(obj_1,'fro') + norm(Theta,'fro') + norm(A,'fro') + norm(B,'fro') + norm(obj_2,'fro') +norm(obj_3,'fro');

end


function Y=preprocess_Y(Y,Sd,St,K,eta)

    eta = eta .^ (0:K-1);

    y2_new1 = zeros(size(Y));
    y2_new2 = zeros(size(Y));

    empty_rows = find(any(Y,2) == 0);   % get indices of empty rows
    empty_cols = find(any(Y)   == 0);   % get indices of empty columns

    % for each drug i...
    for i=1:length(Sd)
        drug_sim = Sd(i,:); % get similarities of drug i to other drugs
        drug_sim(i) = 0;    % set self-similiraty to ZERO

        indices  = 1:length(Sd);    % ignore similarities 
        drug_sim(empty_rows) = [];  % to drugs of 
        indices(empty_rows) = [];   % empty rows

        [~,indx] = sort(drug_sim,'descend');    % sort descendingly
        indx = indx(1:K);       % keep only similarities of K nearest neighbors
        indx = indices(indx);   % and their indices

        drug_sim = Sd(i,:);
        y2_new1(i,:) = (eta .* drug_sim(indx)) * Y(indx,:) ./ sum(drug_sim(indx));
    end

    % for each target j...
    for j=1:length(St)
        target_sim = St(j,:); % get similarities of target j to other targets
        target_sim(j) = 0;    % set self-similiraty to ZERO

        indices  = 1:length(St);        % ignore similarities 
        target_sim(empty_cols) = [];    % to targets of
        indices(empty_cols) = [];       % empty columns

        [~,indx] = sort(target_sim,'descend');  % sort descendingly
        indx = indx(1:K);       % keep only similarities of K nearest neighbors
        indx = indices(indx);   % and their indices


        target_sim = St(j,:);
        y2_new2(:,j) = Y(:,indx) * (eta .* target_sim(indx))' ./ sum(target_sim(indx));
    end


    Y = max(Y,(y2_new1 + y2_new2)/2);

end