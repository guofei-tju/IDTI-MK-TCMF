clear
seed = 12345678;
rand('seed', seed);
nfolds = 10; nruns=1;

%dataname = 'nr';%alph = 1;p_nearest_neighbor=9; //MKL-2^-0,2^-1
%dataname = 'gpcr';%k1 = 80;k2 = 200;lambda1 = 2^-0;lambda2=2^-0;lamda_T=2^-0;lamda_L=2^-0;lamda_w = 2^-0;interMax = 2;
%dataname = 'ic';%k1 = 200;k2 = 200;lambda1 = 2^-0;lambda2=2^-0;lamda_T=2^-0;lamda_L=2^-0;lamda_w = 2^-0;interMax = 2;
dataname = 'e';%k1 = 300;k2 = 300;lambda1 = 2^-1;lambda2=2^-1;lamda_T=2^-1;lamda_L=2^-1;lamda_w = 2^-1;interMax = 5;
load(['data/kernels/' dataname '_Drug_MACCS_fingerprint.mat']);
dataname

%load('Luo_dataset.mat');
% load adjacency matrix
[y,l1,l2] = loadtabfile(['data/interactions/' dataname '_admat_dgc.txt']);

gamma=0.5;
gamma_fp = 4;

fold_aupr_MKTMF_ka=[];fold_auc_MKTMF_ka=[];

preW = 1;
k1 = 300;k2 = 300;
lambda1 = 2^-0;lambda2=2^-0;
lamda_T=2^-0;lamda_L=2^-0;lamda_w = 2^-0;
interMax = 5;
globa_true_y_lp=[];
globa_predict_y_lp=[];
for run=1:nruns
    % split folds
%     crossval_idx = crossvalind('Kfold', length(y(:)), nfolds);
   % crossval_idx = crossvalind('Kfold',y(:),nfolds);
     crossval_idx = crossvali_func(y(:),nfolds);

    for fold=1:nfolds
        t1 = clock;
        train_idx = find(crossval_idx~=fold);
        test_idx  = find(crossval_idx==fold);

        y_train = y;
        y_train(test_idx) = 0;

        %%  1.kernels
		%% load kernels
        k1_paths = {['data/kernels/' dataname '_simmat_proteins_sw-n.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_go.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_ppi.txt'],...
                    };
		K1 = [];
        for i=1:length(k1_paths)
            [mat, labels] = loadtabfile(k1_paths{i});
            mat = process_kernel(mat);
            K1(:,:,i) = Knormalized(mat);
        end
		
        %K1(:,:,i+1) = kernel_gip_0(y_train,1, gamma);
		K1(:,:,i+1) = getGipKernel(y_train,gamma);
        k2_paths = {['data/kernels/' dataname '_simmat_drugs_simcomp.txt'],...
                   
                    ['data/kernels/' dataname '_simmat_drugs_sider.txt'],...
                    };
        K2 = [];
        for i=1:length(k2_paths)
            [mat, labels] = loadtabfile(k2_paths{i});
            mat = process_kernel(mat);
            K2(:,:,i) = Knormalized(mat);
        end
		K2(:,:,i+1) = kernel_gip_0(Drug_MACCS_fingerprint,1, gamma_fp);
        %K2(:,:,i+1+1) = kernel_gip_0(y_train,2, gamma);
		K2(:,:,i+1+1) = getGipKernel(y_train',gamma);
		
        %% perform predictions
        %lambda = 1;
     


		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		% 2. multiple kernel 

		[A_cos_com,beta_1,beta_2] = mktcmf(K1,K2,y_train,lamda_T,lamda_L,lambda1,lambda2,lamda_w,k1,k2,interMax,preW);beta_1
		beta_2
		
		t2=clock;
		etime(t2,t1)
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
        %% 4. evaluate predictions
        yy=y;
        %yy(yy==0)=-1;
        %stats = evaluate_performance(y2(test_idx),yy(test_idx),'classification');
		test_labels = yy(test_idx);
		predict_scores = A_cos_com(test_idx);
		%[X,Y,tpr,aupr_MKTMF_A_KA] = perfcurve(test_labels,predict_scores,1, 'xCrit', 'reca', 'yCrit', 'prec');
		aupr_MKTMF_A_KA=calculate_aupr(test_labels,predict_scores);
		%[X,Y,THRE,AUC_MKTMF_KA,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_labels,predict_scores,1);
    AUC_MKTMF_KA=calculate_auc(test_labels,predict_scores);
		
		fprintf('---------------\nRUN %d - FOLD %d  \n', run, fold)

		fprintf('%d - FOLD %d - weighted_kernels_MKTMF_AUPR: %f \n', run, fold, aupr_MKTMF_A_KA)
		

		fold_aupr_MKTMF_ka=[fold_aupr_MKTMF_ka;aupr_MKTMF_A_KA];
		fold_auc_MKTMF_ka=[fold_auc_MKTMF_ka;AUC_MKTMF_KA];

		
		globa_true_y_lp=[globa_true_y_lp;test_labels];
		globa_predict_y_lp=[globa_predict_y_lp;predict_scores];
		%break;
		
    end
    
    
end
RMSE = sqrt(sum((globa_predict_y_lp-globa_true_y_lp).^2)/length(globa_predict_y_lp))



mean_aupr_kronls_ka = mean(fold_aupr_MKTMF_ka)
mean_auc_kronls_ka = mean(fold_auc_MKTMF_ka)
