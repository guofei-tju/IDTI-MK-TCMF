function d = kernel_gip_0(adjmat,dim, gamma)
% Calculates the Gaussian Interaction Profile (Laarhoven, 2011) kernel from a graph adjacency 
% matrix. If tha graph is unipartite, ka = kb.
%INPUT: 
% adjmat : binary adjacency matrix
% dim    : dimension (1 - rows, 2 - cols)
%OUTPUT:
% d : kernel matrix for adjmat over dimension 'dim'

    y = adjmat;

    % Graph based kernel
	if dim == 1
        d = kernel_RBF(y,y,gamma);
    else
        d = kernel_RBF(y',y',gamma);
    end

	
	d(d==1)=0;
	d(logical(eye(size(d,1))))=1;
	
end

function New_x=remove_zeros(X)
	val_index = [];
	New_x = [];
	for i=1:size(X,2)
		a = X(:,i);
		val_a = var(a);
		val_index = [val_index,val_a];
		if val_a~=0
			New_x = [New_x,a];
		end
	end
	
end