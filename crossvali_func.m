function cv_index = crossvali_func(y,nfolds)

cv_index = zeros(size(y,1),1);
step =floor(size(y,1)/nfolds);
p=randperm(size(y,1));

for j=1:nfolds

	if j~=nfolds
		st=(j-1)*step+1;
		sed=(j)*step;

	else
		st=(j-1)*step+1;
		sed=size(y,1);
	end
		cv_p=[st:sed];
		ix = p(cv_p);
		cv_index(ix) = j;
end

end
