function fakeData_f = generateFakeData(theta_vae,gan_thetaD,mu_candidates,tree_candidates,sampleNum,fTanh,fTanh_prime)

Wrande1 = theta_vae.Wrande1;
Wrande2 = theta_vae.Wrande2;
brande1 = theta_vae.brande1;
brande2 = theta_vae.brande2;

WdecoS1Left = theta_vae.WdecoS1Left;
WdecoS1Right = theta_vae.WdecoS1Right;
WdecoS2 = theta_vae.WdecoS2;
bdecoS2 = theta_vae.bdecoS2;
bdecoS1Left = theta_vae.bdecoS1Left;
bdecoS1Right = theta_vae.bdecoS1Right;

WdecoBox = theta_vae.WdecoBox;
bdecoBox = theta_vae.bdecoBox;

WsymdecoS2 = theta_vae.WsymdecoS2;
WsymdecoS1 = theta_vae.WsymdecoS1;
bsymdecoS2 = theta_vae.bsymdecoS2;
bsymdecoS1 = theta_vae.bsymdecoS1;

 %find topK candidates for each latent code
topK = 1; %2, 3
fakeData = cell(1,ceil(sampleNum/topK));

candidateNum = 5; %10,20
for ii = 1:ceil(sampleNum/topK)
    sample_z = randn(80,1);
    p_fTanh = fTanh;
    disALL = mu_candidates;
    for jj = 1:size(mu_candidates,2)
        disALL(:,jj) = mu_candidates(:,jj)-sample_z;
    end
    disALL = abs(disALL);
    disALL = disALL.^2;
    disALL = sum(disALL);
    [~,sortIndex] = sort(disALL);

    selectIndexs = sortIndex(1:candidateNum);
      
    rd2 = p_fTanh(Wrande2*sample_z+brande2);
    rd1 = p_fTanh(Wrande1*rd2+brande1);
    
    dis_scores = zeros(1,candidateNum);
    gen_data_list = cell(1,candidateNum);
    gen_kidssym_list = cell(1,candidateNum);
    treekids_list = cell(1,candidateNum);
    
    p_tree_candidates = tree_candidates;
    for kk = 1:candidateNum
        genTree = tree2;
        treekids = p_tree_candidates{selectIndexs(kk)};
        nodenums = size(treekids,1);
        genTree.nodeFeatures = zeros(80,nodenums);
        genTree.nodeFeatures(:,nodenums) = rd1;
        
        gen_data = [];
        gen_kidssym = cell(1,nodenums);
        for jj = nodenums:-1:1
            feature = genTree.nodeFeatures(:,jj);
            nodetype = treekids(jj,3);
            sl = numel(find(treekids(:,1)==0));
            if (jj > sl)
                if (nodetype)
                    id1 = treekids(jj,1);
                    ym = p_fTanh(WsymdecoS2*feature + bsymdecoS2);
                    yp = p_fTanh(WsymdecoS1*ym + bsymdecoS1);  
                    genTree.nodeFeatures(:,id1) = yp(1:end-8);
                    reconSymparams = yp(end-7:end)';
                    gen_kidssym{id1} = reconSymparams;
                else
                    id1 = treekids(jj,1);
                    id2 = treekids(jj,2);
                    ym = p_fTanh(WdecoS2*feature + bdecoS2);
                    genTree.nodeFeatures(:,id1) = p_fTanh(WdecoS1Left*ym + bdecoS1Left);
                    genTree.nodeFeatures(:,id2) = p_fTanh(WdecoS1Right*ym + bdecoS1Right);
                    if (~isempty(gen_kidssym{jj}))
                        gen_kidssym{id1} = gen_kidssym{jj};
                        gen_kidssym{id2} = gen_kidssym{jj};
                    end
                end        
            else
                yp = p_fTanh(WdecoBox*feature + bdecoBox);
                genTree.boxes(:,jj) = yp;     
                gen_data(:,jj) = yp;     
            end
        end
        
        [~,p_score,~,~] = forwardAndBackwardGanTreeD(gan_thetaD,0,1,gen_data,treekids,gen_kidssym,fTanh,fTanh_prime);
        dis_scores(kk) = p_score;
        gen_data_list{kk} = gen_data;
        gen_kidssym_list{kk} = gen_kidssym;
        treekids_list{kk} = treekids;
    end
    
    [~,b] = sort(dis_scores, 'descend');
    selectId = b(1:topK);
    

    fakeData{ii}.symshapes_K = gen_data_list(selectId);
    fakeData{ii}.treekids_K = treekids_list(selectId);
    fakeData{ii}.symparams_K = gen_kidssym_list(selectId);
    fakeData{ii}.noise = sample_z;
    
end

fakeData_f = cell(1,topK*length(fakeData));
for ii = 1:length(fakeData)
    for jj = 1:topK
        symshapes_K = fakeData{ii}.symshapes_K;
        treekids_K = fakeData{ii}.treekids_K;
        symparams_K = fakeData{ii}.symparams_K;
        fakeData_f{(ii-1)*topK+jj}.symshapes = symshapes_K{jj};
        fakeData_f{(ii-1)*topK+jj}.treekids = treekids_K{jj};
        fakeData_f{(ii-1)*topK+jj}.symparams = symparams_K{jj};
        fakeData_f{(ii-1)*topK+jj}.noise = fakeData{ii}.noise;
    end
    
end

fakeData_f = fakeData_f(1:sampleNum);

