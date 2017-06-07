function [muCandidates, treeCandidates] = findCandidates(vae_theta, data_full, fTanh)

WencoV1Left = vae_theta.WencoV1Left; WencoV1Right = vae_theta.WencoV1Right; WencoV2 = vae_theta.WencoV2;
WsymencoV1 = vae_theta.WsymencoV1; WsymencoV2= vae_theta.WsymencoV2; 
WencoBox = vae_theta.WencoBox;
bencoV1 = vae_theta.bencoV1; bencoV2= vae_theta.bencoV2;
bsymencoV1 = vae_theta.bsymencoV1; bsymencoV2 = vae_theta.bsymencoV2;
bencoBox = vae_theta.bencoBox; 

Wranen1 = vae_theta.Wranen1; Wranen2 = vae_theta.Wranen2; 
branen1 = vae_theta.branen1; branen2 = vae_theta.branen2; 

data = data_full(1:20:end);
muCandidates = zeros(80,length(data));
treeCandidates = cell(1,length(data));
parfor ii = 1:length(data)
    symboxes = data{ii}.symshapes;
    treekids = data{ii}.treekids;
    sl = numel(find(treekids(:,1)==0));
    symparams = data{ii}.symparams;
    p_fTanh = fTanh;
    Tree = tree2;
    Tree.nodeFeatures = zeros(80,size(treekids,1));
    for jj=1:size(treekids,1)
        nodetype = treekids(jj,3);
        if (jj > sl)
            if (nodetype)
                sym_index = treekids(jj,1);
                sym_params = symparams{sym_index};
                id1 = treekids(jj,1);
                c1 = Tree.nodeFeatures(:,id1);
                pm = p_fTanh(WsymencoV1*[c1;sym_params']+bsymencoV1);
                p = p_fTanh(WsymencoV2*pm+bsymencoV2);
                Tree.nodeFeatures(:, jj) = p;
            else
                id1 = treekids(jj,1);
                id2 = treekids(jj,2);
                c1 = Tree.nodeFeatures(:,id1);
                c2 = Tree.nodeFeatures(:,id2);
                pm = p_fTanh(WencoV1Left*c1 + WencoV1Right*c2 + bencoV1);
                p = p_fTanh(WencoV2*pm + bencoV2);
                Tree.nodeFeatures(:, jj) = p;
            end
        else
            box_f = symboxes(:,jj);
            p = p_fTanh(WencoBox*box_f + bencoBox);
            Tree.nodeFeatures(:,jj) = p;
        end    
    end
    

    re1 = p_fTanh(Wranen1*Tree.nodeFeatures(:,end)+branen1);
    re2 = Wranen2*re1+branen2;
    mu = re2(1:end/2);
    
    muCandidates(:,ii) = mu;
    treeCandidates{ii} = treekids;


end
    

