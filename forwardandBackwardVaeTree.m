function [vae_grad, recon_loss, sym_loss, cat_loss, kld_loss ] = forwardandBackwardVaeTree(theta,data,randkids,kidssym,fTanh,fTanh_prime,alpha_cat)

vae_grad = [];
forwardTree = tree2;
[~, sl] = size(data);
nodenums = size(randkids,1);
forwardTree.pp = zeros(nodenums,1);
forwardTree.kids = zeros(nodenums,2);
forwardTree.nodeFeatures = zeros(length(theta.bencoV2),nodenums);
forwardTree.nodeDelta_out = zeros(length(theta.bencoV2),nodenums);

WencoV1Left = theta.WencoV1Left;
WencoV1Right = theta.WencoV1Right;
WencoV2 = theta.WencoV2;
bencoV1 = theta.bencoV1;
bencoV2 = theta.bencoV2;

vae_gradWencoV1Left = zeros(size(WencoV1Left));
vae_gradWencoV1Right = zeros(size(WencoV1Right));
vae_gradWencoV2 = zeros(size(WencoV2));
vae_gradbencoV1 = zeros(size(bencoV1));
vae_gradbencoV2 = zeros(size(bencoV2));

WsymencoV1 = theta.WsymencoV1;
WsymencoV2 = theta.WsymencoV2;
bsymencoV1 = theta.bsymencoV1;
bsymencoV2 = theta.bsymencoV2;

vae_gradWsymencoV1 = zeros(size(WsymencoV1));
vae_gradWsymencoV2 = zeros(size(WsymencoV2));
vae_gradbsymencoV1 = zeros(size(bsymencoV1));
vae_gradbsymencoV2 = zeros(size(bsymencoV2));

WencoBox = theta.WencoBox;
bencoBox = theta.bencoBox;

vae_gradWencoBox = zeros(size(WencoBox));
vae_gradbencoBox = zeros(size(bencoBox));

Wranen1 = theta.Wranen1;
Wranen2 = theta.Wranen2;
branen1 = theta.branen1;
branen2 = theta.branen2;

Wrande1 = theta.Wrande1;
Wrande2 = theta.Wrande2;
brande1 = theta.brande1;
brande2 = theta.brande2;

Wcat1 = theta.Wcat1;
Wcat2 = theta.Wcat2;
bcat1 = theta.bcat1;
bcat2 = theta.bcat2;

vae_gradWcat1 = zeros(size(Wcat1));
vae_gradWcat2 = zeros(size(Wcat2));
vae_gradbcat1 = zeros(size(bcat1));
vae_gradbcat2 = zeros(size(bcat2));

WdecoS1Left = theta.WdecoS1Left;
WdecoS1Right = theta.WdecoS1Right;
WdecoS2 = theta.WdecoS2;
bdecoS2 = theta.bdecoS2;
bdecoS1Left = theta.bdecoS1Left;
bdecoS1Right = theta.bdecoS1Right;

vae_gradWdecoS1Left = zeros(size(WdecoS1Left));
vae_gradWdecoS1Right = zeros(size(WdecoS1Right));
vae_gradWdecoS2 = zeros(size(WdecoS2));
vae_gradbdecoS2 = zeros(size(bdecoS2));
vae_gradbdecoS1Left = zeros(size(bdecoS1Left));
vae_gradbdecoS1Right = zeros(size(bdecoS1Right));

WdecoBox = theta.WdecoBox;
bdecoBox = theta.bdecoBox;

vae_gradWdecoBox = zeros(size(WdecoBox));
vae_gradbdecoBox = zeros(size(bdecoBox));

WsymdecoS2 = theta.WsymdecoS2;
WsymdecoS1 = theta.WsymdecoS1;
bsymdecoS2 = theta.bsymdecoS2;
bsymdecoS1 = theta.bsymdecoS1;

vae_gradWsymdecoS2 = zeros(size(WsymdecoS2));
vae_gradWsymdecoS1 = zeros(size(WsymdecoS1));
vae_gradbsymdecoS2 = zeros(size(bsymdecoS2));
vae_gradbsymdecoS1 = zeros(size(bsymdecoS1));


for i = 1:sl
    forwardTree.nodeLabels(:,i) = [1;0;0];
end

recon_loss = 0;
sym_loss = 0;
cat_loss = 0;

%forward through the tree
for jj = 1:size(randkids,1)
    nodetype = randkids(jj,3);
    % non-leaf node
    if (jj > sl)
        if (nodetype)
            sym_index = randkids(jj,1);
            sym_params = kidssym{sym_index};
            id1 = randkids(jj,1);
            c1 = forwardTree.nodeFeatures(:,id1);
            pm = fTanh(WsymencoV1*[c1;sym_params']+bsymencoV1);
            parent = fTanh(WsymencoV2*pm+bsymencoV2);  

            forwardTree.nodeFeatures(:, jj) = parent;
            forwardTree.pp(id1) = jj;
            forwardTree.kids(jj,:) = [id1 0];
            label = [0; 0; 1];
            forwardTree.nodeLabels(:,jj) = label;
        else
            id1 = randkids(jj,1);
            id2 = randkids(jj,2);
            c1 = forwardTree.nodeFeatures(:,id1);
            c2 = forwardTree.nodeFeatures(:,id2);
            ym = fTanh(WencoV1Left*c1 + WencoV1Right*c2 + bencoV1);
            parent = fTanh(WencoV2*ym + bencoV2);
                
            forwardTree.nodeFeatures(:, jj) = parent;
            forwardTree.pp(id1) = jj;
            forwardTree.pp(id2) = jj;
            forwardTree.kids(sl,:) = [id1 id2];
            label = [0; 1; 0];
            forwardTree.nodeLabels(:,jj) = label;
        end
    else %leaf node
        box_f = data(:,jj);
        parent = fTanh(WencoBox*box_f + bencoBox);
        forwardTree.nodeFeatures(:,jj) = parent;
        label = [1;0;0];
        forwardTree.nodeLabels(:,jj) = label;
    end
end

nodenums = size(randkids,1);
shapecode = forwardTree.nodeFeatures(:,nodenums);

re1 = fTanh(Wranen1*shapecode+branen1);
re2 = Wranen2*re1+branen2;

mu = re2(1:end/2);
logvar = re2(end/2+1:end);

sig = exp(logvar/2);
eps = randn(size(mu));
sample_z = mu + sig.*eps;

kld_loss = -1/2 * sum(sum(1+log(sig.^2)-mu.^2-sig.^2))*0.1;

rd2 = fTanh(Wrande2*sample_z+brande2);
rd1 = fTanh(Wrande1*rd2+brande1);


genTree = tree2;
[~, sl] = size(data);
nodenums = size(randkids,1);
genTree.pp = zeros(nodenums,1);
genTree.kids = zeros(nodenums,2);
genTree.nodeSymDelta_out = zeros(8,nodenums);
genTree.nodeFeatures = zeros(length(theta.bencoV2),nodenums);

genTree.nodeFeatures(:,nodenums) = rd1;

alpha_catSym = 1;
%forward Gen Tree
dLeafcount = 0;
for jj = nodenums:-1:1
    feature = genTree.nodeFeatures(:,jj);
    nodetype = randkids(jj,3);
    if (jj > sl)
        if (nodetype)
            label = [0;0;1];
            id1 = randkids(jj,1);
            ym = fTanh(WsymdecoS2*feature + bsymdecoS2);
            yp = fTanh(WsymdecoS1*ym + bsymdecoS1);  
            genTree.nodeFeatures(:,id1) = yp(1:end-8);
            reconSymparams = yp(end-7:end)';
            genTree.nodeSymFeatures{id1} = reconSymparams;
            gtSymparams = kidssym{id1};
            sym_loss = sym_loss + alpha_catSym*0.5*sum((reconSymparams-gtSymparams).^2);
            params_de = alpha_catSym*(reconSymparams-gtSymparams);
            genTree.nodeSymDelta_out(:,id1) = params_de;
            genTree.nodeLabels(:,jj) = label;
        else
            label = [0;1;0];
            id1 = randkids(jj,1);
            id2 = randkids(jj,2);
            ym = fTanh(WdecoS2*feature + bdecoS2);
            genTree.nodeFeatures(:,id1) = fTanh(WdecoS1Left*ym + bdecoS1Left);
            genTree.nodeFeatures(:,id2) = fTanh(WdecoS1Right*ym + bdecoS1Right);
            genTree.nodeLabels(:,jj) = label;
        end        
    else
        label = [1;0;0];
        yp = fTanh(WdecoBox*feature + bdecoBox);
        genTree.boxes(:,jj) = yp; 
        genTree.nodeLabels(:,jj) = label;      
        gt_box = data(:,jj);
        recon_loss = recon_loss + (1-alpha_cat)*0.5*sum((yp-gt_box).^2);
        
        yp_de = fTanh_prime(yp).*((1-alpha_cat)*(yp-gt_box));
        
        vae_gradWdecoBox = vae_gradWdecoBox + yp_de*feature';
        vae_gradbdecoBox = vae_gradbdecoBox + yp_de;
        dLeafcount = dLeafcount + 1;
        
        genTree.nodeDelta_out(:,jj) = WdecoBox'*yp_de;
    end
end

%backpropagation for Gen Tree
gSymcount = 0;
gAssemcount = 0;
for jj = 1:nodenums
    feature = genTree.nodeFeatures(:,jj);
    nodetype = randkids(jj,3);
    if (jj > sl)
        if (nodetype)
            id1 = randkids(jj,1);
            ym = fTanh(WsymdecoS2*feature + bsymdecoS2);
            yp = fTanh(WsymdecoS1*ym + bsymdecoS1);
            yp_de = fTanh_prime(yp).*[genTree.nodeDelta_out(:,id1);genTree.nodeSymDelta_out(:,id1)];
            ym_de = fTanh_prime(ym).*(WsymdecoS1'*yp_de);
            genTree.nodeDelta_out(:,jj) = WsymdecoS2'*ym_de;
            gSymcount = gSymcount + 1;
            
            vae_gradWsymdecoS2 = vae_gradWsymdecoS2 + ym_de*feature';
            vae_gradWsymdecoS1 = vae_gradWsymdecoS1 + yp_de*ym';
            vae_gradbsymdecoS2 = vae_gradbsymdecoS2 + ym_de;
            vae_gradbsymdecoS1 = vae_gradbsymdecoS1 + yp_de; 
        else
            id1 = randkids(jj,1);
            id2 = randkids(jj,2);          
            ym = fTanh(WdecoS2*feature + bdecoS2);
            leftchild = fTanh(WdecoS1Left*ym + bdecoS1Left);
            rightchild = fTanh(WdecoS1Right*ym + bdecoS1Right);

            leftchild_de = fTanh_prime(leftchild).*genTree.nodeDelta_out(:,id1);
            rightchild_de = fTanh_prime(rightchild).*genTree.nodeDelta_out(:,id2);

            ym_de = fTanh_prime(ym).*(WdecoS1Left'*leftchild_de+WdecoS1Right'*rightchild_de);
            genTree.nodeDelta_out(:,jj) = WdecoS2'*ym_de;
            gAssemcount = gAssemcount + 1;
            
            vae_gradWdecoS2 = vae_gradWdecoS2 + ym_de*feature';
            vae_gradWdecoS1Left = vae_gradWdecoS1Left + leftchild_de*ym';
            vae_gradWdecoS1Right = vae_gradWdecoS1Right + rightchild_de*ym';
            vae_gradbdecoS2 = vae_gradbdecoS2 + ym_de;
            vae_gradbdecoS1Left = vae_gradbdecoS1Left + leftchild_de;
            vae_gradbdecoS1Right = vae_gradbdecoS1Right + rightchild_de;
            
        end        
    end
    label = genTree.nodeLabels(:,jj);
    lp = fTanh(Wcat1*feature+bcat1);
    sm = softmax(Wcat2*lp + bcat2);
    lbl_sm = alpha_cat.*label.*log(sm);
    cat_prime = alpha_cat.*(sm-label);
    cat_loss = cat_loss-sum(lbl_sm);

    vae_gradWcat2 = vae_gradWcat2 + cat_prime*lp';
    vae_gradbcat2 = vae_gradbcat2 + cat_prime;
    lp_de = fTanh_prime(lp).*(Wcat2'*cat_prime);
    vae_gradWcat1 = vae_gradWcat1 + lp_de*feature';
    vae_gradbcat1 = vae_gradbcat1 + lp_de;

    genTree.nodeDelta_out(:,jj) = genTree.nodeDelta_out(:,jj) + Wcat1'*lp_de;
    
end

%backforward througth the tree
rd1_de = fTanh_prime(rd1).*genTree.nodeDelta_out(:,nodenums);
rd2_de = fTanh_prime(rd2).*(Wrande1'*rd1_de);
sample_z_de = Wrande2'*rd2_de;

vae_gradWrande2 = rd2_de*sample_z';
vae_gradbrande2 = rd2_de;
vae_gradWrande1 = rd1_de*rd2';
vae_gradbrande1 = rd1_de;

dermu_recon = sample_z_de;
derlogvar_recon = sample_z_de.*(eps.*sig/2);       
dermu_KL = mu;
derlogvar_KL = -(1 - exp(logvar))/2;

dermu = dermu_recon + dermu_KL*0.1;
derlogvar = derlogvar_recon + derlogvar_KL*0.1;

vae_gradWranen2 = [dermu;derlogvar]*re1';
vae_gradbranen2 = [dermu;derlogvar];
vae_gradWranen1 = (fTanh_prime(re1).*(Wranen2'*[dermu;derlogvar]))*shapecode';
vae_gradbranen1 = fTanh_prime(re1).*(Wranen2'*[dermu;derlogvar]);

forwardTree.nodeDelta_out(:,nodenums) = Wranen1'*(fTanh_prime(re1).*(Wranen2'*[dermu;derlogvar]));

%backpropagation for Encoder Tree
for jj = size(randkids,1):-1:1
    label = forwardTree.nodeLabels(:,jj);
    % non-leaf node
    if (label(3))
        sym_index = randkids(jj,1);
        sym_params = kidssym{sym_index};
        id1 = randkids(jj,1);
        c1 = forwardTree.nodeFeatures(:,id1);
        ym = fTanh(WsymencoV1*[c1;sym_params'] + bsymencoV1);
        yp = fTanh(WsymencoV2*ym + bsymencoV2);
        yp_de = fTanh_prime(yp).*forwardTree.nodeDelta_out(:,jj);
        ym_de = fTanh_prime(ym).*(WsymencoV2'*yp_de);
        child_de = WsymencoV1'*ym_de;
        forwardTree.nodeDelta_out(:,id1) = child_de(1:end-8);
        forwardTree.nodeSymDelta_out(:,id1) = child_de(end-7:end);
        
        vae_gradWsymencoV1 = vae_gradWsymencoV1 + ym_de*[c1;sym_params']';
        vae_gradWsymencoV2 = vae_gradWsymencoV2 + yp_de*ym';
        vae_gradbsymencoV1 = vae_gradbsymencoV1 + ym_de;
        vae_gradbsymencoV2 = vae_gradbsymencoV2 + yp_de;
        
    elseif (label(2))
        id1 = randkids(jj,1);
        id2 = randkids(jj,2);
        c1 = forwardTree.nodeFeatures(:,id1);
        c2 = forwardTree.nodeFeatures(:,id2);
        ym = fTanh(WencoV1Left*c1 + WencoV1Right*c2 + bencoV1);
        yp = fTanh(WencoV2*ym + bencoV2);
        yp_de = fTanh_prime(yp).*forwardTree.nodeDelta_out(:,jj);
        ym_de = fTanh_prime(ym).*(WencoV2'*yp_de);
        forwardTree.nodeDelta_out(:,id1) = WencoV1Left'*ym_de;
        forwardTree.nodeDelta_out(:,id2) = WencoV1Right'*ym_de;
        
        vae_gradWencoV1Left = vae_gradWencoV1Left + ym_de*c1';
        vae_gradWencoV1Right = vae_gradWencoV1Right + ym_de*c2';
        vae_gradWencoV2 = vae_gradWencoV2 + yp_de*ym';
        vae_gradbencoV1 = vae_gradbencoV1 + ym_de;
        vae_gradbencoV2 = vae_gradbencoV2 + yp_de;
                
    elseif (label(1)) %leaf node
        box_f = data(:,jj);
        yp = fTanh(WencoBox*box_f + bencoBox);
        yp_de = fTanh_prime(yp).*forwardTree.nodeDelta_out(:,jj); 
        vae_gradWencoBox = vae_gradWencoBox + yp_de*box_f';
        vae_gradbencoBox = vae_gradbencoBox + yp_de;
    end
end

if (gSymcount == 0); gSymcount = 1; end;

vae_grad.WencoV1Left = vae_gradWencoV1Left/gAssemcount;
vae_grad.WencoV1Right = vae_gradWencoV1Right/gAssemcount;
vae_grad.WencoV2 = vae_gradWencoV2/gAssemcount;
vae_grad.bencoV1 = vae_gradbencoV1/gAssemcount;
vae_grad.bencoV2 = vae_gradbencoV2/gAssemcount;

vae_grad.WdecoS1Left = vae_gradWdecoS1Left/gAssemcount;
vae_grad.WdecoS1Right = vae_gradWdecoS1Right/gAssemcount;
vae_grad.WdecoS2 = vae_gradWdecoS2/gAssemcount;
vae_grad.bdecoS2 = vae_gradbdecoS2/gAssemcount;
vae_grad.bdecoS1Left = vae_gradbdecoS1Left/gAssemcount;
vae_grad.bdecoS1Right = vae_gradbdecoS1Right/gAssemcount;

vae_grad.WsymencoV1 = vae_gradWsymencoV1/gSymcount;
vae_grad.WsymencoV2 = vae_gradWsymencoV2/gSymcount;
vae_grad.bsymencoV1 = vae_gradbsymencoV1/gSymcount;
vae_grad.bsymencoV2 = vae_gradbsymencoV2/gSymcount;

vae_grad.WsymdecoS2 = vae_gradWsymdecoS2/gSymcount;
vae_grad.WsymdecoS1 = vae_gradWsymdecoS1/gSymcount;
vae_grad.bsymdecoS2 = vae_gradbsymdecoS2/gSymcount;
vae_grad.bsymdecoS1 = vae_gradbsymdecoS1/gSymcount;

vae_grad.WencoBox = vae_gradWencoBox/dLeafcount;
vae_grad.bencoBox = vae_gradbencoBox/dLeafcount;
vae_grad.WdecoBox = vae_gradWdecoBox/dLeafcount;
vae_grad.bdecoBox = vae_gradbdecoBox/dLeafcount;

vae_grad.Wcat1 = vae_gradWcat1/nodenums;
vae_grad.Wcat2 = vae_gradWcat2/nodenums;
vae_grad.bcat1 = vae_gradbcat1/nodenums;
vae_grad.bcat2 = vae_gradbcat2/nodenums;

vae_grad.Wranen1 = vae_gradWranen1;
vae_grad.Wranen2 = vae_gradWranen2;
vae_grad.branen1 = vae_gradbranen1;
vae_grad.branen2 = vae_gradbranen2;

vae_grad.Wrande1 = vae_gradWrande1;
vae_grad.Wrande2 = vae_gradWrande2;
vae_grad.brande1 = vae_gradbrande1;
vae_grad.brande2 = vae_gradbrande2;

end

