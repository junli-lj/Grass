function [gen_grad, gan_loss, cat_loss] = forwardAndBackwardGanTreeG(theta_vae,thetaD, noise, data,randkids,kidssym,fTanh,fTanh_prime,alpha_cat )

gen_grad = [];

Wrande1 = theta_vae.Wrande1;
Wrande2 = theta_vae.Wrande2;
brande1 = theta_vae.brande1;
brande2 = theta_vae.brande2;

Wcat1 = theta_vae.Wcat1;
Wcat2 = theta_vae.Wcat2;
bcat1 = theta_vae.bcat1;
bcat2 = theta_vae.bcat2;

vae_gradWcat1 = zeros(size(Wcat1));
vae_gradWcat2 = zeros(size(Wcat2));
vae_gradbcat1 = zeros(size(bcat1));
vae_gradbcat2 = zeros(size(bcat2));

WdecoS1Left = theta_vae.WdecoS1Left;
WdecoS1Right = theta_vae.WdecoS1Right;
WdecoS2 = theta_vae.WdecoS2;
bdecoS2 = theta_vae.bdecoS2;
bdecoS1Left = theta_vae.bdecoS1Left;
bdecoS1Right = theta_vae.bdecoS1Right;

vae_gradWdecoS1Left = zeros(size(WdecoS1Left));
vae_gradWdecoS1Right = zeros(size(WdecoS1Right));
vae_gradWdecoS2 = zeros(size(WdecoS2));
vae_gradbdecoS2 = zeros(size(bdecoS2));
vae_gradbdecoS1Left = zeros(size(bdecoS1Left));
vae_gradbdecoS1Right = zeros(size(bdecoS1Right));

WdecoBox = theta_vae.WdecoBox;
bdecoBox = theta_vae.bdecoBox;

vae_gradWdecoBox = zeros(size(WdecoBox));
vae_gradbdecoBox = zeros(size(bdecoBox));

WsymdecoS2 = theta_vae.WsymdecoS2;
WsymdecoS1 = theta_vae.WsymdecoS1;
bsymdecoS2 = theta_vae.bsymdecoS2;
bsymdecoS1 = theta_vae.bsymdecoS1;

vae_gradWsymdecoS2 = zeros(size(WsymdecoS2));
vae_gradWsymdecoS1 = zeros(size(WsymdecoS1));
vae_gradbsymdecoS2 = zeros(size(bsymdecoS2));
vae_gradbsymdecoS1 = zeros(size(bsymdecoS1));


rd2 = fTanh(Wrande2*noise+brande2);
rd1 = fTanh(Wrande1*rd2+brande1);

%generate 
genTree = tree2;
[~, sl] = size(data);
nodenums = size(randkids,1);
genTree.pp = zeros(nodenums,1);
genTree.kids = zeros(nodenums,2);
genTree.nodeSymDelta_out = zeros(8,nodenums);
genTree.nodeFeatures = zeros(length(theta_vae.bencoV2),nodenums);

genTree.nodeFeatures(:,nodenums) = rd1;

%forward Gen Tree
gen_data = data;
gen_kidssym = kidssym;

gLeafcount = 0;
gSymcount = 0;
gAssemcount = 0;

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
            params_de = reconSymparams-gtSymparams;
            genTree.nodeSymDelta_out(:,id1) = params_de;
            genTree.nodeLabels(:,jj) = label;
            gen_kidssym{id1} = reconSymparams;
            gSymcount = gSymcount + 1;
        else
            label = [0;1;0];
            id1 = randkids(jj,1);
            id2 = randkids(jj,2);
            ym = fTanh(WdecoS2*feature + bdecoS2);
            genTree.nodeFeatures(:,id1) = fTanh(WdecoS1Left*ym + bdecoS1Left);
            genTree.nodeFeatures(:,id2) = fTanh(WdecoS1Right*ym + bdecoS1Right);
            genTree.nodeLabels(:,jj) = label;
            gAssemcount = gAssemcount + 1;
        end        
    else
        label = [1;0;0];
        yp = fTanh(WdecoBox*feature + bdecoBox);
        genTree.boxes(:,jj) = yp; 
        genTree.nodeLabels(:,jj) = label;      
        gen_data(:,jj) = yp;
        gLeafcount = gLeafcount + 1;
    end
end

[~, gan_loss, boxes_de, symparams_de ] = forwardAndBackwardGanTreeD(thetaD,1,0, gen_data,randkids,gen_kidssym,fTanh,fTanh_prime);

for jj = sl:-1:1
    yp = genTree.boxes(:,jj); 
    feature = genTree.nodeFeatures(:,jj);
    yp_de = fTanh_prime(yp).*boxes_de(:,jj);
    vae_gradWdecoBox = vae_gradWdecoBox + yp_de*feature';
    vae_gradbdecoBox = vae_gradbdecoBox + yp_de;
    genTree.nodeDelta_out(:,jj) = WdecoBox'*yp_de;

end

cat_loss = 0;
%backpropagation for Gen Tree
for jj = 1:nodenums
    feature = genTree.nodeFeatures(:,jj);
    nodetype = randkids(jj,3);
    if (jj > sl)
        if (nodetype)
            id1 = randkids(jj,1);
            ym = fTanh(WsymdecoS2*feature + bsymdecoS2);
            yp = fTanh(WsymdecoS1*ym + bsymdecoS1);
            yp_de = fTanh_prime(yp).*[genTree.nodeDelta_out(:,id1);symparams_de{id1}'];
            ym_de = fTanh_prime(ym).*(WsymdecoS1'*yp_de);
            genTree.nodeDelta_out(:,jj) = WsymdecoS2'*ym_de;
            
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

rd1_de = fTanh_prime(rd1).*genTree.nodeDelta_out(:,nodenums);
rd2_de = fTanh_prime(rd2).*(Wrande1'*rd1_de);


vae_gradWrande1 = rd1_de*rd2';
vae_gradWrande2 = rd2_de*noise';
vae_gradbrande1 = rd1_de;
vae_gradbrande2 = rd2_de;

if (gSymcount == 0); gSymcount = 1; end;

gen_grad.Wrande1 = vae_gradWrande1;
gen_grad.Wrande2 = vae_gradWrande2;
gen_grad.brande1 = vae_gradbrande1;
gen_grad.brande2 = vae_gradbrande2;

gen_grad.Wcat1 = vae_gradWcat1/nodenums;
gen_grad.Wcat2 = vae_gradWcat2/nodenums;
gen_grad.bcat1 = vae_gradbcat1/nodenums;
gen_grad.bcat2 = vae_gradbcat2/nodenums;

gen_grad.WdecoS1Left = vae_gradWdecoS1Left/gAssemcount;
gen_grad.WdecoS1Right = vae_gradWdecoS1Right/gAssemcount;
gen_grad.WdecoS2 = vae_gradWdecoS2/gAssemcount;
gen_grad.bdecoS2 = vae_gradbdecoS2/gAssemcount;
gen_grad.bdecoS1Left = vae_gradbdecoS1Left/gAssemcount;
gen_grad.bdecoS1Right = vae_gradbdecoS1Right/gAssemcount;

gen_grad.WdecoBox = vae_gradWdecoBox/gLeafcount;
gen_grad.bdecoBox = vae_gradbdecoBox/gLeafcount;

gen_grad.WsymdecoS2 = vae_gradWsymdecoS2/gSymcount;
gen_grad.WsymdecoS1 = vae_gradWsymdecoS1/gSymcount;
gen_grad.bsymdecoS2 = vae_gradbsymdecoS2/gSymcount;
gen_grad.bsymdecoS1 = vae_gradbsymdecoS1/gSymcount;


end

