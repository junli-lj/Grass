function [costCat,costG,costD, gradG, gradD,genTree, validcount_fake, validcount_gen] = forwardAndbackproGenRAE(theta,noiseV,weightCat,data,randkids,kidssym,fTanh,fTanh_prime,fLeakyReLU,fLeakyReLU_prime)

[~, sl] = size(data);
nodenums = size(randkids,1);

%noise
Wn1 = theta.Wn1;
Wn2 = theta.Wn2;
bn1 = theta.bn1;
bn2 = theta.bn2;

%decoder
WdecoS1Left = theta.WdecoS1Left;
WdecoS1Right = theta.WdecoS1Right;
WdecoS2 = theta.WdecoS2;
WdecoBox = theta.WdecoBox;

%sym decoder
WsymdecoS2 = theta.WsymdecoS2;
WsymdecoS1 = theta.WsymdecoS1;

%node type
Wcat = theta.Wcat;
bcat = theta.bcat;

bdecoS2 = theta.bdecoS2;
bdecoS1Left = theta.bdecoS1Left;
bdecoS1Right = theta.bdecoS1Right;
bdecoBox = theta.bdecoBox;

bsymdecoS2 = theta.bsymdecoS2;
bsymdecoS1 = theta.bsymdecoS1;

%noise
gradWn1 = zeros(size(Wn1));
gradWn2 = zeros(size(Wn2));
gradbn1 = zeros(size(bn1));
gradbn2 = zeros(size(bn2));

%decoder
gradWdecoS1Left = zeros(size(WdecoS1Left));
gradWdecoS1Right = zeros(size(WdecoS1Right));
gradWdecoS2 = zeros(size(WdecoS2));
gradWdecoBox = zeros(size(WdecoBox));

%sym decoder
gradWsymdecoS2 = zeros(size(WsymdecoS2));
gradWsymdecoS1 = zeros(size(WsymdecoS1));

%node type
gradWcat = zeros(size(Wcat));
gradbcat = zeros(size(bcat));

gradbdecoS2 = zeros(size(bdecoS2));
gradbdecoS1Left = zeros(size(bdecoS1Left));
gradbdecoS1Right = zeros(size(bdecoS1Right));
gradbdecoBox = zeros(size(bdecoBox));

gradbsymdecoS2 = zeros(size(bsymdecoS2));
gradbsymdecoS1 = zeros(size(bsymdecoS1));

% noiseV = randn(50,1);

%forward G
genTree = forwardGenRAE(theta,noiseV,data,randkids,fTanh,fLeakyReLU);

%forward D
treeDisfoward = forwardDisRAE(theta,genTree.boxes,randkids,genTree.nodeSymFeatures,fTanh);

[costD, gradD, ~, validcount_fake] = backproDisRAE(theta,genTree.boxes,randkids,genTree.nodeSymFeatures,treeDisfoward,0,fTanh,fTanh_prime,fLeakyReLU,fLeakyReLU_prime);

[costG, ~, treeDis,validcount_gen] = backproDisRAE(theta,genTree.boxes,randkids,genTree.nodeSymFeatures,treeDisfoward,1,fTanh,fTanh_prime,fLeakyReLU,fLeakyReLU_prime);

%backforward G
gLeafcount = 0;
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
            yp_de = fTanh_prime(yp).*[genTree.nodeDelta_out(:,id1);treeDis.nodeSymDelta_out(:,id1)];
            ym_de = fTanh_prime(ym).*(WsymdecoS1'*yp_de);
            genTree.nodeDelta_out(:,jj) = WsymdecoS2'*ym_de;
            gSymcount = gSymcount + 1;
            
            gradWsymdecoS2 = gradWsymdecoS2 + ym_de*feature';
            gradWsymdecoS1 = gradWsymdecoS1 + yp_de*ym';
            gradbsymdecoS2 = gradbsymdecoS2 + ym_de;
            gradbsymdecoS1 = gradbsymdecoS1 + yp_de; 
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
            
            gradWdecoS2 = gradWdecoS2 + ym_de*feature';
            gradWdecoS1Left = gradWdecoS1Left + leftchild_de*ym';
            gradWdecoS1Right = gradWdecoS1Right + rightchild_de*ym';
            gradbdecoS2 = gradbdecoS2 + ym_de;
            gradbdecoS1Left = gradbdecoS1Left + leftchild_de;
            gradbdecoS1Right = gradbdecoS1Right + rightchild_de;
            
        end        
    else
        box_de = fTanh_prime(genTree.boxes(:,jj)).*treeDis.nodeBoxDelta_out(:,jj);
        genTree.nodeDelta_out(:,jj) = WdecoBox'*box_de;
        gLeafcount = gLeafcount + 1;
        gradWdecoBox = gradWdecoBox + box_de*feature';
        gradbdecoBox = gradbdecoBox + box_de;
    end
    label = genTree.nodeLabels(:,jj);
    sm = softmax(Wcat*feature + bcat);
    lbl_sm = 1/nodenums*weightCat.*label.*log(sm);
    cat_prime = 1/nodenums*weightCat.*(sm-label);
    genTree.nodeScores(jj) = -sum(lbl_sm);

    gradWcat = gradWcat + cat_prime*feature';
    gradbcat = gradbcat + cat_prime;

    genTree.nodeDelta_out(:,jj) = genTree.nodeDelta_out(:,jj) + Wcat'*cat_prime;
    
end

%backward noise
ym = fLeakyReLU(Wn1*noiseV + bn1);
fakerootfeature = fTanh(Wn2*ym + bn2);
yp_de = fTanh_prime(fakerootfeature).*genTree.nodeDelta_out(:,nodenums);
ym_de = fLeakyReLU_prime(ym).*(Wn2'*yp_de);

gradWn1 = gradWn1 + ym_de*noiseV';
gradWn2 = gradWn2 + yp_de*ym';
gradbn1 = gradbn1 + ym_de;
gradbn2 = gradbn2 + yp_de;

%noise
gradG.gradWn1 = gradWn1;
gradG.gradWn2 = gradWn2;
gradG.gradbn1 = gradbn1;
gradG.gradbn2 = gradbn2;

%decoder
gradG.gradWdecoS1Left = gradWdecoS1Left/gAssemcount;
gradG.gradWdecoS1Right = gradWdecoS1Right/gAssemcount;
gradG.gradWdecoS2 = gradWdecoS2/gAssemcount;
gradG.gradbdecoS2 = gradbdecoS2/gAssemcount;
gradG.gradbdecoS1Left = gradbdecoS1Left/gAssemcount;
gradG.gradbdecoS1Right = gradbdecoS1Right/gAssemcount;

%sym decoder
if gSymcount > 0
    gradG.gradWsymdecoS2 = gradWsymdecoS2/gSymcount;
    gradG.gradWsymdecoS1 = gradWsymdecoS1/gSymcount;
    gradG.gradbsymdecoS2 = gradbsymdecoS2/gSymcount;
    gradG.gradbsymdecoS1 = gradbsymdecoS1/gSymcount;
else
    gradG.gradWsymdecoS2 = gradWsymdecoS2;
    gradG.gradWsymdecoS1 = gradWsymdecoS1;
    gradG.gradbsymdecoS2 = gradbsymdecoS2;
    gradG.gradbsymdecoS1 = gradbsymdecoS1;
end
%node type
gradG.gradWcat = gradWcat/nodenums;
gradG.gradbcat = gradbcat/nodenums;

gradG.gradWdecoBox = gradWdecoBox/gLeafcount;
gradG.gradbdecoBox = gradbdecoBox/gLeafcount;

costCat = 0;
for jj = 1:nodenums
    costCat = costCat + genTree.nodeScores(jj);
end



