function [dis_grad, dis_loss, boxes_de, symparams_de] = forwardAndBackwardGanTreeD(thetaD,ifreal,only_score,data,treekids,kidssym,fTanh,fTanh_prime)

Wdc1 = thetaD.Wdc1;
Wdc2 = thetaD.Wdc2;
bdc1 = thetaD.bdc1;
bdc2 = thetaD.bdc2;
Wscore = thetaD.Wscore; 
bscore = thetaD.bscore;

gradWdc1 = zeros(size(Wdc1));
gradWdc2 = zeros(size(Wdc2));
gradbdc1 = zeros(size(bdc1));
gradbdc2 = zeros(size(bdc2));
gradWscore = zeros(size(Wscore)); 
gradbscore = zeros(size(bscore));

%encoder in D
WencoS1Left_D = thetaD.WencoS1Left_D;
WencoS1Right_D = thetaD.WencoS1Right_D;
WencoS2_D = thetaD.WencoS2_D;
WencoBox_D = thetaD.WencoBox_D;

gradWencoS1Left_D = zeros(size(WencoS1Left_D));
gradWencoS1Right_D = zeros(size(WencoS1Right_D));
gradWencoS2_D = zeros(size(WencoS2_D));
gradWencoBox_D = zeros(size(WencoBox_D));

bencoS1_D = thetaD.bencoS1_D;
bencoS2_D = thetaD.bencoS2_D;
bencoBox_D = thetaD.bencoBox_D;

gradbencoS1_D = zeros(size(bencoS1_D));
gradbencoS2_D = zeros(size(bencoS2_D));
gradbencoBox_D = zeros(size(bencoBox_D));

WsymencoS1_D = thetaD.WsymencoS1_D;
WsymencoS2_D = thetaD.WsymencoS2_D;

gradWsymencoS1_D = zeros(size(WsymencoS1_D));
gradWsymencoS2_D = zeros(size(WsymencoS2_D));

bsymencoS1_D = thetaD.bsymencoS1_D;
bsymencoS2_D = thetaD.bsymencoS2_D;

gradbsymencoS1_D = zeros(size(bsymencoS1_D));
gradbsymencoS2_D = zeros(size(bsymencoS2_D));

forwardTreeDis = tree2;
[~, sl] = size(data);
nodenums = size(treekids,1);
forwardTreeDis.pp = zeros(nodenums,1);
forwardTreeDis.kids = zeros(nodenums,2);
forwardTreeDis.nodeFeatures = zeros(length(thetaD.bencoS2_D),nodenums);
forwardTreeDis.nodeDelta_out = zeros(length(thetaD.bencoS2_D),nodenums);

%forward through the Dis tree
gLeafcount = 0;
gSymcount = 0;
gAssemcount = 0;
for jj = 1:size(treekids,1)
    nodetype = treekids(jj,3);
    % non-leaf node
    if (jj > sl)
        if (nodetype)
            sym_index = treekids(jj,1);
            sym_params = kidssym{sym_index};
            id1 = treekids(jj,1);
            c1 = forwardTreeDis.nodeFeatures(:,id1);
            pm = fTanh(WsymencoS1_D*[c1;sym_params']+bsymencoS1_D);
            parent = fTanh(WsymencoS2_D*pm+bsymencoS2_D);  
            forwardTreeDis.nodeFeatures(:, jj) = parent;
            label = [0; 0; 1];
            forwardTreeDis.nodeLabels(:,jj) = label;
            
            gSymcount = gSymcount + 1;
        else
            id1 = treekids(jj,1);
            id2 = treekids(jj,2);
            c1 = forwardTreeDis.nodeFeatures(:,id1);
            c2 = forwardTreeDis.nodeFeatures(:,id2);
            ym = fTanh(WencoS1Left_D*c1 + WencoS1Right_D*c2 + bencoS1_D);
            parent = fTanh(WencoS2_D*ym + bencoS2_D);
                
            forwardTreeDis.nodeFeatures(:, jj) = parent;
            label = [0; 1; 0];
            forwardTreeDis.nodeLabels(:,jj) = label;
            
            gAssemcount = gAssemcount + 1;
        end
    else %leaf node
        box_f = data(:,jj);
        parent = fTanh(WencoBox_D*box_f + bencoBox_D);
        forwardTreeDis.nodeFeatures(:,jj) = parent;
        label = [1;0;0];
        forwardTreeDis.nodeLabels(:,jj) = label;
        gLeafcount = gLeafcount + 1;
    end
end

nodenums = size(treekids,1);
shapecode = forwardTreeDis.nodeFeatures(:,nodenums);

f1 = fTanh(Wdc1*shapecode+bdc1);
f2 = fTanh(Wdc2*f1+bdc2);

fscore = Wscore*f2+bscore;

if (only_score)
    dis_grad = [];
    dis_loss = fscore;
    boxes_de = [];
    symparams_de = [];
    return;
end

%calculate loss and derivative
if(ifreal)
    dis_loss = -fscore;
    f2_de = fTanh_prime(f2).*(-Wscore'*ones(size(fscore)));
    Wscore_de = -1*ones(size(fscore))*f2';
    bscore_de = -1*ones(size(fscore));
else
    dis_loss = fscore;
    f2_de = fTanh_prime(f2).*(Wscore'*ones(size(fscore)));
    Wscore_de = ones(size(fscore))*f2';
    bscore_de = ones(size(fscore));    
end


gradWscore = gradWscore + Wscore_de;
gradbscore = gradbscore + bscore_de;

dis_grad.Wscore = gradWscore;
dis_grad.bscore = gradbscore;
%derivative
Wdc2_de = f2_de*f1';
bdc2_de = f2_de;
f1_de = fTanh_prime(f1).*(Wdc2'*f2_de);
Wdc1_de = f1_de*shapecode';
bdc1_de = f1_de;
gradWdc1 = gradWdc1 + Wdc1_de;
gradbdc1 = gradbdc1 + bdc1_de;
gradWdc2 = gradWdc2 + Wdc2_de;
gradbdc2 = gradbdc2 + bdc2_de;
dis_grad.Wdc1 = gradWdc1;
dis_grad.bdc1 = gradbdc1;
dis_grad.Wdc2 = gradWdc2;
dis_grad.bdc2 = gradbdc2;


shapecode_de = Wdc1'*f1_de;
forwardTreeDis.nodeDelta_out(:,nodenums) = shapecode_de;
%backward the Dis tree
boxes_de = data;
symparams_de = kidssym;
for jj = size(treekids,1):-1:1
    label = forwardTreeDis.nodeLabels(:,jj);
    % non-leaf node
    if (label(3))
        sym_index = treekids(jj,1);
        sym_params = kidssym{sym_index};
        id1 = treekids(jj,1);
        c1 = forwardTreeDis.nodeFeatures(:,id1);
        ym = fTanh(WsymencoS1_D*[c1;sym_params'] + bsymencoS1_D);
        yp = fTanh(WsymencoS2_D*ym + bsymencoS2_D);
        yp_de = fTanh_prime(yp).*forwardTreeDis.nodeDelta_out(:,jj);
        ym_de = fTanh_prime(ym).*(WsymencoS2_D'*yp_de);
        child_de = WsymencoS1_D'*ym_de;
        forwardTreeDis.nodeDelta_out(:,id1) = child_de(1:end-8);
        forwardTreeDis.nodeSymDelta_out(:,id1) = child_de(end-7:end);
        symparams_de{id1} = child_de(end-7:end)';
        
        gradWsymencoS1_D = gradWsymencoS1_D + ym_de*[c1;sym_params']';
        gradWsymencoS2_D = gradWsymencoS2_D + yp_de*ym';
        gradbsymencoS1_D = gradbsymencoS1_D + ym_de;
        gradbsymencoS2_D = gradbsymencoS2_D + yp_de;
        
    elseif (label(2))
        id1 = treekids(jj,1);
        id2 = treekids(jj,2);
        c1 = forwardTreeDis.nodeFeatures(:,id1);
        c2 = forwardTreeDis.nodeFeatures(:,id2);
        ym = fTanh(WencoS1Left_D*c1 + WencoS1Right_D*c2 + bencoS1_D);
        yp = fTanh(WencoS2_D*ym + bencoS2_D);
        yp_de = fTanh_prime(yp).*forwardTreeDis.nodeDelta_out(:,jj);
        ym_de = fTanh_prime(ym).*(WencoS2_D'*yp_de);
        forwardTreeDis.nodeDelta_out(:,id1) = WencoS1Left_D'*ym_de;
        forwardTreeDis.nodeDelta_out(:,id2) = WencoS1Right_D'*ym_de;
        
        gradWencoS1Left_D = gradWencoS1Left_D + ym_de*c1';
        gradWencoS1Right_D = gradWencoS1Right_D + ym_de*c2';
        gradWencoS2_D = gradWencoS2_D + yp_de*ym';
        gradbencoS1_D = gradbencoS1_D + ym_de;
        gradbencoS2_D = gradbencoS2_D + yp_de;
                
    elseif (label(1)) %leaf node
        box_f = data(:,jj);
        yp = fTanh(WencoBox_D*box_f + bencoBox_D);
        yp_de = fTanh_prime(yp).*forwardTreeDis.nodeDelta_out(:,jj); 
        gradWencoBox_D = gradWencoBox_D + yp_de*box_f';
        gradbencoBox_D = gradbencoBox_D + yp_de;
        box_f_de = WencoBox_D'*yp_de;
        boxes_de(:,jj) = box_f_de;
    end
end

if (gSymcount == 0); gSymcount = 1; end;

dis_grad.WencoS1Left_D = gradWencoS1Left_D/gAssemcount;
dis_grad.WencoS1Right_D = gradWencoS1Right_D/gAssemcount;
dis_grad.WencoS2_D = gradWencoS2_D/gAssemcount;
dis_grad.WencoBox_D = gradWencoBox_D/gLeafcount;

dis_grad.bencoS1_D = gradbencoS1_D/gAssemcount;
dis_grad.bencoS2_D = gradbencoS2_D/gAssemcount;
dis_grad.bencoBox_D = gradbencoBox_D/gLeafcount;

dis_grad.WsymencoS1_D = gradWsymencoS1_D/gSymcount;
dis_grad.WsymencoS2_D = gradWsymencoS2_D/gSymcount;

dis_grad.bsymencoS1_D = gradbsymencoS1_D/gSymcount;
dis_grad.bsymencoS2_D = gradbsymencoS2_D/gSymcount;

end

