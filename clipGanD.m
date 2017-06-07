function Wbparas = clipGanD(theta, lp, size_params)
    
clip_theta = [theta.WencoS1Left_D(:) ; theta.WencoS1Right_D(:) ; theta.WencoS2_D(:); theta.bencoS1_D(:) ; theta.bencoS2_D(:);...
              theta.WsymencoS1_D(:); theta.WsymencoS2_D(:); theta.WencoBox_D(:);...
              theta.bsymencoS1_D(:); theta.bsymencoS2_D(:); theta.bencoBox_D(:);...
              theta.Wdc1(:); theta.Wdc2(:); theta.Wscore(:); theta.bdc1(:); theta.bdc2(:); theta.bscore(:)];

clip_theta(clip_theta>lp) = lp;
clip_theta(clip_theta<-lp) = -lp;
    
    
boxSize = size_params.boxSize;
hiddenSize = size_params.hiddenSize;
latentSize = size_params.latentSize;
symSize = size_params.symSize;

Wbparas = [];

lstart = 1; lend = hiddenSize*latentSize; Wbparas.WencoS1Left_D = reshape(clip_theta(lstart:lend), hiddenSize, latentSize);
lstart = lend+1; lend = lend+hiddenSize*latentSize;Wbparas.WencoS1Right_D = reshape(clip_theta(lstart:lend), hiddenSize, latentSize);
lstart = lend+1; lend = lend+latentSize*hiddenSize;Wbparas.WencoS2_D = reshape(clip_theta(lstart:lend), latentSize, hiddenSize);
lstart = lend+1; lend = lend+hiddenSize;Wbparas.bencoS1_D = clip_theta(lstart:lend);
lstart = lend+1; lend = lend+latentSize;Wbparas.bencoS2_D = clip_theta(lstart:lend);
lstart = lend+1; lend = lend+hiddenSize*(latentSize+symSize);Wbparas.WsymencoS1_D = reshape(clip_theta(lstart:lend), hiddenSize, latentSize+symSize);
lstart = lend+1; lend = lend+latentSize*hiddenSize;Wbparas.WsymencoS2_D = reshape(clip_theta(lstart:lend),latentSize, hiddenSize);
lstart = lend+1; lend = lend+latentSize*boxSize;Wbparas.WencoBox_D = reshape(clip_theta(lstart:lend),latentSize, boxSize);
lstart = lend+1; lend = lend+hiddenSize;Wbparas.bsymencoS1_D = clip_theta(lstart:lend);
lstart = lend+1; lend = lend+latentSize;Wbparas.bsymencoS2_D = clip_theta(lstart:lend);
lstart = lend+1; lend = lend+latentSize;Wbparas.bencoBox_D = clip_theta(lstart:lend);
lstart = lend+1; lend = lend+hiddenSize*latentSize;Wbparas.Wdc1 = reshape(clip_theta(lstart:lend), hiddenSize, latentSize);
lstart = lend+1; lend = lend+hiddenSize*hiddenSize;Wbparas.Wdc2 = reshape(clip_theta(lstart:lend), hiddenSize, hiddenSize);
lstart = lend+1; lend = lend+hiddenSize;Wbparas.Wscore = reshape(clip_theta(lstart:lend),1, hiddenSize);
lstart = lend+1; lend = lend+hiddenSize;Wbparas.bdc1 = clip_theta(lstart:lend);    
lstart = lend+1; lend = lend+hiddenSize;Wbparas.bdc2 = clip_theta(lstart:lend);
lstart = lend+1; lend = lend+1;Wbparas.bscore = clip_theta(lstart:lend);

end

