function gan_thetaD = initializeGanParametersD(size_params)

% Initialize parameters randomly based on layer sizes.
latentSize = size_params.latentSize;
hiddenSize = size_params.hiddenSize;
boxSize = size_params.boxSize;
symSize = size_params.symSize;

r  = sqrt(6) / sqrt(hiddenSize+latentSize+1);   % we'll choose weights uniformly from the interval [-r, r]

%classifier
gan_thetaD.Wdc1 = rand(hiddenSize, latentSize) * 2 * r - r;
gan_thetaD.Wdc2 = rand(hiddenSize, hiddenSize) * 2 * r - r;
gan_thetaD.bdc1 = zeros(hiddenSize,1);
gan_thetaD.bdc2 = zeros(hiddenSize,1);
gan_thetaD.Wscore = rand(1,hiddenSize) * 2 * r - r; 
gan_thetaD.bscore = 0;

gan_thetaD.WencoS1Left_D = rand(hiddenSize, latentSize) * 2 * r - r;
gan_thetaD.WencoS1Right_D = rand(hiddenSize, latentSize) * 2 * r - r;
gan_thetaD.WencoS2_D = rand(latentSize, hiddenSize) * 2 * r - r;
gan_thetaD.WencoBox_D = rand(latentSize, boxSize) * 2 * r - r;

gan_thetaD.bencoS1_D = zeros(hiddenSize, 1);
gan_thetaD.bencoS2_D = zeros(latentSize,1);
gan_thetaD.bencoBox_D = zeros(latentSize, 1);

gan_thetaD.WsymencoS1_D = rand(hiddenSize, latentSize+symSize) * 2 * r - r;
gan_thetaD.WsymencoS2_D = rand(latentSize, hiddenSize) * 2 * r - r;

gan_thetaD.bsymencoS1_D = zeros(hiddenSize,1);
gan_thetaD.bsymencoS2_D = zeros(latentSize,1);
