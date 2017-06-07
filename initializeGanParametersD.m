function gan_thetaD = initializeGanParametersD(vae_preTrain, size_params)

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

%encoder and decoder
gan_thetaD.WencoS1Left_D = rand(hiddenSize, latentSize) * 2 * r - r;
gan_thetaD.WencoS1Right_D = rand(hiddenSize, latentSize) * 2 * r - r;
gan_thetaD.WencoS2_D = rand(latentSize, hiddenSize) * 2 * r - r;
gan_thetaD.WencoBox_D = rand(latentSize, boxSize) * 2 * r - r;

gan_thetaD.bencoS1_D = zeros(hiddenSize, 1);
gan_thetaD.bencoS2_D = zeros(latentSize,1);
gan_thetaD.bencoBox_D = zeros(latentSize, 1);

%sym encoder and decoder
gan_thetaD.WsymencoS1_D = rand(hiddenSize, latentSize+symSize) * 2 * r - r;
gan_thetaD.WsymencoS2_D = rand(latentSize, hiddenSize) * 2 * r - r;

gan_thetaD.bsymencoS1_D = zeros(hiddenSize,1);
gan_thetaD.bsymencoS2_D = zeros(latentSize,1);

if ~isempty(vae_preTrain)
    
    gan_thetaD.WencoS1Left_D = vae_preTrain.WencoV1Left;
    gan_thetaD.WencoS1Right_D = vae_preTrain.WencoV1Right;
    gan_thetaD.WencoS2_D = vae_preTrain.WencoV2;
    gan_thetaD.bencoS1_D = vae_preTrain.bencoV1;
    gan_thetaD.bencoS2_D = vae_preTrain.bencoV2;

    gan_thetaD.WencoBox_D = vae_preTrain.WencoBox;
    gan_thetaD.bencoBox_D = vae_preTrain.bencoBox;

    gan_thetaD.WsymencoS1_D = vae_preTrain.WsymencoV1;
    gan_thetaD.WsymencoS2_D = vae_preTrain.WsymencoV2;

    gan_thetaD.bsymencoS1_D = vae_preTrain.bsymencoV1;
    gan_thetaD.bsymencoS2_D = vae_preTrain.bsymencoV2;
end
         