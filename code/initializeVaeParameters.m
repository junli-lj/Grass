function [ vae_theta ] = initializeVaeParameters(vae_preTrain,size_params)

% Initialize parameters randomly based on layer sizes.
latentSize = size_params.latentSize;
hiddenSize = size_params.hiddenSize;
boxSize = size_params.boxSize;
catSize = size_params.catSize;
symSize = size_params.symSize;

r  = sqrt(6) / sqrt(hiddenSize+hiddenSize+1);   % we'll choose weights uniformly from the interval [-r, r]

%sample layers
vae_theta.Wranen1 = rand(hiddenSize, latentSize) * 2 * r - r;
vae_theta.Wranen2 = rand(latentSize*2,hiddenSize) * 2 * r - r;
vae_theta.branen1 = zeros(hiddenSize,1);
vae_theta.branen2 = zeros(latentSize*2,1);

vae_theta.Wrande2 = rand(hiddenSize, latentSize) * 2 * r - r;
vae_theta.Wrande1 = rand(latentSize, hiddenSize) * 2 * r - r;
vae_theta.brande2 = zeros(hiddenSize,1);
vae_theta.brande1 = zeros(latentSize,1);

%VAE encoder
vae_theta.WencoV1Left = rand(hiddenSize, latentSize) * 2 * r - r;
vae_theta.WencoV1Right = rand(hiddenSize, latentSize) * 2 * r - r;
vae_theta.WencoV2 = rand(latentSize, hiddenSize) * 2 * r - r;
vae_theta.bencoV1 = zeros(hiddenSize,1);
vae_theta.bencoV2 = zeros(latentSize,1);


%VAE decoder
vae_theta.WdecoS1Left = rand(latentSize, hiddenSize) * 2 * r - r;
vae_theta.WdecoS1Right = rand(latentSize, hiddenSize) * 2 * r - r;
vae_theta.WdecoS2 = rand(hiddenSize, latentSize) * 2 * r - r;
vae_theta.bdecoS2 = zeros(hiddenSize, 1);
vae_theta.bdecoS1Left = zeros(latentSize, 1);
vae_theta.bdecoS1Right = zeros(latentSize,1);

%box encoder and decoder
vae_theta.WencoBox = rand(latentSize, boxSize) * 2 * r - r;
vae_theta.WdecoBox = rand(boxSize, latentSize) * 2 * r - r;
vae_theta.bencoBox = zeros(latentSize, 1);
vae_theta.bdecoBox = zeros(boxSize,1);

%sym encoder and decoder
vae_theta.WsymencoV1 = rand(hiddenSize, latentSize+symSize) * 2 * r - r;
vae_theta.WsymencoV2 = rand(latentSize, hiddenSize) * 2 * r - r;
vae_theta.WsymdecoS2 = rand(hiddenSize,latentSize) * 2 * r - r;
vae_theta.WsymdecoS1 = rand(latentSize+symSize, hiddenSize) * 2 * r - r;

vae_theta.bsymencoV1 = zeros(hiddenSize,1);
vae_theta.bsymencoV2 = zeros(latentSize,1);
vae_theta.bsymdecoS2 = zeros(hiddenSize,1);
vae_theta.bsymdecoS1 = zeros(latentSize+symSize,1);

%node type
vae_theta.Wcat1 = rand(hiddenSize, latentSize) * 2 * r - r;
vae_theta.Wcat2 = rand(catSize, hiddenSize) * 2 * r - r;
vae_theta.bcat1 = zeros(hiddenSize, 1);
vae_theta.bcat2 = zeros(catSize, 1);

if ~isempty(vae_preTrain)
    Wbparas = getW_vae(vae_preTrain, size_params);
    
    vae_theta.WencoV1Left = Wbparas.W1;
    vae_theta.WencoV1Right = Wbparas.W2;
    vae_theta.WencoV2 = Wbparas.W12;
    vae_theta.bencoV1 = Wbparas.b1;
    vae_theta.bencoV2 = Wbparas.b12;
    
    vae_theta.WdecoS1Left = Wbparas.W3;
    vae_theta.WdecoS1Right = Wbparas.W4;
    vae_theta.WdecoS2 = Wbparas.W34;
    vae_theta.bdecoS2 = Wbparas.b34;
    vae_theta.bdecoS1Left = Wbparas.b2;
    vae_theta.bdecoS1Right = Wbparas.b3;
    
    vae_theta.WencoBox = Wbparas.We;
    vae_theta.WdecoBox = Wbparas.Wd;
    vae_theta.bencoBox = Wbparas.be;
    vae_theta.bdecoBox = Wbparas.bd;
    
    vae_theta.WsymencoV1 = Wbparas.Wsym1;
    vae_theta.WsymencoV2 = Wbparas.Wsym2;
    vae_theta.WsymdecoS2 = Wbparas.Wsymd2;
    vae_theta.WsymdecoS1 = Wbparas.Wsymd1;

    vae_theta.bsymencoV1 = Wbparas.bsym1;
    vae_theta.bsymencoV2 = Wbparas.bsym2;
    vae_theta.bsymdecoS2 = Wbparas.bsymd2;
    vae_theta.bsymdecoS1 = Wbparas.bsymd1;
    
    vae_theta.Wcat1 = Wbparas.Wcat1;
    vae_theta.Wcat2 = Wbparas.Wcat2;
    vae_theta.bcat1 = Wbparas.bcat1;
    vae_theta.bcat2 = Wbparas.bcat2;
    
    vae_theta.Wranen1 = Wbparas.Wranen1;
    vae_theta.Wranen2 = Wbparas.Wranen2;
    vae_theta.branen1 = Wbparas.branen1;
    vae_theta.branen2 = Wbparas.branen2;

    vae_theta.Wrande2 = Wbparas.Wrande2;
    vae_theta.Wrande1 = Wbparas.Wrande1;
    vae_theta.brande2 = Wbparas.brande2;
    vae_theta.brande1 = Wbparas.brande1;
    
end

end

