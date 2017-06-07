function [ vae_theta ] = initializeVaeParameters(size_params)

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

end


