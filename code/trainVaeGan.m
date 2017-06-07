
addpath(genpath('../tools/'))
%load training data
trainingData = load('../data/trainingData_chair.mat');
%kernel size parmas
size_params.hiddenSize = 200;
size_params.latentSize = 80;
size_params.boxSize = 12;
size_params.catSize = 3;
size_params.symSize = 8;
alpha_cat = 0.2;

funcTanh = @norm1tanh;
funcTanh_prime = @norm1tanh_prime;

vae_theta = initializeVaeParameters(size_params);
gan_thetaD = initializeGanParametersD(size_params);
data = trainingData.data;
sampleNum = length(data);

%set initial learning rate 
dis_lr = 0.01;
en_lr = 0.2;
de_lr = 0.2;
cat_lr = 0.5;

gan_weight = 0.01;
vae_weight = 1;

D_iterations = 1; %iterations for D
subsetnum = 20;
subsampleNum = sampleNum/subsetnum;
for ii = 1:1500
    
    %after 500 epochs, adjust learning rate 
    if (ii > 500)
        en_lr = 0.1;
        de_lr = 0.1;
        cat_lr = 0.1;
    end
    
    %after 1200 epochs, adjust learning rate
    if (ii > 1200)
        en_lr = 0.01;
        de_lr = 0.01;
        cat_lr = 0.01;
    end
 
    reconLoss = 0;
    symLoss = 0;
    catLoss = 0;
    kldLoss = 0;
    
    train_index = randperm(sampleNum);
    
    [muCandidates,treeCandidates] = findCandidates(vae_theta, data, funcTanh);

    %generate fake data for GAN training
    fakeData = generateFakeData(vae_theta,gan_thetaD,muCandidates,treeCandidates,sampleNum,funcTanh,funcTanh_prime);

    for tt = 1:D_iterations
        for ss = 1:subsetnum
            startindex = (ss-1)*subsampleNum+1;
            p_subset = train_index(startindex:startindex+subsampleNum-1);
            realLoss = 0;
            fakeLoss = 0;

            %gradient for D
            gradWdc1 = zeros(size(gan_thetaD.Wdc1));
            gradWdc2 = zeros(size(gan_thetaD.Wdc2));
            gradbdc1 = zeros(size(gan_thetaD.bdc1));
            gradbdc2 = zeros(size(gan_thetaD.bdc2));
            gradWscore = zeros(size(gan_thetaD.Wscore)); 
            gradbscore = zeros(size(gan_thetaD.bscore));

            gradWencoS1Left_D = zeros(size(gan_thetaD.WencoS1Left_D));
            gradWencoS1Right_D = zeros(size(gan_thetaD.WencoS1Right_D));
            gradWencoS2_D = zeros(size(gan_thetaD.WencoS2_D));
            gradWencoBox_D = zeros(size(gan_thetaD.WencoBox_D));

            gradbencoS1_D = zeros(size(gan_thetaD.bencoS1_D));
            gradbencoS2_D = zeros(size(gan_thetaD.bencoS2_D));
            gradbencoBox_D = zeros(size(gan_thetaD.bencoBox_D));

            gradWsymencoS1_D = zeros(size(gan_thetaD.WsymencoS1_D));
            gradWsymencoS2_D = zeros(size(gan_thetaD.WsymencoS2_D));

            gradbsymencoS1_D = zeros(size(gan_thetaD.bsymencoS1_D));
            gradbsymencoS2_D = zeros(size(gan_thetaD.bsymencoS2_D));

            %real data for D
            parfor jj = 1:subsampleNum
                p_index = p_subset(jj);
                p_data = data;
                symboxes = p_data{p_index}.symshapes;
                treekids = p_data{p_index}.treekids;
                symparams = p_data{p_index}.symparams;

                [grad_dis_real, dis_loss_real, ~, ~ ] = forwardAndBackwardGanTreeD(gan_thetaD,1,0,symboxes,treekids,symparams,funcTanh,funcTanh_prime);

                realLoss = realLoss - dis_loss_real;
                gradWdc1 = gradWdc1 + grad_dis_real.Wdc1;
                gradWdc2 = gradWdc2 + grad_dis_real.Wdc2;
                gradbdc1 = gradbdc1 + grad_dis_real.bdc1;
                gradbdc2 = gradbdc2 + grad_dis_real.bdc2;
                gradWscore = gradWscore + grad_dis_real.Wscore; 
                gradbscore = gradbscore + grad_dis_real.bscore;

                gradWencoS1Left_D = gradWencoS1Left_D + grad_dis_real.WencoS1Left_D;
                gradWencoS1Right_D = gradWencoS1Right_D + grad_dis_real.WencoS1Right_D;
                gradWencoS2_D = gradWencoS2_D + grad_dis_real.WencoS2_D;
                gradWencoBox_D = gradWencoBox_D + grad_dis_real.WencoBox_D;

                gradbencoS1_D = gradbencoS1_D + grad_dis_real.bencoS1_D;
                gradbencoS2_D = gradbencoS2_D + grad_dis_real.bencoS2_D;
                gradbencoBox_D = gradbencoBox_D + grad_dis_real.bencoBox_D;

                gradWsymencoS1_D = gradWsymencoS1_D + grad_dis_real.WsymencoS1_D;
                gradWsymencoS2_D = gradWsymencoS2_D + grad_dis_real.WsymencoS2_D;

                gradbsymencoS1_D = gradbsymencoS1_D + grad_dis_real.bsymencoS1_D;
                gradbsymencoS2_D = gradbsymencoS2_D + grad_dis_real.bsymencoS2_D;

            end
            
            %update discriminator
            gan_thetaD.Wdc1 = gan_thetaD.Wdc1 - dis_lr * gradWdc1/subsetnum;
            gan_thetaD.Wdc2 = gan_thetaD.Wdc2 - dis_lr * gradWdc2/subsetnum;
            gan_thetaD.bdc1 = gan_thetaD.bdc1 - dis_lr * gradbdc1/subsetnum;
            gan_thetaD.bdc2 = gan_thetaD.bdc2 - dis_lr * gradbdc2/subsetnum;
            gan_thetaD.Wscore = gan_thetaD.Wscore - dis_lr * gradWscore/subsetnum; 
            gan_thetaD.bscore = gan_thetaD.bscore - dis_lr * gradbscore/subsetnum;

            gan_thetaD.WencoS1Left_D = gan_thetaD.WencoS1Left_D - dis_lr * gradWencoS1Left_D/subsetnum;
            gan_thetaD.WencoS1Right_D = gan_thetaD.WencoS1Right_D - dis_lr * gradWencoS1Right_D/subsetnum;
            gan_thetaD.WencoS2_D = gan_thetaD.WencoS2_D - dis_lr * gradWencoS2_D/subsetnum;
            gan_thetaD.WencoBox_D = gan_thetaD.WencoBox_D - dis_lr * gradWencoBox_D/subsetnum;

            gan_thetaD.bencoS1_D = gan_thetaD.bencoS1_D - dis_lr * gradbencoS1_D/subsetnum;
            gan_thetaD.bencoS2_D = gan_thetaD.bencoS2_D - dis_lr * gradbencoS2_D/subsetnum;
            gan_thetaD.bencoBox_D = gan_thetaD.bencoBox_D - dis_lr * gradbencoBox_D/subsetnum;

            gan_thetaD.WsymencoS1_D = gan_thetaD.WsymencoS1_D - dis_lr * gradWsymencoS1_D/subsetnum;
            gan_thetaD.WsymencoS2_D = gan_thetaD.WsymencoS2_D - dis_lr * gradWsymencoS2_D/subsetnum;

            gan_thetaD.bsymencoS1_D = gan_thetaD.bsymencoS1_D - dis_lr * gradbsymencoS1_D/subsetnum;
            gan_thetaD.bsymencoS2_D = gan_thetaD.bsymencoS2_D - dis_lr * gradbsymencoS2_D/subsetnum;

            gan_thetaD = clipGanD(gan_thetaD,0.1,size_params);
            
            gradWdc1 = zeros(size(gan_thetaD.Wdc1));
            gradWdc2 = zeros(size(gan_thetaD.Wdc2));
            gradbdc1 = zeros(size(gan_thetaD.bdc1));
            gradbdc2 = zeros(size(gan_thetaD.bdc2));
            gradWscore = zeros(size(gan_thetaD.Wscore)); 
            gradbscore = zeros(size(gan_thetaD.bscore));

            gradWencoS1Left_D = zeros(size(gan_thetaD.WencoS1Left_D));
            gradWencoS1Right_D = zeros(size(gan_thetaD.WencoS1Right_D));
            gradWencoS2_D = zeros(size(gan_thetaD.WencoS2_D));
            gradWencoBox_D = zeros(size(gan_thetaD.WencoBox_D));

            gradbencoS1_D = zeros(size(gan_thetaD.bencoS1_D));
            gradbencoS2_D = zeros(size(gan_thetaD.bencoS2_D));
            gradbencoBox_D = zeros(size(gan_thetaD.bencoBox_D));

            gradWsymencoS1_D = zeros(size(gan_thetaD.WsymencoS1_D));
            gradWsymencoS2_D = zeros(size(gan_thetaD.WsymencoS2_D));

            gradbsymencoS1_D = zeros(size(gan_thetaD.bsymencoS1_D));
            gradbsymencoS2_D = zeros(size(gan_thetaD.bsymencoS2_D));

            %fakeData for D
            parfor jj = 1:subsampleNum
                p_index = p_subset(jj);
                p_data = fakeData;
                symboxes = p_data{p_index}.symshapes;
                treekids = p_data{p_index}.treekids;
                symparams = p_data{p_index}.symparams;

                [grad_dis_fake, dis_loss_fake, ~, ~ ] = forwardAndBackwardGanTreeD(gan_thetaD,0,0,symboxes,treekids,symparams,funcTanh,funcTanh_prime);

                fakeLoss = fakeLoss + dis_loss_fake;
                gradWdc1 = gradWdc1 + grad_dis_fake.Wdc1;
                gradWdc2 = gradWdc2 + grad_dis_fake.Wdc2;
                gradbdc1 = gradbdc1 + grad_dis_fake.bdc1;
                gradbdc2 = gradbdc2 + grad_dis_fake.bdc2;
                gradWscore = gradWscore + grad_dis_fake.Wscore; 
                gradbscore = gradbscore + grad_dis_fake.bscore;

                gradWencoS1Left_D = gradWencoS1Left_D + grad_dis_fake.WencoS1Left_D;
                gradWencoS1Right_D = gradWencoS1Right_D + grad_dis_fake.WencoS1Right_D;
                gradWencoS2_D = gradWencoS2_D + grad_dis_fake.WencoS2_D;
                gradWencoBox_D = gradWencoBox_D + grad_dis_fake.WencoBox_D;

                gradbencoS1_D = gradbencoS1_D + grad_dis_fake.bencoS1_D;
                gradbencoS2_D = gradbencoS2_D + grad_dis_fake.bencoS2_D;
                gradbencoBox_D = gradbencoBox_D + grad_dis_fake.bencoBox_D;

                gradWsymencoS1_D = gradWsymencoS1_D + grad_dis_fake.WsymencoS1_D;
                gradWsymencoS2_D = gradWsymencoS2_D + grad_dis_fake.WsymencoS2_D;

                gradbsymencoS1_D = gradbsymencoS1_D + grad_dis_fake.bsymencoS1_D;
                gradbsymencoS2_D = gradbsymencoS2_D + grad_dis_fake.bsymencoS2_D;

            end

            %update discriminator
            gan_thetaD.Wdc1 = gan_thetaD.Wdc1 - dis_lr * gradWdc1/subsetnum;
            gan_thetaD.Wdc2 = gan_thetaD.Wdc2 - dis_lr * gradWdc2/subsetnum;
            gan_thetaD.bdc1 = gan_thetaD.bdc1 - dis_lr * gradbdc1/subsetnum;
            gan_thetaD.bdc2 = gan_thetaD.bdc2 - dis_lr * gradbdc2/subsetnum;
            gan_thetaD.Wscore = gan_thetaD.Wscore - dis_lr * gradWscore/subsetnum; 
            gan_thetaD.bscore = gan_thetaD.bscore - dis_lr * gradbscore/subsetnum;

            gan_thetaD.WencoS1Left_D = gan_thetaD.WencoS1Left_D - dis_lr * gradWencoS1Left_D/subsetnum;
            gan_thetaD.WencoS1Right_D = gan_thetaD.WencoS1Right_D - dis_lr * gradWencoS1Right_D/subsetnum;
            gan_thetaD.WencoS2_D = gan_thetaD.WencoS2_D - dis_lr * gradWencoS2_D/subsetnum;
            gan_thetaD.WencoBox_D = gan_thetaD.WencoBox_D - dis_lr * gradWencoBox_D/subsetnum;

            gan_thetaD.bencoS1_D = gan_thetaD.bencoS1_D - dis_lr * gradbencoS1_D/subsetnum;
            gan_thetaD.bencoS2_D = gan_thetaD.bencoS2_D - dis_lr * gradbencoS2_D/subsetnum;
            gan_thetaD.bencoBox_D = gan_thetaD.bencoBox_D - dis_lr * gradbencoBox_D/subsetnum;

            gan_thetaD.WsymencoS1_D = gan_thetaD.WsymencoS1_D - dis_lr * gradWsymencoS1_D/subsetnum;
            gan_thetaD.WsymencoS2_D = gan_thetaD.WsymencoS2_D - dis_lr * gradWsymencoS2_D/subsetnum;

            gan_thetaD.bsymencoS1_D = gan_thetaD.bsymencoS1_D - dis_lr * gradbsymencoS1_D/subsetnum;
            gan_thetaD.bsymencoS2_D = gan_thetaD.bsymencoS2_D - dis_lr * gradbsymencoS2_D/subsetnum;

            gan_thetaD = clipGanD(gan_thetaD,0.1,size_params);

        end
    end
    
    %Train G and VAE
    for ss = 1:subsetnum
        startindex = (ss-1)*subsampleNum+1;
        p_subset = train_index(startindex:startindex+subsampleNum-1);
        
        %gradient for G/decoder
        gradWrande1 = zeros(size(vae_theta.Wrande1));
        gradWrande2 = zeros(size(vae_theta.Wrande2));
        gradbrande1 = zeros(size(vae_theta.brande1));
        gradbrande2 = zeros(size(vae_theta.brande2));

        gradWdecoS1Left = zeros(size(vae_theta.WdecoS1Left));
        gradWdecoS1Right = zeros(size(vae_theta.WdecoS1Right));
        gradbdecoS1Left = zeros(size(vae_theta.bdecoS1Left));
        gradbdecoS1Right = zeros(size(vae_theta.bdecoS1Right));
        gradWdecoS2 = zeros(size(vae_theta.WdecoS2));
        gradbdecoS2 = zeros(size(vae_theta.bdecoS2));

        gradWdecoBox = zeros(size(vae_theta.WdecoBox));
        gradbdecoBox = zeros(size(vae_theta.bdecoBox));

        gradWsymdecoS2 = zeros(size(vae_theta.WsymdecoS2));
        gradWsymdecoS1 = zeros(size(vae_theta.WsymdecoS1));
        gradbsymdecoS2 = zeros(size(vae_theta.bsymdecoS2));
        gradbsymdecoS1 = zeros(size(vae_theta.bsymdecoS1));

        gradWcat1 = zeros(size(vae_theta.Wcat1));
        gradbcat1 = zeros(size(vae_theta.bcat1));
        gradWcat2 = zeros(size(vae_theta.Wcat2));
        gradbcat2 = zeros(size(vae_theta.bcat2));

        %gradient for encoder
        gradWranen1 = zeros(size(vae_theta.Wranen1));
        gradWranen2 = zeros(size(vae_theta.Wranen2));
        gradbranen1 = zeros(size(vae_theta.branen1));
        gradbranen2 = zeros(size(vae_theta.branen2));

        gradWencoV1Left = zeros(size(vae_theta.WencoV1Left));
        gradWencoV1Right = zeros(size(vae_theta.WencoV1Right));
        gradWencoV2 = zeros(size(vae_theta.WencoV2));
        gradbencoV1 = zeros(size(vae_theta.bencoV1));
        gradbencoV2 = zeros(size(vae_theta.bencoV2));

        gradWsymencoV1 = zeros(size(vae_theta.WsymencoV1));
        gradWsymencoV2 = zeros(size(vae_theta.WsymencoV2));
        gradbsymencoV1 = zeros(size(vae_theta.bsymencoV1));
        gradbsymencoV2 = zeros(size(vae_theta.bsymencoV2));

        gradWencoBox = zeros(size(vae_theta.WencoBox));
        gradbencoBox = zeros(size(vae_theta.bencoBox));

        parfor jj = 1:subsampleNum
            p_index = p_subset(jj);
            p_data = data;
            symboxes = p_data{p_index}.symshapes;
            treekids = p_data{p_index}.treekids;
            symparams = p_data{p_index}.symparams;
            %vae
            [grad_vae, recon_loss, sym_loss, cat_loss, kld_loss ] = forwardandBackwardVaeTree(vae_theta, symboxes, treekids, symparams,funcTanh, funcTanh_prime,alpha_cat);
            
            fake_symboxes = fakeData{jj}.symshapes;
            fake_treekids = fakeData{jj}.treekids;
            fake_symparams = fakeData{jj}.symparams;
            noise = fakeData{jj}.noise;
            %gan
            [grad_gan, gan_loss, gan_cat_loss] = forwardAndBackwardGanTreeG(vae_theta,gan_thetaD, noise,fake_symboxes,fake_treekids,fake_symparams,funcTanh,funcTanh_prime,alpha_cat);

            reconLoss = reconLoss + recon_loss;
            symLoss = symLoss + sym_loss;
            catLoss = catLoss + cat_loss;
            kldLoss = kldLoss + kld_loss;

            gradWencoV1Left = gradWencoV1Left + grad_vae.WencoV1Left;
            gradWencoV1Right = gradWencoV1Right + grad_vae.WencoV1Right;
            gradWencoV2 = gradWencoV2 + grad_vae.WencoV2;
            gradbencoV1 = gradbencoV1 + grad_vae.bencoV1;
            gradbencoV2 = gradbencoV2 + grad_vae.bencoV2;

            gradWsymencoV1 = gradWsymencoV1 + grad_vae.WsymencoV1;
            gradWsymencoV2 = gradWsymencoV2 + grad_vae.WsymencoV2;
            gradbsymencoV1 = gradbsymencoV1 + grad_vae.bsymencoV1;
            gradbsymencoV2 = gradbsymencoV2 + grad_vae.bsymencoV2;

            gradWencoBox = gradWencoBox + grad_vae.WencoBox;
            gradbencoBox = gradbencoBox + grad_vae.bencoBox;

            gradWranen1 = gradWranen1 + grad_vae.Wranen1;
            gradWranen2 = gradWranen2 + grad_vae.Wranen2;
            gradbranen1 = gradbranen1 + grad_vae.branen1;
            gradbranen2 = gradbranen2 + grad_vae.branen2;

            gradWrande1 = gradWrande1 + vae_weight*grad_vae.Wrande1 + gan_weight*grad_gan.Wrande1;
            gradWrande2 = gradWrande2 + vae_weight*grad_vae.Wrande2 + gan_weight*grad_gan.Wrande2;
            gradbrande1 = gradbrande1 + vae_weight*grad_vae.brande1 + gan_weight*grad_gan.brande1;
            gradbrande2 = gradbrande2 + vae_weight*grad_vae.brande2 + gan_weight*grad_gan.brande2;

            gradWdecoS1Left = gradWdecoS1Left + vae_weight*grad_vae.WdecoS1Left + gan_weight*grad_gan.WdecoS1Left;
            gradWdecoS1Right = gradWdecoS1Right + vae_weight*grad_vae.WdecoS1Right + gan_weight*grad_gan.WdecoS1Right;
            gradWdecoS2 = gradWdecoS2 + vae_weight*grad_vae.WdecoS2 + gan_weight*grad_gan.WdecoS2;
            gradbdecoS2 = gradbdecoS2 + vae_weight*grad_vae.bdecoS2 + gan_weight*grad_gan.bdecoS2;
            gradbdecoS1Left = gradbdecoS1Left + vae_weight*grad_vae.bdecoS1Left + gan_weight*grad_gan.bdecoS1Left;
            gradbdecoS1Right = gradbdecoS1Right + vae_weight*grad_vae.bdecoS1Right + gan_weight*grad_gan.bdecoS1Right;

            gradWdecoBox = gradWdecoBox + vae_weight*grad_vae.WdecoBox + gan_weight*grad_gan.WdecoBox;
            gradbdecoBox = gradbdecoBox + vae_weight*grad_vae.bdecoBox + gan_weight*grad_gan.bdecoBox;

            gradWsymdecoS2 = gradWsymdecoS2 + vae_weight*grad_vae.WsymdecoS2 + gan_weight*grad_gan.WsymdecoS2;
            gradWsymdecoS1 = gradWsymdecoS1 + vae_weight*grad_vae.WsymdecoS1 + gan_weight*grad_gan.WsymdecoS1;
            gradbsymdecoS2 = gradbsymdecoS2 + vae_weight*grad_vae.bsymdecoS2 + gan_weight*grad_gan.bsymdecoS2;
            gradbsymdecoS1 = gradbsymdecoS1 + vae_weight*grad_vae.bsymdecoS1 + gan_weight*grad_gan.bsymdecoS1;

            gradWcat1 = gradWcat1 + vae_weight*grad_vae.Wcat1 + gan_weight*grad_gan.Wcat1;
            gradbcat1 = gradbcat1 + vae_weight*grad_vae.bcat1 + gan_weight*grad_gan.bcat1;
            gradWcat2 = gradWcat2 + vae_weight*grad_vae.Wcat2 + gan_weight*grad_gan.Wcat2;
            gradbcat2 = gradbcat2 + vae_weight*grad_vae.bcat2 + gan_weight*grad_gan.bcat2;

        end

        %updata encoder network
        vae_theta.WencoV1Left = vae_theta.WencoV1Left - en_lr*gradWencoV1Left/subsampleNum;
        vae_theta.WencoV1Right = vae_theta.WencoV1Right - en_lr*gradWencoV1Right/subsampleNum;
        vae_theta.WencoV2 = vae_theta.WencoV2 - en_lr*gradWencoV2/subsampleNum;
        vae_theta.bencoV1 = vae_theta.bencoV1 - en_lr*gradbencoV1/subsampleNum;
        vae_theta.bencoV2 = vae_theta.bencoV2 - en_lr*gradbencoV2/subsampleNum;

        vae_theta.WsymencoV1 = vae_theta.WsymencoV1 - en_lr*gradWsymencoV1/subsampleNum;
        vae_theta.WsymencoV2 = vae_theta.WsymencoV2 - en_lr*gradWsymencoV2/subsampleNum;
        vae_theta.bsymencoV1 = vae_theta.bsymencoV1 - en_lr*gradbsymencoV1/subsampleNum;
        vae_theta.bsymencoV2 = vae_theta.bsymencoV2 - en_lr*gradbsymencoV2/subsampleNum;

        vae_theta.WencoBox = vae_theta.WencoBox - en_lr*gradWencoBox/subsampleNum;
        vae_theta.bencoBox = vae_theta.bencoBox - en_lr*gradbencoBox/subsampleNum;

        vae_theta.Wranen1 = vae_theta.Wranen1 - en_lr*gradWranen1/subsampleNum;
        vae_theta.Wranen2 = vae_theta.Wranen2 - en_lr*gradWranen2/subsampleNum;
        vae_theta.branen1 = vae_theta.branen1 - en_lr*gradbranen1/subsampleNum;
        vae_theta.branen2 = vae_theta.branen2 - en_lr*gradbranen2/subsampleNum;
        
        %update G/decoder network
        vae_theta.Wrande1 = vae_theta.Wrande1 - de_lr*gradWrande1/subsampleNum;
        vae_theta.Wrande2 = vae_theta.Wrande2 - de_lr*gradWrande2/subsampleNum;
        vae_theta.brande1 = vae_theta.brande1 - de_lr*gradbrande1/subsampleNum;
        vae_theta.brande2 = vae_theta.brande2 - de_lr*gradbrande2/subsampleNum;

        vae_theta.WdecoS1Left = vae_theta.WdecoS1Left - de_lr*gradWdecoS1Left/subsampleNum;
        vae_theta.WdecoS1Right = vae_theta.WdecoS1Right - de_lr*gradWdecoS1Right/subsampleNum;
        vae_theta.WdecoS2 = vae_theta.WdecoS2 - de_lr*gradWdecoS2/subsampleNum;
        vae_theta.bdecoS2 = vae_theta.bdecoS2 - de_lr*gradbdecoS2/subsampleNum;
        vae_theta.bdecoS1Left = vae_theta.bdecoS1Left - de_lr*gradbdecoS1Left/subsampleNum;
        vae_theta.bdecoS1Right = vae_theta.bdecoS1Right - de_lr*gradbdecoS1Right/subsampleNum;

        vae_theta.WsymdecoS2 = vae_theta.WsymdecoS2 - de_lr*gradWsymdecoS2/subsampleNum;
        vae_theta.WsymdecoS1 = vae_theta.WsymdecoS1 - de_lr*gradWsymdecoS1/subsampleNum;
        vae_theta.bsymdecoS2 = vae_theta.bsymdecoS2 - de_lr*gradbsymdecoS2/subsampleNum;
        vae_theta.bsymdecoS1 = vae_theta.bsymdecoS1 - de_lr*gradbsymdecoS1/subsampleNum;

        vae_theta.WdecoBox = vae_theta.WdecoBox - de_lr*gradWdecoBox/subsampleNum;
        vae_theta.bdecoBox = vae_theta.bdecoBox - de_lr*gradbdecoBox/subsampleNum;

        vae_theta.Wcat1 = vae_theta.Wcat1 - cat_lr*gradWcat1/subsampleNum;
        vae_theta.bcat1 = vae_theta.bcat1 - cat_lr*gradbcat1/subsampleNum;
        vae_theta.Wcat2 = vae_theta.Wcat2 - cat_lr*gradWcat2/subsampleNum;
        vae_theta.bcat2 = vae_theta.bcat2 - cat_lr*gradbcat2/subsampleNum;

    end
        
    fprintf('DisLoss_real: %.4f GenLoss_fake: %.4f \n',realLoss/sampleNum,fakeLoss/sampleNum);
    fprintf('reconLoss: %.4f symLoss: %.4f  catLoss: %.4f kldLoss: %.4f \n',reconLoss/sampleNum,symLoss/sampleNum, catLoss/sampleNum, kldLoss/sampleNum);    

end

save('vae_gan.mat', 'vae_theta');


