# GRASS: Generative Recursive Autoencoders for Shape Structures

By Jun Li, Kai Xu, Siddhartha Chaudhuri, Ersin Yumer, Hao Zhang, Leonidas Guibas

This repository contains the pre-trained models for box structure generation, as well as the training/testing code for the generation model.

Details of the work can be found [here](http://kevinkaixu.net/projects/grass.html)

## Citation

If you find our work useful in your research, please consider citing:
   
    @article {li_sig17,
        title = {GRASS: Generative Recursive Autoencoders for Shape Structures},
        author = {Jun Li and Kai Xu and Siddhartha Chaudhuri and Ersin Yumer and Hao Zhang and Leonidas Guibas},
        journal = {ACM Transactions on Graphics (Proc. of SIGGRAPH 2017)},
        volume = {36},
        number = {4},
        pages = {to appear},
        year = {2017}
    }

## Guide:

Training:
Run trainTestVAEGAN.m to train the vae-gan model on the provided chair data set.

Testing:
Use test_demo.m to generate shapes based on trained model. There is already a pre-trained model inside. The generated shape structures could be visulized like this:


For any questions, please contact Jun Li(jun.johnson.li@gmail.com) and Kai Xu(kevin.kai.xu@gmail.com).
