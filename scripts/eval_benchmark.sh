#!/usr/bin/env bash

#MODEL_LIST=(VAE)
MODEL_LIST=(VAE AE Beta_VAE MS_SSIM_VAE S_VAE VQ_VAE RAE_L2 WAE Adversarial_AE Factor_VAE Beta_TC_VAE INFO_VAE_MMD VAE_LinNF VAE_IAF Disentangled_Beta_VAE VAMP RAE_GP IWAE CIWAE H_VAE RH_VAE MIWAE PIWAE VAEGAN)

for MODEL in ${MODEL_LIST[@]} ; do
    echo $MODEL
    sbatch --job-name=predict_${MODEL} scripts/eval_model_testsets.sh $MODEL
done