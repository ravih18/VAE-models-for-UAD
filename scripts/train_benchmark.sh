#!/usr/bin/env bash

MODEL_SHORT=(VAE AE Beta_VAE MS_SSIM_VAE VQ_VAE RAE_L2 WAE Adversarial_AE Factor_VAE Beta_TC_VAE INFO_VAE_MMD VAE_LinNF VAE_IAF Disentangled_Beta_VAE)
for MODEL in ${MODEL_SHORT[@]} ; do
    echo Training $MODEL
    sbatch --job-name=train_${MODEL} --time=10:00:00 scripts/train_model.sh $MODEL
done

MODEL_MED=(VAMP RAE_GP IWAE CIWAE H_VAE RH_VAE )
for MODEL in ${MODEL_MED[@]} ; do
    echo Training $MODEL
    sbatch --job-name=train_${MODEL} --time=20:00:00 scripts/train_model.sh $MODEL
done

MODEL_LONG=(MIWAE PIWAE VAEGAN)
for MODEL in ${MODEL_LONG[@]} ; do
    echo Training $MODEL
    sbatch --job-name=train_${MODEL} --time=50:00:00 --qos=qos_gpu-t4 scripts/train_model.sh $MODEL
done