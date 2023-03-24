#!/bin/bash
#SBATCH --output=logs/slurm_%j.log
#SBATCH --constraint=v100-32g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:40:00
#SBATCH --hint=nomultithread
#SBATCH --account=krk@v100


MAPS_DIR=/gpfswork/rech/krk/usy14zi/vae_benchmark/models/evaluated_models_256/MAPS_$1
echo $MAPS_DIR
# Predict on test AD
GROUPE=test_AD
CAPS_DIR=/gpfswork/rech/krk/commun/datasets/adni/caps/caps_pet_uniform/
PARTICIPANT_TSV=/gpfswork/rech/krk/usy14zi/vae_benchmark/adni_tsv/deep_learning_exp/test/AD_baseline.tsv

echo clinicadl predict $MAPS_DIR $GROUPE --caps_directory $CAPS_DIR --participants_tsv $PARTICIPANT_TSV --diagnoses AD --split 0 --selection_metrics loss --save_latent_tensor  --save_tensor  --overwrite
clinicadl   predict $MAPS_DIR $GROUPE \
            --caps_directory $CAPS_DIR \
            --participants_tsv $PARTICIPANT_TSV \
            --diagnoses AD \
            --split 0 \
            --selection_metrics loss \
            --save_latent_tensor  \
            --save_tensor \
            --overwrite

# Predict on test CN
GROUPE=test_CN
PARTICIPANT_TSV=/gpfswork/rech/krk/usy14zi/vae_benchmark/adni_tsv/deep_learning_exp/test/CN_baseline.tsv

echo clinicadl predict $GROUPE
clinicadl   predict $MAPS_DIR $GROUPE \
            --caps_directory $CAPS_DIR \
            --participants_tsv $PARTICIPANT_TSV \
            --diagnoses CN \
            --split 0 \
            --selection_metrics loss \
            --save_latent_tensor  \
            --save_tensor \
            --overwrite

# Predict on all hypometabolic CAPS
PATHOLOGY_LIST=(bvftd lvppa svppa nvppa pca)
for PATHOLOGY in ${PATHOLOGY_LIST[@]} ; do
    GROUPE=test_hypo_${PATHOLOGY}_30
    CAPS_DIR=/gpfswork/rech/krk/commun/datasets/adni/caps/hypometabolic_caps/caps_${PATHOLOGY}_30

    echo clinicadl predict $GROUPE
    clinicadl   predict $MAPS_DIR $GROUPE \
                --caps_directory $CAPS_DIR \
                --participants_tsv $PARTICIPANT_TSV \
                --diagnoses CN \
                --split 0 \
                --selection_metrics loss \
                --save_latent_tensor  \
                --save_tensor \
                --overwrite
    echo making tsv
    python python_scripts/healthiness_measurment.py $MAPS_DIR $PATHOLOGY 30
done

PERCENTAGE_LIST=(5 10 15 20 30 40 50 70)
for PERCENTAGE in ${PERCENTAGE_LIST[@]} ; do
    GROUPE=test_hypo_ad_${PERCENTAGE}
    CAPS_DIR=/gpfswork/rech/krk/commun/datasets/adni/caps/hypometabolic_caps/caps_ad_${PERCENTAGE}

    echo clinicadl predict $GROUPE
    clinicadl   predict $MAPS_DIR $GROUPE \
                --caps_directory $CAPS_DIR \
                --participants_tsv $PARTICIPANT_TSV \
                --diagnoses CN \
                --split 0 \
                --selection_metrics loss \
                --save_latent_tensor  \
                --save_tensor \
                --overwrite
    echo making tsv
    python python_scripts/healthiness_measurment.py $MAPS_DIR ad $PERCENTAGE
done