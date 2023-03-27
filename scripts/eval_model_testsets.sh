#!/bin/bash
#SBATCH --output=logs/slurm_%j.log
#SBATCH --constraint=v100-32g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:40:00
#SBATCH --hint=nomultithread
#SBATCH --account=krk@v100


MAPS_DIR=models/MAPS_$1
echo $MAPS_DIR

source config.sh
echo $ADNI_CAPS_DIR
echo $HYPO_CAPS_DIR
echo $PARTICIPANTS_TSV_FOLDER

# Predict on test AD
GROUPE=test_AD
PARTICIPANT_TSV=${PARTICIPANTS_TSV_FOLDER}/test/AD_baseline.tsv

echo clinicadl predict $MAPS_DIR $GROUPE --caps_directory $ADNI_CAPS_DIR --participants_tsv $PARTICIPANT_TSV --diagnoses AD --split 0 --selection_metrics loss --save_latent_tensor  --save_tensor  --overwrite
clinicadl   predict $MAPS_DIR $GROUPE \
            --caps_directory $ADNI_CAPS_DIR \
            --participants_tsv $PARTICIPANT_TSV \
            --diagnoses AD \
            --split 0 \
            --selection_metrics loss \
            --save_latent_tensor  \
            --save_tensor \
            --overwrite

# Predict on test CN
GROUPE=test_CN
PARTICIPANT_TSV=${PARTICIPANTS_TSV_FOLDER}/test/CN_baseline.tsv

echo clinicadl predict $GROUPE
clinicadl   predict $MAPS_DIR $GROUPE \
            --caps_directory $ADNI_CAPS_DIR \
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
    CAPS_DIR=${HYPO_CAPS_DIR}/caps_${PATHOLOGY}_30

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
    CAPS_DIR=${HYPO_CAPS_DIR}//caps_ad_${PERCENTAGE}

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
    python python_scripts/healthiness_measurment.py $MAPS_DIR ad $PERCENTAGE $ADNI_CAPS_DIR
done