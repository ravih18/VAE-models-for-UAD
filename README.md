# VAE-models-for-UAD

Benchmark of VAE models for Unsupervised Anomaly Detection

This repository aims to give all the steps and scripts to reproduce experiments in *link to article*.
## Get the data

### Download FDG PET from ADNI
For this work, we used the FDG PET available in the [ADNI dataset](https://adni.loni.usc.edu/). Data can be downloaded from the [IDA plateform](https://ida.loni.usc.edu/login.jsp?project=ADNI). We select the FDG PET that are [co-registered, averaged, with standardized image and voxel size and with uniform resolution](https://adni.loni.usc.edu/methods/pet-analysis-method/pet-analysis/). Data are stored in the `DATASET_DIRECTORY` and the CSV files with the clinical data are stored in `CLINICAL_DATA_DIRECTORY`.

### Convert dataset to BIDS
The data can then be converted to BIDS using Clinica software using the [adni-to-bids](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Converters/ADNI2BIDS/) pipeline. The `BIDS_DIRECTORY` is the output directory of the pipeline:
```
clinica convert adni-to-bids DATASET_DIRECTORY CLINICAL_DATA_DIRECTORY BIDS_DIRECTORY
```

### Apply linear registration and normalization
PET images are then processed using the Clinica [PET Linear](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/PET_Linear/) pipeline. We use the cerebellum and pons as reference regions for the SUVR:
```
clinica run pet-linear BIDS_DIRECTORY CAPS_DIRECTORY 18FFDG cerebellumPons2
```
with `CAPS_DIRECTORY` being the output [CAPS directory](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/) of the pipeline.

## Prepare the data

Once the data are preprocessed, we used [ClinicaDL](https://clinicadl.readthedocs.io/en/latest/) to extract Pytorch 3D tensors from Nifti files:
```
clinicadl extract CAPS_DIRECTORY image
```

As the registration algorithm is quite simple, we use the QC method in ClinicaDL to filter the images that are not well registered.
QC T1
QC PET

We then use the `get-labels` command to select the cognitively normal subject with stable diagnosis and `split` function to generate our train, test and validation sets.

## Train the models
Install ClinicaDL with the correct branch.
Install Pythae.

Launch the script.

## Make evaluation test sets
Simulate hypometabolism using clinicadl

launch evaluation script