import sys
import os
from os import path
import pandas as pd
import torch
import nibabel as nib
from nilearn.image import resample_to_img

from python_scripts.metrics import ssim_map_3D, mse_in_mask, ssim_in_mask, minkowski_distance

from clinica.utils.inputs import RemoteFileStructure, fetch_file

maps_directory = sys.argv[1]
pathology = sys.argv[2]
percentage = sys.argv[3]

group = f"test_hypo_{pathology}_{percentage}"

# Load all sub/session from tsv
tsv_file = path.join(maps_directory, "groups", group, "data.tsv")
df = pd.read_csv(tsv_file, sep="\t", usecols=["participant_id", "session_id"])
sessions_list = list(df.to_records(index=False))

# Load region masks
caps_directory = sys.argv[4]
img_path = path.join(caps_directory,
                     "subjects",
                     "sub-ADNI002S0685",
                     "ses-M48",
                     "t1_linear",
                     "sub-ADNI002S0685_ses-M48_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz")

mask_directory = "masks"
url_aramis = "https://aramislab.paris.inria.fr/files/data/masks/hypo/"
FILE1 = RemoteFileStructure(
    filename=f"mask_hypo_{pathology}.nii",
    url=url_aramis,
    checksum="drgjeryt",
)
if not path.exists(mask_directory):
    os.makedirs(mask_directory)

if not path.isfile(path.join(mask_directory, f"mask_hypo_{pathology}.nii")):
    try:
        mask_path = fetch_file(FILE1, mask_directory)
    except:
        raise(
            """Unable to download masks, please download them
            manually at https://aramislab.paris.inria.fr/files/data/masks/
            and provide a valid path."""
        )
else:
    mask_path = path.join(mask_directory, f"mask_hypo_{pathology}.nii")

mask_nii = nib.load(mask_path)
mask_nii = resample_to_img(mask_nii, nib.load(img_path), interpolation='nearest')
pathology_mask = mask_nii.get_fdata()[4:164:2, 8:200:2, 8:168:2]

# Load MNI template mask
mni_path = "/gpfswork/rech/krk/usy14zi/vae_benchmark/data/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii"
mni_nii = nib.load(mni_path)
mni_nii = resample_to_img(mni_nii, nib.load(img_path), interpolation='nearest')
brain_mask = mni_nii.get_fdata()[4:164:2, 8:200:2, 8:168:2]

out_mask = brain_mask - pathology_mask

# Initialize output dataframe
columns = {
    "participant_id": pd.Series(dtype='str'),
    "session_id": pd.Series(dtype='str'),
    "MSE_recon": pd.Series(dtype='float'),
    "SSIM_recon": pd.Series(dtype='float'),
    "MSE_ground_truth": pd.Series(dtype='float'),
    "MSE_input": pd.Series(dtype='float'),
    "MSE_mask_ground_truth": pd.Series(dtype='float'),
    "MSE_mask_input": pd.Series(dtype='float'),
    "MSE_out_ground_truth": pd.Series(dtype='float'),
    "MSE_out_input": pd.Series(dtype='float'),
    "SSIM_ground_truth": pd.Series(dtype='float'),
    "SSIM_input": pd.Series(dtype='float'),
    "SSIM_mask_ground_truth": pd.Series(dtype='float'),
    "SSIM_mask_input": pd.Series(dtype='float'),
    "SSIM_out_ground_truth": pd.Series(dtype='float'),
    "SSIM_out_input": pd.Series(dtype='float'),
}
results_df = pd.DataFrame(columns)
results_df_path = path.join(maps_directory, "split-0", "best-loss", group, "heathiness_measures.tsv")

# Datadrame to assess variational inference
columns_vi = {
    "participant_id": pd.Series(dtype='str'),
    "session_id": pd.Series(dtype='str'),
    "MSE_recon": pd.Series(dtype='float'),
}
df_vi = pd.DataFrame(columns_vi)
df_vi_path = path.join(maps_directory, "split-0", "best-loss", group, "latent_distance_heathyvssim.tsv")

for session in sessions_list:
    sub, ses = session[0], session[1]

    # Load all IO image
    gt_file = sub + "_" + ses + "_image-0_input.pt"
    gt_path = path.join(maps_directory, "split-0", "best-loss", "test_CN", "tensors", gt_file)
    gt_array = torch.load(gt_path).numpy()

    gt_recon_file = sub + "_" + ses + "_image-0_output.pt"
    gt_recon_path = path.join(maps_directory, "split-0", "best-loss", "test_CN", "tensors", gt_recon_file)
    gt_recon_array = torch.load(gt_recon_path).detach().numpy()

    input_file = sub + "_" + ses + "_image-0_input.pt"
    input_path = path.join(maps_directory, "split-0", "best-loss", group, "tensors", input_file)
    input_array = torch.load(input_path).numpy()

    recon_file = sub + "_" + ses + "_image-0_output.pt"
    recon_path = path.join(maps_directory, "split-0", "best-loss", group, "tensors", recon_file)
    recon_array = torch.load(recon_path).detach().numpy()

    # Compute metrics between both reconstructions
    mse_recon = ((gt_recon_array[0] - recon_array[0])**2).mean()
    ssim_recon = ssim_map_3D(gt_recon_array, recon_array).numpy().mean()

    # Compute MSE on whole image
    mse_gt = ((gt_array[0] - recon_array[0])**2).mean()
    mse_input = ((input_array[0] - recon_array[0])**2).mean()

    # Compute MSE in pathology mask
    mse_pathology_mask_gt = mse_in_mask(gt_array[0], recon_array[0], pathology_mask)
    mse_pathology_mask_recon = mse_in_mask(input_array[0], recon_array[0], pathology_mask)

    # Compute MSE outside pathology mask
    mse_out_mask_gt = mse_in_mask(gt_array[0], recon_array[0], out_mask)
    mse_out_mask_recon = mse_in_mask(input_array[0], recon_array[0], out_mask)

    # Compute SSIM on whole image
    ssim_gt = ssim_map_3D(gt_array, recon_array).numpy().mean()
    ssim_input = ssim_map_3D(input_array, recon_array).numpy().mean()

    # Compute SSIM in pathology mask
    ssim_pathology_mask_gt = ssim_in_mask(gt_array, recon_array, pathology_mask)
    ssim_pathology_mask_recon = ssim_in_mask(input_array, recon_array, pathology_mask)

    # Compute SSIM outside pathology mask
    ssim_out_mask_gt = ssim_in_mask(gt_array, recon_array, out_mask)
    ssim_out_mask_recon = ssim_in_mask(input_array, recon_array, out_mask)

    # Add results in pandas dataframe
    row = [
        sub,
        ses,
        mse_recon,
        ssim_recon,
        mse_gt,
        mse_input,
        mse_pathology_mask_gt,
        mse_pathology_mask_recon,
        mse_out_mask_gt,
        mse_out_mask_recon,
        ssim_gt,
        ssim_input,
        ssim_pathology_mask_gt,
        ssim_pathology_mask_recon,
        ssim_out_mask_gt,
        ssim_out_mask_recon,
    ]
    row_df = pd.DataFrame([row], columns=columns.keys())
    results_df = pd.concat([results_df, row_df])

    # Load latent_tensor
    gt_latent_file = sub + "_" + ses + "_image-0_latent.pt"
    gt_latent_path = path.join(maps_directory, "split-0", "best-loss", "test_CN", "latent_tensors", gt_latent_file)
    gt_latent_array = torch.load(gt_latent_path).detach().numpy()

    sim_latent_file = sub + "_" + ses + "_image-0_latent.pt"
    sim_latent_path = path.join(maps_directory, "split-0", "best-loss", group, "latent_tensors", sim_latent_file)
    sim_latent_array = torch.load(sim_latent_path).detach().numpy()

    dist = minkowski_distance(gt_latent_array, sim_latent_array)

    row_vi = [
        sub,
        ses,
        dist
    ]
    row_vi_df = pd.DataFrame([row_vi], columns=columns_vi.keys())
    df_vi = pd.concat([df_vi, row_vi_df])

results_df.to_csv(results_df_path, index=False, sep="\t")
df_vi.to_csv(df_vi_path, index=False, sep="\t")
