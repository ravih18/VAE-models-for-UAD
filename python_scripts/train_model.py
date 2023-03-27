import sys
import os
from os import path

from clinicadl import MapsManager
import clinicadl.utils.network as network_package

from pythae.pipelines import TrainingPipeline

from train_utils import get_model_parameters, get_parameters_dict, return_datasets


# Define paths
CAPS_DIR = sys.argv[2]
TSV_PATH = path.join(sys.argv[3], "train")

config_folder = "./config/"
config_path = {
    'VAE': path.join(config_folder, 'vae_config.toml'),
    'Beta_VAE': path.join(config_folder, 'beta_vae_config.toml'),
    'VAE_LinNF': path.join(config_folder, 'vae_linnf_config.toml'),
    'VAE_IAF': path.join(config_folder, 'vae_iaf_config.toml'),
    'Disentangled_Beta_VAE': path.join(config_folder, 'disentangled_beta_vae_config.toml'),
    'Factor_VAE': path.join(config_folder, 'factor_vae_config.toml'),
    'Beta_TC_VAE': path.join(config_folder, 'beta_tc_vae_config.toml'),
    'MS_SSIM_VAE': path.join(config_folder, 'ms_ssim_vae_config.toml'),
    'INFO_VAE_MMD': path.join(config_folder, 'info_vae_mmd_config.toml'),
    'S_VAE': path.join(config_folder, 's_vae_config.toml'),
    'Poincare_VAE': path.join(config_folder, 'poincare_vae_config.toml'),
    'Adversarial_AE': path.join(config_folder, 'adversarial_ae_config.toml'),
    'VAEGAN': path.join(config_folder, 'vaegan_config.toml'),
    'VQ_VAE': path.join(config_folder, 'vq_vae_config.toml'),
    'H_VAE': path.join(config_folder, 'h_vae_config.toml'),
    'RH_VAE': path.join(config_folder, 'rh_vae_config.toml'),
    'IWAE': path.join(config_folder, 'iwae_config.toml'),
    'MIWAE': path.join(config_folder, 'miwae_config.toml'),
    'PIWAE': path.join(config_folder, 'piwae_config.toml'),
    'CIWAE': path.join(config_folder, 'ciwae_config.toml'),
    'VAMP': path.join(config_folder, 'vamp_config.toml'),
    'WAE': path.join(config_folder, 'wasserstein_ae_config.toml'),
    'AE': path.join(config_folder, 'ae_config.toml'),
    'RAE_L2': path.join(config_folder, 'rae_l2_config.toml'),
    'RAE_GP': path.join(config_folder, 'rae_gp_config.toml'),
}

def train(model_name):

    # Read parameters
    config_toml = config_path[model_name]

    # Get parameters from config file
    parameters = get_parameters_dict(config_toml, CAPS_DIR, TSV_PATH)
    parameters["latent_space_size"] = 256

    # Get train et validation datasets
    train_dataset, eval_dataset = return_datasets(parameters)

    # Inialise MAPS
    maps_dir = path.join("./models", f"MAPS_{model_name}")
    maps_manager = MapsManager(maps_dir, parameters, verbose="info")
    # Create model output dir
    model_dir = path.join(maps_dir, "split-0", "best-loss")
    if not path.exists(model_dir):
        os.makedirs(model_dir)

    # Import the model
    model_class = getattr(network_package, parameters["architecture"])
    kwargs = get_model_parameters(model_class, parameters)
    clinicadl_model = model_class(**kwargs)

    model = clinicadl_model.model
    config = clinicadl_model.get_trainer_config(
        output_dir=model_dir,
        num_epochs=parameters["epochs"],
        learning_rate=parameters["learning_rate"],
        batch_size=parameters["batch_size"],
    )

    # Create Pythae Training Pipeline
    pipeline = TrainingPipeline(
        training_config=config,
        model=model
    )

    # Launch training
    pipeline(
	train_data=train_dataset, # must be torch.Tensor or np.array
        eval_data=eval_dataset # must be torch.Tensor or np.array
    )

    # Move saved model to the correct path in the MAPS
    src = path.join(model_dir, "*_training_*/final_model/model.pt")
    os.system(f"mv {src} {model_dir}")

if __name__=="__main__":
    model_name = sys.argv[1]
    print("="*10, model_name, "="*10)
    train(model_name)
