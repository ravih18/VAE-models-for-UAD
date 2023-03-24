def get_parameters_dict(config_toml, caps_directory, tsv_path):

    from clinicadl.train.train_utils import build_train_dict
    from clinicadl.utils.preprocessing import read_preprocessing

    parameters = build_train_dict(config_toml, "reconstruction")

    parameters["caps_directory"] = caps_directory
    parameters["tsv_path"] = tsv_path
    parameters["mode"] = "image"
    parameters["network_task"] = "reconstruction"
    parameters["input_size"] = (1, 80, 96, 80)
    parameters["preprocessing_dict"] = read_preprocessing(
            "/gpfswork/rech/krk/commun/datasets/adni/"\
            "caps/caps_pet_uniform/tensor_extraction/"\
            "extract_pet_uniform_image.json")

    return parameters


def return_datasets(parameters):
    from clinicadl.utils.caps_dataset.data import PythaeCAPS, get_transforms
    from clinicadl.utils.split_manager import KFoldSplit, SingleSplit
    # split_manager = KFoldSplit(
    #    caps_directory = parameters["caps_directory"],
    #    tsv_path = parameters["tsv_path"],
    #    diagnoses = parameters["diagnoses"],
    #    n_splits = parameters["n_splits"],
    #    baseline = parameters["baseline"],
    #    multi_cohort = parameters["multi_cohort"],
    #    split_list = parameters["split"],
    #)
    split_manager = SingleSplit(
        caps_directory = parameters["caps_directory"],
        tsv_path = parameters["tsv_path"],
        diagnoses = parameters["diagnoses"],
        baseline = parameters["baseline"],
        multi_cohort = parameters["multi_cohort"],
        split_list = parameters["split"],
    )
    split_df_dict = split_manager[0]
    train_transforms, all_transforms = get_transforms(
        normalize=parameters["normalize"],
        data_augmentation=parameters["data_augmentation"],
        size_reduction=parameters["size_reduction"],
        size_reduction_factor=parameters["size_reduction_factor"],
    )

    train_dataset = PythaeCAPS(
        parameters["caps_directory"],
        split_df_dict["train"],
        parameters["preprocessing_dict"],
        train_transformations=train_transforms,
        all_transformations=all_transforms,
    )

    eval_dataset = PythaeCAPS(
        parameters["caps_directory"],
        split_df_dict["validation"],
        parameters["preprocessing_dict"],
        train_transformations=train_transforms,
        all_transformations=all_transforms,
    )
    return train_dataset, eval_dataset


def get_model_parameters(model_class, parameters):
    args = list(
        model_class.__init__.__code__.co_varnames[
            : model_class.__init__.__code__.co_argcount
        ]
    )
    args.remove("self")
    model_parameters = dict()
    for arg in args:
        model_parameters[arg] = parameters[arg]
    return model_parameters