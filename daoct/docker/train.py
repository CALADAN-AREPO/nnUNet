import os
import subprocess
import sys


def run_nnunet_plan_and_preprocess():
    subprocess.check_call([
        sys.executable, "-m", "nnunetv2.experiment_planning.DatasetAnalyzer",
        "--dataset_name_or_id", os.environ.get("nnUNet_dataset", "Dataset001_DAOCT"),
    ])


def train_model(trainer_name, fold=0):
    subprocess.check_call([
        sys.executable, "-m", "nnunetv2.run.run_training",
        os.environ.get("nnUNet_dataset", "Dataset001_DAOCT"),
        "2d",
        fold,
        "--trainer", trainer_name,
    ])


def main():
    run_nnunet_plan_and_preprocess()

    train_model("nnUNetTrainerDAOCT_Baseline", fold=0)

    da_strategy = os.environ.get("DA_STRATEGY", "augmentation_only")
    if da_strategy == "pseudo_label":
        from daoct.da.mean_teacher import generate_pseudo_labels
        generate_pseudo_labels(model_name="Baseline", confidence_threshold=0.9)

    for trainer in ["nnUNetTrainerDAOCT_DINOv3", "nnUNetTrainerDAOCT_RETFound", "nnUNetTrainerDAOCT_MedSAM3"]:
        train_model(trainer, fold=0)


if __name__ == "__main__":
    main()
