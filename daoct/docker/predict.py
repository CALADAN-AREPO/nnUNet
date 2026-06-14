import os
import torch
import torch.nn.functional as F
import numpy as np
from glob import glob


def normalize_per_image(x):
    return (x - x.mean()) / (x.std() + 1e-8)


@torch.no_grad()
def ensemble_predict(image, models, weights):
    softmax_outputs = []
    for model, w in zip(models, weights):
        logits = model(image)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        softmax_outputs.append(w * F.softmax(logits, dim=1))
    avg = sum(softmax_outputs) / sum(weights)
    return avg.argmax(dim=1).cpu().numpy()


def main():
    models = []
    weights = [1.0, 1.0, 1.0, 1.0]
    trainer_names = [
        "nnUNetTrainerDAOCT_Baseline",
        "nnUNetTrainerDAOCT_DINOv3",
        "nnUNetTrainerDAOCT_RETFound",
        "nnUNetTrainerDAOCT_MedSAM3",
    ]
    for trainer_name in trainer_names:
        from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=False, allow_tqdm=False,
        )
        models.append(predictor)

    input_folder = os.environ.get("INPUT_FOLDER", "/input")
    output_folder = os.environ.get("OUTPUT_FOLDER", "/output")
    os.makedirs(output_folder, exist_ok=True)

    image_files = sorted(glob(os.path.join(input_folder, "*.npy")) + glob(os.path.join(input_folder, "*.nii.gz")))
    for img_path in image_files:
        if img_path.endswith(".npy"):
            image = np.load(img_path)
        else:
            import nibabel as nib
            image = nib.load(img_path).get_fdata()
        image = normalize_per_image(image)
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        seg = ensemble_predict(image_tensor, models, weights)
        basename = os.path.basename(img_path)
        np.save(os.path.join(output_folder, basename.replace(".npy", "_seg.npy")), seg)


if __name__ == "__main__":
    main()
