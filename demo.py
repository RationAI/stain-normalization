import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from stain_normalization.stain_normalization_model import StainNormalizationModel


class StainNormalizerDemo:
    CHECKPOINT_PATH = "./demo_data/checkpoint.ckpt"
    MEAN = (0.780361961, 0.614529804, 0.725567843)
    STD = (0.144428627, 0.183275686, 0.140768627)

    NORMALIZE_TRANSFORM = A.Normalize(mean=MEAN, std=STD, max_pixel_value=1)
    TO_TENSOR = ToTensorV2()

    def __init__(self, use_cpu: bool = True) -> None:
        self.device = torch.device(
            "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")

        self.model = StainNormalizationModel()
        checkpoint = torch.load(self.CHECKPOINT_PATH, map_location=self.device)
        if "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        normalized = self.NORMALIZE_TRANSFORM(image=img_np)["image"]
        tensor = self.TO_TENSOR(image=normalized)["image"].to(self.device).unsqueeze(0)
        return tensor

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.MEAN).view(3, 1, 1).to(tensor.device)
        std = torch.tensor(self.STD).view(3, 1, 1).to(tensor.device)
        return tensor * std + mean

    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        tensor = self.denormalize(tensor)
        tensor = tensor.clamp(0, 1)
        tensor = (tensor * 255).byte()
        return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy())

    def save_image(self, tensor: torch.Tensor, path: Path) -> None:
        img = self.tensor_to_image(tensor.squeeze(0))
        img.save(path)

    def predict_image(self, input_path: Path, output_path: Path) -> None:
        with torch.no_grad():
            input_tensor = self.load_image(input_path)
            output = self.model(input_tensor)
            self.save_image(output, output_path)
        print(f"Saved normalized image to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stain Normalization Demo")
    parser.add_argument(
        "--input",
        type=str,
        default="./demo_data/modified",
        help="Input image or folder path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./demo_data",
        help="Output folder path",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Force CPU even if GPU available",
    )
    args = parser.parse_args()

    normalizer = StainNormalizerDemo(use_cpu=args.use_cpu)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    if input_path.is_file():
        out_filename = input_path.stem + "_normalized" + input_path.suffix
        out_path = output_path / out_filename
        normalizer.predict_image(input_path, out_path)

    elif input_path.is_dir():
        norm_folder = output_path / "normalized"
        norm_folder.mkdir(exist_ok=True)

        input_files = sorted(
            f
            for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not input_files:
            print(f"No image files found in {input_path}.")
            return
        for in_path in input_files:
            out_filename = in_path.stem + "_normalized" + in_path.suffix
            out_path = norm_folder / out_filename
            normalizer.predict_image(in_path, out_path)

    else:
        raise ValueError(f"Input path {input_path} does not exist.")


if __name__ == "__main__":
    main()
