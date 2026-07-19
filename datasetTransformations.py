import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

IMAGE_EXTENSIONS= {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}

def loadImages(dir: Path) ->list:
    result = []

    for path in sorted(dir.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            img = Image.open(path).convert("RGB")
            result.append((img, path.name))
    
    if not result:
        raise FileNotFoundError(f"No images found in: {dir}")
    
    print(f"Loaded {len(result)} images from '{dir}'")

    return result

def saveImage(img: Image.Image, output: Path, filename: str, suffix: str = "") -> None:
    output.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem
    ext = Path(filename).suffix.lower()
    dest = output / f"{stem}{suffix}{ext}"

    img.save(dest)

def flip(img: Image.Image, direction: str = "random") -> Image.Image:
    if direction == "random":
        direction = random.choice(["horizontal", "vertical"])
    if direction == "horizontal":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if direction == "vertical":
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    
    raise ValueError("Direction must be 'horizonta', 'vertical' or 'random'")
    
def rotate(img: Image.Image, angle: float = None, max_angle: float = 20.0) -> Image.Image:
    if angle is None:
        angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, expand=False, fillcolor=(0, 0, 0))

def brightness(img: Image.Image, factor: float = None, low: float = 0.6, high: float = 1.4) -> Image.Image:
    if factor is None:
        factor = random.uniform(low, high)
    return ImageEnhance.Brightness(img).enhance(factor)

def contrast(img: Image.Image, factor: float = None, low: float = 0.6, high: float = 1.4) -> Image.Image:
    if factor is None:
        factor = random.uniform(low, high)
    return ImageEnhance.Contrast(img).enhance(factor)

def saturation(img: Image.Image, factor: float = None, low: float = 0.5, high: float = 1.5) -> Image.Image:
    if img.mode != "RGB":
        return img
    if factor is None:
        factor = random.uniform(low, high)
    return ImageEnhance.Color(img).enhance(factor)

def gaussianBlur(img: Image.Image, radius: float = None, max_radius: float = 1.5) -> Image.Image:
    if radius is None:
        radius = random.uniform(0.3, max_radius)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def noise(img: Image.Image, amount: float = 0.02) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    noise = arr + np.random.normal(0, amount * 255, arr.shape)
    return Image.fromarray(np.clip(noise, 0, 255).astype(np.uint8))

if __name__ == "__main__":
    imagesRails = loadImages(Path("Dataset") / "train" / "Rails")
    imagesDrone = loadImages(Path("Dataset") / "train" / "Drone")

    for img, name in imagesRails:
        saveImage(flip(img), Path("Dataset") / "train" / "Rails", name, suffix="_flip")
        saveImage(rotate(img), Path("Dataset") / "train" / "Rails", name, suffix="_rotate")
        saveImage(brightness(img), Path("Dataset") / "train" / "Rails", name, suffix="_brightness")
        saveImage(saturation(img), Path("Dataset") / "train" / "Rails", name, suffix="_saturation")
        saveImage(contrast(img), Path("Dataset") / "train" / "Rails", name, suffix="_contrast")
        saveImage(noise(img), Path("Dataset") / "validation" / "Rails", name, suffix="_noise")
        saveImage(gaussianBlur(img), Path("Dataset") / "validation" / "Rails", name, suffix="_blur")

    for img, name in imagesDrone:
        saveImage(flip(img), Path("Dataset") / "train" / "Drone", name, suffix="_flip")
        saveImage(rotate(img), Path("Dataset") / "train" / "Drone", name, suffix="_rotate")
        saveImage(brightness(img), Path("Dataset") / "train" / "Drone", name, suffix="_brightness")
        saveImage(saturation(img), Path("Dataset") / "train" / "Drone", name, suffix="_saturation")
        saveImage(contrast(img), Path("Dataset") / "train" / "Drone", name, suffix="_contrast")
        saveImage(noise(img), Path("Dataset") / "validation" / "Drone", name, suffix="_noise")
        saveImage(gaussianBlur(img), Path("Dataset") / "validation" / "Drone", name, suffix="_blur")