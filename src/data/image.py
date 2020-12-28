from PIL import Image

# Threshold used in mask binarisation. All values smaller than the threshold will be
# set to 0 (black), all other values to 255 (white).
BINARY_THRESHOLD = 1


def binarise(img: Image.Image) -> Image.Image:
    # Binarise image.
    img = img.convert("L")
    img = img.point(lambda x: 0 if x < BINARY_THRESHOLD else 255, "1")
    return img


def pad(img: Image.Image) -> Image.Image:
    """Pad to square with original image in center."""
    width, height = img.size

    new_size = max(width, height)

    new_im = Image.new("RGB", (new_size, new_size))
    region = ((new_size - width) // 2, (new_size - height) // 2)
    new_im.paste(img, region)
    img.close()
    return new_im


def center_crop(img: Image.Image, size: int) -> Image.Image:
    width, height = img.size
    left = (width - size) // 2
    top = (height - size) // 2
    right = (width + size) // 2
    bottom = (height + size) // 2

    # Crop the center of the image
    new_img = img.crop((left, top, right, bottom))
    img.close()
    return new_img


def resize(img: Image.Image, size: int) -> Image.Image:
    new_img = img.resize((size, size), resample=Image.ANTIALIAS)
    img.close()
    return new_img
