from PIL import Image

BINARY_THRESHOLD = 1


def binarise_image(img: Image.Image):
    # Binarise image.
    img = img.convert("L")
    img = img.point(lambda x: 0 if x < BINARY_THRESHOLD else 255, "1")
    return img


def pad_image(img: Image.Image) -> Image.Image:
    desired_size = 128
    old_size = img.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = img.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(
        img, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
    )
    return new_im


def crop_image(img: Image.Image) -> Image.Image:
    width, height = img.size
    new_width = 2048
    new_height = 2048
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    return img
