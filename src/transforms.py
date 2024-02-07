from torchvision import transforms as v2transforms


def crop_nonzero(image):
    bbox = image.getbbox()
    cropped_image = image.crop(bbox)
    return cropped_image

class CropNonzero(object):
    def __call__(self, image):
        return crop_nonzero(image)

# Define transformations
transforms = v2transforms.Compose([
    CropNonzero(),
    v2transforms.Resize(size=(128,128)),
    v2transforms.ToTensor(),
    v2transforms.Normalize(mean=[0.0177, 0.0195, 0.0210], std=[0.2271,0.2271, 0.2271]),
])