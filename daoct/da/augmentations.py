from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform


class MultiplicativeNoiseTransform:
    def __init__(self, multiplier=(0.9, 1.1), per_channel=False, p=1.0):
        self.multiplier = multiplier
        self.per_channel = per_channel
        self.p = p

    def __call__(self, image, **kwargs):
        import numpy as np
        if np.random.random() > self.p:
            return image
        if self.per_channel:
            for c in range(image.shape[0]):
                m = np.random.uniform(*self.multiplier)
                image[c] = image[c] * m
        else:
            m = np.random.uniform(*self.multiplier)
            image = image * m
        return image


class RandomCLAHE:
    def __init__(self, clip_limit=(1.0, 3.0), p=0.2):
        self.clip_limit = clip_limit
        self.p = p

    def __call__(self, image, **kwargs):
        import numpy as np
        from skimage.exposure import equalize_adapthist
        if np.random.random() > self.p:
            return image
        clip = np.random.uniform(*self.clip_limit)
        for c in range(image.shape[0]):
            img_slice = image[c].astype(np.float64)
            v_min, v_max = img_slice.min(), img_slice.max()
            if v_max - v_min < 1e-8:
                continue
            img_norm = (img_slice - v_min) / (v_max - v_min)
            enhanced = equalize_adapthist(img_norm, clip_limit=clip)
            image[c] = enhanced * (v_max - v_min) + v_min
        return image


CROSS_VENDOR_TRANSFORMS = [
    RandomTransform(
        GammaTransform(gamma=BGContrast((0.6, 1.6)), p_invert_image=1, synchronize_channels=False, p_per_channel=1, p_retain_stats=1),
        apply_probability=0.5,
    ),
    RandomTransform(
        MultiplicativeBrightnessTransform(multiplier_range=BGContrast((-0.2, 0.2)), synchronize_channels=False, p_per_channel=1),
        apply_probability=0.3,
    ),
    RandomTransform(
        ContrastTransform(contrast_range=BGContrast((0.75, 1.25)), preserve_range=True, synchronize_channels=False, p_per_channel=1),
        apply_probability=0.3,
    ),
    RandomTransform(
        GaussianNoiseTransform(noise_variance=(0.0, 0.05), p_per_channel=1, synchronize_channels=True),
        apply_probability=0.4,
    ),
    RandomTransform(
        SimulateLowResolutionTransform(scale=(0.5, 1.0), synchronize_channels=False, synchronize_axes=True, ignore_axes=None, allowed_channels=None, p_per_channel=0.5),
        apply_probability=0.25,
    ),
    RandomTransform(
        GammaTransform(gamma=BGContrast((0.7, 1.5)), p_invert_image=0, synchronize_channels=False, p_per_channel=1, p_retain_stats=1),
        apply_probability=0.3,
    ),
    MirrorTransform(allowed_axes=(0,)),
]
