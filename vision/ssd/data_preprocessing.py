import logging

from ..transforms.transforms import (Compose, ConvertFromInts, Expand,
                                     PhotometricDistort, RandomMirror,
                                     RandomSampleCrop, Resize, SubtractMeans,
                                     ToPercentCoords, ToTensor)


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0, mode=None):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size

        if mode is None:
            logging.info("Using mode with hard augs")
            self.augment = Compose([
                ConvertFromInts(),
                PhotometricDistort(),
                Expand(self.mean),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(self.size),
                SubtractMeans(self.mean),
                lambda img, boxes=None, labels=None: (img / std, boxes, labels),
                ToTensor(),
            ])

        elif mode == 'light':
            logging.info("Using mode with light augs")
            self.augment = Compose([
                ConvertFromInts(),
                Expand(self.mean),
                RandomMirror(),
                ToPercentCoords(),
                Resize(self.size),
                SubtractMeans(self.mean),
                lambda img, boxes=None, labels=None: (img / std, boxes, labels),
                ToTensor(),
            ])

        else:
            raise ValueError(mode)

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image
