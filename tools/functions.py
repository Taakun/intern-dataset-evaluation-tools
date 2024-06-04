
import albumentations as albu


class Functions:
    def get_augmentation(self, phase):
        if phase == "train":
            train_transform = [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
                albu.RandomBrightnessContrast()
            ]
            return albu.Compose(train_transform)

        if phase=="valid":
            return None

    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def get_preprocessing(self, preprocessing_fn):
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=self.to_tensor, mask=self.to_tensor),
        ]
        return albu.Compose(_transform)
    