
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):

    # データセットのクラス名
    CLASSES = ['backgrounds','aeroplane','bicycle','bird','boat','bottle',
                           'bus','car','cat','chair','cow', 
                           'diningtable','dog','horse','motorbike','person', 
                           'potted plant','sheep','sofa','train','monitor','unlabeled'
                           ]

    def __init__(self, images_path, masks_path, segment_class, 
                 augmentation=None, preprocessing=None):

        self.images_path = images_path
        self.masks_path = masks_path
        self.segment_class = segment_class
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, i):
        # 元画像の読み込み、整形
        image = Image.open(self.images_path[i])
        image_resize = image.resize((128,128), Image.ANTIALIAS)
        image = np.asarray(image_resize)
        if len(image.shape)==2:
            image = Image.open(self.images_path[i]).convert("L")
            image_resize = image.resize((128,128), Image.ANTIALIAS)
            image_color = ImageOps.colorize(image_resize, black=(0, 0, 0), white=(255, 255, 0))
            image = np.asarray(image_color)
        elif image.shape[2] == 4:
            image = np.delete(image, 3, axis=2)

        # maskの読み込み、整形
        masks = Image.open(self.masks_path[i])
        masks = masks.resize((128,128), Image.ANTIALIAS)
        masks = np.asarray(masks)

        # maskデータの境界線を表す255は扱いにくいので21に変換
        masks = np.where(masks == 255, 21, masks)

        # maskデータを正解ラベル毎の1hotに変換
        cls_idx = [self.CLASSES.index(cls) for cls in self.segment_class]
        masks = [(masks == idx) for idx in cls_idx]
        mask = np.stack(masks, axis=-1).astype("float")

        # augmentationの実行
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # 前処理の実行
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask
    