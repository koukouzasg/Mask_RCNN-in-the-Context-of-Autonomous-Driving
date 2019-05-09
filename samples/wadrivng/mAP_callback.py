# This current implementation calculates the average precision for every image and then takes the mean of the values.
# In papers they calculate the average precision per class and then calculates the mean of the class-wise values.

from keras.callbacks import Callback
import numpy as np
import sys
import os

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/koukouzas/CODE/Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import model as modellib


class MeanAveragePrecisionCallback(Callback):
    def __init__(self, train_model: modellib.MaskRCNN, inference_model: modellib.MaskRCNN, dataset: utils.Dataset,
                 calculate_at_every_X_epoch: int = 3, dataset_limit: int = None,
                 verbose: int = 1):
        """
        Callback which calculates the mAP on the defined test/validation dataset
        :param train_model: Mask RCNN model in training mode
        :param inference_model: Mask RCNN model in inference mode
        :param dataset: test/validation dataset, it will calculate the mAP on this set
        :param calculate_at_every_X_epoch: With this parameter we can define if we want to do the calculation at
        every epoch or every second, etc...
        :param dataset_limit: When we have a huge dataset calculation can take a lot of time, with this we can set a
        limit to the number of data points used
        :param verbose: set verbosity (1 = verbose, 0 = quiet)
        """

        super().__init__()

        if train_model.mode != "training":
            raise ValueError("Train model should be in training mode, instead it is in: {0}".format(train_model.mode))

        if inference_model.mode != "inference":
            raise ValueError(
                "Inference model should be in inference mode, instead it is in: {0}".format(train_model.mode))

        if inference_model.config.BATCH_SIZE != 1:
            raise ValueError("This callback only works with the bacth size of 1, instead: {0} was defined".format(
                inference_model.config.BATCH_SIZE))

        self.train_model = train_model
        self.inference_model = inference_model
        self.dataset = dataset
        self.calculate_at_every_X_epoch = calculate_at_every_X_epoch
        self.dataset_limit = len(self.dataset.image_ids)
        if dataset_limit is not None:
            self.dataset_limit = dataset_limit
        self.dataset_image_ids = self.dataset.image_ids.copy()

        self._verbose_print = print if verbose > 0 else lambda *a, **k: None

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0 and epoch % self.calculate_at_every_X_epoch == 0:
            self._verbose_print("Calculating mAP...")
            self._load_weights_for_model()

            mAPs = self._calculate_mean_average_precision()
            mAP = np.mean(mAPs)

            if logs is not None:
                logs["val_mean_average_precision"] = mAP

            self._verbose_print("mAP at epoch {0} is: {1}".format(epoch, mAP))

        super().on_epoch_end(epoch, logs)


    def _load_weights_for_model(self):
        last_weights_path = self.train_model.find_last()
        self._verbose_print("Loaded weights for the inference model (last checkpoint of the train model): {0}".format(
            last_weights_path))
        self.inference_model.load_weights(last_weights_path,
                                          by_name=True)

    def _calculate_mean_average_precision(self):
        mAPs = []

        np.random.shuffle(self.dataset_image_ids)

        for image_id in self.dataset_image_ids[:self.dataset_limit]:
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(self.dataset, self.inference_model.config,
                                                                             image_id, use_mini_mask=False)
            results = self.inference_model.detect([image], verbose=0)
            r = results[0]
            # Compute mAP - VOC uses IoU 0.5
            AP, _, _, _ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                           r["class_ids"], r["scores"], r['masks'])
            mAPs.append(AP)

        return np.array(mAPs)
