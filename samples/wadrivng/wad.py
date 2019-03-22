"""
Mask R-CNN
Configurations and data loading code for 
autonomous driving segmentation dataset from the
Kaggle 2018 CVPR 2018 WAD Video Segmentation Challenge
https://www.kaggle.com/c/cvpr-2018-autonomous-driving

Licensed under the MIT License (see LICENSE for details)
Written by Giorgos Koukouzas

------------------------------------------------------------

    # Train a new model starting from pre-trained COCO weights
    python3 wad.py train --dataset=/path/to/wad/dataset/ --model=coco --subset="train"

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 wad.py train --dataset=/path/to/wad/dataset/ --model=imagenet 

    # Continue training a model that you had trained earlier
    python3 wad.py train --dataset=/path/to/wad/dataset/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 wad.py train --dataset=/path/to/wad/dataset/ --model=last
"""
 
# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import logging
import sys
import datetime
import numpy as np
import skimage.io
from pathlib import Path
from imgaug import augmenters as iaa # https://github.com/aleju/imgaug (pip3 install imgaug)
import tensorflow as tf
from keras.callbacks import Callback

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/koukouzas/CODE/Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils 
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/wad/")

# Define the objects of the dataset
label_to_name = {33: 'car', 
                 34: 'motorcycle', 35: 'bicycle', 36: 'pedestrian', 
                 38: 'truck', 39: 'bus', 40: 'tricycle'} 

label_to_class = {33:1, 34:2, 35:3, 36:4, 38:5, 39:6, 40:7}

class_to_label = {1:33, 2:34, 3:35, 4:36, 5:38, 6:39, 7:40}

############################################################
#  Set CUDA Variables
############################################################

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()

# Dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True

# To log device placement (on which device the operation ran)
# config.log_device_placement = True

# Set this TensorFlow session as the default session for Keras
sess = tf.Session(config=config)
set_session(sess)


############################################################
#  Configurations
############################################################


class WadConfig(Config):
    """Configuration for training on the kaggle 
    autonomous driving segmentation dataset"""

    # Give the configuration a recognizable name
    NAME = "wad"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # The competition has 7 classes.

    # Number of training and validation steps per epoch    
    # STEPS_PER_EPOCH = 100 is what woody used
    # STEPS_PER_EPOCH = 1 if you do want to have one different batch per epoch
    # STEPS_PER_EPOCH = 35300 // IMAGES_PER_GPU
    # VALIDATION_STEPS = 3922 // IMAGES_PER_GPU


    # Skip detections with < 90% confidence
    # DETECTION_MIN_CONFIDENCE = 0.5 # minimum value
    
    # Backbone network architecture
    BACKBONE = "resnet101"

    # Input image resizing
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    # IMAGE_RESIZE_MODE = "none" # consider using crop
    # IMAGE_MIN_SCALE = 2.0

    # Length of square achor side in pixels
    # RPN_ANCHOR_SCALES = (8, 16, 32 ,64, 128)

    # ROIs kepr after non-maximum suppression(training and inference)
    # POST_NMS_ROIS_TRAINING = 1000  # woody used 4000
    # POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals
    # You can increase this during training to generate more proposals.
    # RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 320

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high resolutio images.
    # USE_MINI_MASK = True
    # MINI_MASK_SHAPE = (56,56) # (height, width) of the mini-mask
    # MASK_SHAPE = [28, 28]

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    # TRAIN_ROIS_PER_IMAGE = 600

    # Maximum number of ground truth instanced to use in one image
    # MAX_GT_INSTANCES = 200

    # Maximum number of ground truth instances to use in one image
    # DETECTION_MAX_INSTANCES = 100

    # OPTIMIZER = 'SGD'
    # LEARNING_RATE = 1e-6


class WadInferenceConfig(WadConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "square"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more proposals
    RPN_NMS_THRESHOLD = 0.7

    
############################################################
#  Custom Callbacks
############################################################

'''
This current implementation calculates the average precision for every image 
and then takes the mean of the values. But in papers they calculate the average precision per class and then calculates the mean of the class-wise values.
'''


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


############################################################
#  Dataset
############################################################

class WadDataset(utils.Dataset):
    
    def mask_to_instance(self, mask):
        instances_all = np.unique(mask)
        instances = []
        for i in range(len(instances_all)):
            instance = instances_all[i]
            if np.floor(instance/1000).astype(int) in label_to_name.keys():
                instances.append(instance)
        instances = np.array(instances)
        num_instances = len(instances)
        mask_out = np.zeros([mask.shape[0], mask.shape[1], num_instances], dtype=bool)
        class_ids = np.zeros(num_instances)
        for i in range(num_instances):
            instance = instances[i]
            class_ids[i] = label_to_class[np.floor(instance/1000).astype(int)]
            mask_out[:, :, i] = (mask == instance)
        return mask_out, class_ids.astype(np.int32)


    def load_wad(self, dataset_dir, subset):
        """Load a subset of the Kaggle autonomous driving 2018 dataset.
        dataset_dir: The root directory of the kaggle dataset.
        subset: subset to load: train or val or test
        """

        # Add classes
        for key, name in label_to_name.items():
            class_id = label_to_class[key]
            self.add_class("wad", class_id, name)

        # which subset
        # val : use val 3922 images from /train_val  
        # train : use train 35300 images from /train_val
        # test : use the /data/test dir      
        assert subset in ["train","val", "test"]

        image_dir = ""
        if subset == "test":
            dataset_dir = os.path.join(dataset_dir, subset)
            image_ids = next(os.walk(dataset_dir))[2]
            image_dir = dataset_dir
        else:
            dataset_dir = os.path.join(dataset_dir, "train_val")
            if subset == "val":
                dataset_dir = os.path.join(dataset_dir, subset)
                image_dir = os.path.join(dataset_dir, "image")
                image_ids = next(os.walk(image_dir))[2]
            else:
                dataset_dir = os.path.join(dataset_dir, subset)
                image_dir = os.path.join(dataset_dir, "image")
                image_ids = next(os.walk(image_dir))[2]
        self.dataset_dir = dataset_dir

        print("Base dir : {}".format(dataset_dir))
        print("Data loaded from : {}".format(image_dir))
        print("{} files".format(len(image_ids)))

        # Add the images from the given directory
        for image_id in image_ids:
            image_name = image_id[:-4]
            self.add_image(
                "wad",
                image_id=image_name,
                path=os.path.join(image_dir, image_id))

    def load_mask(self, image_id):
        """Generate instance masks for the given image.
        Returns:
         masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
         class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a wad set image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "wad":
            return super(super.__class__, self).load_mask(image_id)
        
        image_path = self.image_info[image_id]['path']
        image_name = os.path.basename(image_path)
        label_filename = image_name[:-4]
        label_filename += '_instanceIds.png'
        mask_path = os.path.join(self.dataset_dir, "label", label_filename)
        mask_raw = np.array(skimage.io.imread(mask_path)) # shape is (2710, 3384)
        mask, class_ids = self.mask_to_instance(mask_raw)
        return mask, class_ids


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "wad":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################


def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = WadDataset()
    dataset_train.load_wad(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = WadDataset()
    dataset_val.load_wad(dataset_dir, "val")
    dataset_val.prepare()

    # Preparing mAP Callback 
    """
    model_inference = modellib.MaskRCNN(mode="inference", 
                                        config=WadInferenceConfig(),
                                        model_dir=DEFAULT_LOGS_DIR)
    mean_average_precision_callback = MeanAveragePrecisionCallback(model, model_inference, dataset_val, 1, verbose=1)
    """

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])
    
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                augmentation=None,
                epochs=40,
                layers='heads')
                # custom_callbacks=[mean_average_precision_callback])
    '''
    print("Training all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all',
                augmentation=None)
    '''
############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    rle_string = ""
    for n in range(len(rle)):
        rle_string += "{} {}|".format(rle[n, 0],rle[n, 1])
    return rle_string


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""

    rle = list(map(int, rle.split("|")))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores, class_ids):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    # Convert Class ids to Label ids
    label_ids = []
    for class_id in class_ids:
        if class_id in class_to_label.keys():
            label_ids.append(class_to_label.get(class_id))
    # Extract the info needed for kaggle submission
    for o in order:
        m = np.where(mask == o, 1, 0)
        confidence = scores[o-1]
        label_id = label_ids[o-1]
        # Skip if empty
        if m.sum() == 0.0:
            continue
        else:
            pixel_count = m.sum()
        rle = rle_encode(m)
        lines.append("{}, {}, {}, {}, {}".format(image_id, label_id,
                                                 confidence, pixel_count, rle))
    return "\n".join(lines)
    
############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))
    
    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = WadDataset()
    dataset.load_wad(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    submission = "ImageId,LabelId,Confidence,PixelCount,EncodedPixels\n"
    file_path = os.path.join(ROOT_DIR, "submit.csv")
    f = open(file_path, "w")
    f.write(submission)
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects 
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        print("Name of image being processed is {}".format(source_id))
        rle = mask_to_rle(source_id, r["masks"], r["scores"], r["class_ids"])
        # submission.append(rle)
        f.write(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=True, show_mask=True,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    '''
    submission = "ImageId,LabelId,Confidence,PixelCount,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    '''
    print("Saved to ", submit_dir)
    f.close()

############################################################
#  Command Line
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for autonomous driving segmentation.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help='Subset of dataset for train or test')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"
    
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = WadConfig()
    else:
        config = WadInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file 
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        #  Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
    
	# Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
            "Use 'train' or 'detect'".format(args.command))
