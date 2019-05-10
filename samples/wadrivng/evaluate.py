import os
import sys
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/koukouzas/CODE/Mask_RCNN")  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib

# Import WAD config
sys.path.append(os.path.join(ROOT_DIR, "samples/wadrivng/"))  # To find local version
import wad

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
WAD_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_wad_0040.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "data")

config = wad.WadInferenceConfig()
config.display()

# Load Wad dataset
dataset = wad.WadDataset()
dataset.load_wad(IMAGE_DIR, "val")

# Must call before using dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=MODEL_DIR)

print("Loading weights", WAD_MODEL_PATH)
model.load_weights(WAD_MODEL_PATH, by_name=True)

# Compute VOC-style Average Precision
image_ids = dataset.image_ids
APs = []
for image_id in image_ids:
    # Load image
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config,
                               image_id, use_mini_mask=False)
    # Run object detection
    results = model.detect([image], verbose=0)
    # Compute AP
    r = results[0]
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                          r['rois'], r['class_ids'], r['scores'], r['masks'])
    APs.append(AP)
print("mAP @ IoU=50: ", np.mean(APs))