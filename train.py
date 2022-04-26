import os
import sys
import json
import numpy as np
import skimage.draw
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')  # prevents warming from displaying in the terminal

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
config.gpu_options.polling_inactive_delay_msecs = 10
session = tf.compat.v1.Session(config=config)

ROOT = "C:\\Users\\Matt\\PycharmProjects\\banana1"  # Root directory (need to change)
sys.path.append(ROOT)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

COCO_WEIGHTS_PATH = os.path.join(ROOT, "mask_rcnn_coco.h5")  # path to COCO trained weights

LOGS_PATH = os.path.join(ROOT, "logs")  # Path to the logs folder where each epoch will save files to

class LeafConfig(Config):
    NAME = "object"
    IMAGES_PER_GPU = 1  # GPU has only 8 gigabytes of ram and could handle 1-3 images, however 1 performs fastest for the NVIDIA RTX3070ti
    NUM_CLASSES = 2  # background and leaves
    STEPS_PER_EPOCH = 10  # does not have an effect on trainng data of fixed size
    DETECTION_MIN_CONFIDENCE = 0.9
    IMAGE_MAX_DIM = 512 # Reduced max image dimension to save of GPU memory (8Gb NVIDIA RTX3070Ti)

class LeafDataset(utils.Dataset):

    def load_Leaves(self, dataset_dir, subset):

        self.add_class("object", 1, "Leaf")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        if subset == "train":
            print(dataset_dir)
            print(subset)
            annotations = json.load(open(os.path.join(dataset_dir, "train_json.json")))
        else:
            annotations = json.load(open(os.path.join(dataset_dir, "val_json.json")))

        annotation = list(annotations.values())
        annotation = [x for x in annotation if x['regions']]
        print(annotation[0])
        count = 0
        for x in annotation:
            count = count + 1
            print(count)
            polygons = [r['shape_attributes'] for r in x['regions']]
            objects = [s['region_attributes']['name'] for s in x['regions']]
            print("objects:", objects)
            name_dict = {"Leaf": 1}

            num_ids = [name_dict[x] for x in objects]

            print("numids", num_ids)
            image_path = os.path.join(dataset_dir, x["filename"])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",
                image_id=x["filename"],
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                num_ids=num_ids
            )

    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        num_ids = info["num_ids"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # set pixels inside poloygons to 1
            rr, cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])

            mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):

        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    datasetTrain = LeafDataset()
    datasetTrain.load_Leaves(os.path.join(ROOT, "Dataset"), "train")
    datasetTrain.prepare()

    datasetVal = LeafDataset()
    datasetVal.load_Leaves(os.path.join(ROOT, "Dataset"), "val")
    datasetVal.prepare()

    print("Training network all layers")
    model.train(datasetTrain,
                datasetVal,
                learning_rate=config.LEARNING_RATE,
                epochs=1000,
                layers="all")


config = LeafConfig()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=LOGS_PATH)

weights_path = COCO_WEIGHTS_PATH

if not os.path.exists(weights_path):
    utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])

train(model)
