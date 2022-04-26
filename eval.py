import warnings
warnings.filterwarnings('ignore')
import os
import json
import numpy as np
import skimage.draw
import time
import matplotlib.pyplot as plt
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from mrcnn.config import Config
from mrcnn import model as modellib, utils

ROOT = "C:\\Users\\Matt\\PycharmProjects\\banana1"

LOG_PATH = os.path.join(ROOT, "logs")

class LeafConfig(Config):

        NAME = "object"
        IMAGES_PER_GPU = 1  # GPU has only 8 gigabytes of ram and could handle 1-3 images, however 1 performs fastest for the NVIDIA RTX3070ti
        NUM_CLASSES = 2  # background and leaves
        STEPS_PER_EPOCH = 10  # does not have an effect on trainng data of fixed size
        DETECTION_MIN_CONFIDENCE = 0.9
        IMAGE_MAX_DIM = 512  # Reduced max image dimension to save of GPU memory (8Gb NVIDIA RTX3070Ti)

class LeafDataset(utils.Dataset):

    def load_Leaves(self, dataset_dir, subset):

        self.add_class("object", 1, "Leaf")

        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        if subset == "train":
            annotations = json.load(open(os.path.join(dataset_dir, "train_json.json")))
        else:
            annotations = json.load(open(os.path.join(dataset_dir, "test_json.json")))

        annotation = list(annotations.values())
        annotation = [a for a in annotation if a['regions']]

        for a in annotation:
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes']['name'] for s in a['regions']]
            name_dict = {"Leaf": 1}

            num_ids = [name_dict[a] for a in objects]

            image_path = os.path.join(dataset_dir, a["filename"])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",
                image_id=a["filename"],
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


#Merges a set of Mask into one mask
def merged_mask(masks):

    n = masks.shape[2]
    if n != 0:
        merged_mask = np.zeros((masks.shape[0], masks.shape[1]))
        for i in range(n):
            merged_mask += masks[..., i]
        merged_mask = np.asarray(merged_mask, dtype=np.uint8)
        return merged_mask
    return masks[:, :, 0]

#computes the iou for a mask
def compute_iou(predict_mask, gt_mask):

    intersection = np.logical_and(predict_mask, gt_mask)  # *
    union = np.logical_or(predict_mask, gt_mask)  # +
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

TEST_MODE = "inference"

config = LeafConfig() #Creates the config

#Creates the model in inference mode and loads it
model = modellib.MaskRCNN(mode='inference', config=config, model_dir=LOG_PATH)
model_path = model.find_last()

#loads the weights from the 440th model trained
MODEL_PATH = "C:\\Users\\Matt\\PycharmProjects\\banana1\\logs\\object20220424T0837\\mask_rcnn_object_0440.h5" #path to the model
model.load_weights(MODEL_PATH, by_name=True)

#create and prep the training set
datasetTrain = LeafDataset()
datasetTrain.load_Leaves(os.path.join(ROOT, "Dataset"), "train")
datasetTrain.prepare()


image_id = datasetTrain.image_ids
IoUCount = 0
count = 0
mAP = []
timeTrain = 0
print(image_id)
for image_id in image_id:
    count = count + 1
    image = datasetTrain.load_image(image_id)
    mask, class_ids = datasetTrain.load_mask(image_id)

    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(datasetTrain, config, image_id, use_mini_mask=False)

    start = time.process_time()
    results = model.detect([original_image], verbose=1)
    end = time.process_time()
    timeTrain = timeTrain + (end -start)
    r = results[0]
    res = cv2.resize(merged_mask(mask), dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    IoUCount = IoUCount +  compute_iou(res, merged_mask(r['masks']))

    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"],
                                                         r["scores"], r["masks"],iou_threshold=0.80)#iou_threshold can be changed
    mAP.append(AP)


#Creates and prepares the test set
datasetTest = LeafDataset()
datasetTest.load_Leaves(os.path.join(ROOT, "Dataset"), "test")
datasetTest.prepare()


image_idT = datasetTest.image_ids

IoUCountT = 0
countT = 0
mAPT = []
timeTest = 0
print(image_idT)
for image_idT in image_idT:
    countT = countT + 1

    imageT = datasetTest.load_image(image_idT)
    mask, class_ids = datasetTest.load_mask(image_idT)

    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(datasetTest, config, image_idT, use_mini_mask=False)

    start = time.process_time()
    results = model.detect([original_image], verbose=1)
    end = time.process_time()
    timeTest = timeTest = (end - start)
    r = results[0]
    res = cv2.resize(merged_mask(mask), dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    IoU = compute_iou(res, merged_mask(r['masks']))
    IoUCountT = IoUCountT + IoU
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"],
                                                         r["scores"], r["masks"],iou_threshold=0.80) #iou_threshold can be changed
    mAPT.append(AP)


IoUAvgTrain = IoUCount / count
APTrain = np.mean(mAP)
print("train results")
print("IoU Average Train = ",IoUAvgTrain)
print("Average Precision Train = ",APTrain)
print("Inference Time Train = ", (timeTrain/count))

APTest = np.mean(mAPT)
IoUAvgTest = IoUCountT / countT
print("test results")
print("IoU Average Test = ",IoUAvgTest)
print("Average Precision Test = ",APTest)
print("Inference Time Test = ",timeTest/count)


