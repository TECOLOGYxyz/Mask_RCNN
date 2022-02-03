"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""


import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import time
import csv
from imgaug import augmenters as iaa # Uncomment to use New Augmentation
#import imgaug # Comment to use New Augmentation

if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

from mrcnn import Custom_Utilities # These are custom utility functions written by Hjalte

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
RESULTS_DIR = os.path.join(ROOT_DIR, "results/dryas/")

num_epochs = 800


############################################################
#  Configurations
############################################################


class DryasConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "2020_07_01_NorwayAnnotations_TheUltimateModel"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + (flower + syrphidae)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class DryasDataset(utils.Dataset):

    def load_dryas(self, dataset_dir, subset):
        """Load a subset of the Dryas dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("object", 1, "Flower") #changed from self.add_class("object", 1, "Flower") 
        #self.add_class("object", 2, "Syrphidae")

        # Train or validation dataset?
        assert subset in ["train", "val", "test", "path"]

        if subset == "path":
            print("You are running detections on a path file, so there is no dataset_dir. Supply the path to the path file in the dataset argument.")

        else: 
            dataset_dir = os.path.join(dataset_dir, subset)
            print("Here's the dataset_dir")
            print(dataset_dir)
        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list

        if subset == "train" or subset == "val":
            annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
            annotations = list(annotations.values())  # don't need the dict keys

            # The VIA tool saves images in the JSON even if they don't have any
            # annotations. Skip unannotated images.
            annotations = [a for a in annotations if a['regions']]
            image_ids = next(os.walk(dataset_dir))[1]
            # Add images
            for a in annotations:
                # Get the x, y coordinaets of points of the polygons that make up
                # the outline of each object instance. These are stores in the
                # shape_attributes (see json format above)
                # The if condition is needed to support VIA versions 1.x and 2.x.
                # if type(a['regions']) is dict:
                #     polygons = [r['shape_attributes'] for r in a['regions'].values()]
                # else:
                polygons = [r['shape_attributes'] for r in a['regions']] 
                
                class_names_str  = [r['region_attributes']['object'] for r in a['regions']]
                class_name_nums = []
                for i in  class_names_str:
                    if i == 'Flower': # Changed from if i == 'Flower'
                        class_name_nums.append(1)
                    #if i == 'Syrphidae':
                     #   class_name_nums.append(2)

                # load_mask() needs the image size to convert polygons to masks.
                # Unfortunately, VIA doesn't include it in JSON, so we must read
                # the image. This is only managable since the dataset is tiny.
                image_path = os.path.join(dataset_dir, a['filename'])
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                self.add_image(
                    "object",
                    image_id=a['filename'],  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons,
                    class_list = np.array(class_name_nums)) #UNSURE IF  I CAN JUST ADD THIS  HERE. OTHERWISE NEED  TO MODIFY DATASET UTIL
       


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object" :
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        
        class_array = info['class_list']
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s

        #this function returns the masks as normal,  the class array of 3 classes
        return mask.astype(np.bool), class_array
    
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = DryasDataset()
    dataset_train.load_dryas(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DryasDataset()
    dataset_val.load_dryas(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    
    ####### Try this augmentation scheme instead of just flipping!
    max_augs = 3
    augmentation = iaa.SomeOf((0, max_augs), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([ iaa.Affine(rotate = 30 * i) for i in range(0, 12) ]),
    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
    iaa.Add((-40, 40)),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 5.0))])

    model.train(dataset_train, dataset_val,
    learning_rate = config.LEARNING_RATE,
    epochs = num_epochs,
    augmentation = augmentation,
    layers="all")

########################
    
    #### Old training scheme
    #print("Training network all!")

    #augmentation = imgaug.augmenters.Fliplr(0.5)
    
    #Original training scheme:
   # model.train(dataset_train, dataset_val,
    #            learning_rate=config.LEARNING_RATE,
    #            epochs=num_epochs,
    #            layers='all', augmentation=augmentation)

  #########


    # #Alternative training scheme:
    # #Training - Stage 1
    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=10,
    #             layers='heads',
    #             augmentation=augmentation)

    # # Training - Stage 2
    # # Finetune layers from ResNet stage 4 and up
    # print("Fine tuning Resnet stage 4 and up")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=5,
    #             layers='4+',
    #             augmentation=augmentation)

    # # Training - Stage 3
    # # Fine tune all layers
    # print("Fine tuning all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE / 10,
    #             epochs=5,
    #             layers='all',
    #             augmentation=augmentation)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

# def mask_to_rle(image_id, mask, scores):
#     "Encodes instance masks to submission format."
#     assert mask.ndim == 3, "Mask must be [H, W, count]"
#     # If mask is empty, return line with image ID only
#     if mask.shape[-1] == 0:
#         return "{},".format(image_id)
#     # Remove mask overlaps
#     # Multiply each instance mask by its score order
#     # then take the maximum across the last dimension
#     order = np.argsort(scores)[::-1] + 1  # 1-based descending
#     mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
#     # Loop over instance masks
#     lines = []
#     for o in order:
#         m = np.where(mask == o, 1, 0)
#         # Skip if empty
#         if m.sum() == 0.0:
#             continue
#         rle = rle_encode(m)
#         lines.append("{}, {}".format(image_id, rle))
#     return "\n".join(lines)

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Run model detection and generate the color splash effect")
        print("Running on {}".format(args.image))
        # Read image
        print("Read image")
        image = skimage.io.imread(args.image)
        print("Detect objects")
        r = model.detect([image], verbose=1)[0]
        print("Color splash")
        splash = color_splash(image, r['masks'])
        print("Save output")
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
        print("Done with handling image")
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

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
    # Convert second index in each pair to length
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
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


def mask_to_rle(image_id, mask, scores):
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
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
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

    predict_dir = os.path.join(submit_dir, "predictions")
    os.makedirs(predict_dir)


    # Read dataset
    dataset = DryasDataset()
    dataset.load_dryas(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    boxes = []
    


    ###

    # Processing times
    tot_start = time.time()

    proc_times = []
    img_ids = []
    x_mins = []
    y_mins = []
    x_maxs = []
    y_maxs = []



    if subset == "path":

        scale_percent = 12.5

        tot_start = time.time()

        proc_times = []
        img_ids = []
        x_mins = []
        y_mins = []
        x_maxs = []
        y_maxs = []


        with open(dataset_dir, newline='') as csvfile:
            path_file = csv.reader(csvfile)


            for path_to_image in path_file:
                path_to_image = path_to_image[0]
                print("Here's the image path: ", path_to_image)

                start = time.time()
                print(path_to_image)
                #Load image and run detection
                image = skimage.io.imread(path_to_image)
                image = Custom_Utilities.DownscaleImage(scale_percent,image) # Downscale image using the function from the custom utilities.


                r = model.detect([image], verbose=0)[0]
                # Encode image to RLE. Returns a string of multiple lines
                source_id = path_to_image # dataset.image_info[image_id] #["id"]
                rle = mask_to_rle(source_id, r["masks"], r["scores"])
                submission.append(rle)

                box = utils.extract_bboxes(r["masks"])
                print(type(box))
                if not box.any():
                    print(path_to_image, " empty!")
                    x_min = 0
                    y_min = 0
                    x_max = 0
                    y_max = 0

                    x_mins.append(x_min)
                    y_mins.append(y_min)
                    x_maxs.append(x_max)
                    y_maxs.append(y_max)

                    img_ids.append(path_to_image)
                else:   
                    for b in box:
                        x_min = b[1]
                        y_min = b[0]
                        x_max = b[3]
                        y_max = b[2]

                        x_mins.append(x_min)
                        y_mins.append(y_min)
                        x_maxs.append(x_max)
                        y_maxs.append(y_max)
                        
                        img_ids.append(path_to_image)


                visualize.display_instances(
                    image, r['rois'], r['masks'], r['class_ids'],
                    dataset.class_names, r['scores'],
                    show_bbox=True, show_mask=False,
                    title="Predictions")
                plt.savefig("{}_Prediction.JPG".format(os.path.join(predict_dir, os.path.basename(path_to_image))))
                plt.close('all')
                end = time.time()
                print("Processing time for the image: ", end-start)
                proc_times.append(end-start)
                #img_ids.append(image_id)


    else:
        test_dir = os.path.join(dataset_dir, subset)
        test_ids = next(os.walk(test_dir))[2]
        print("Length of test_ids", len(test_ids))


        for image_id in test_ids:
            start = time.time()
            print(image_id)
            #Load image and run detection
            path_to_image = os.path.join(test_dir,image_id)
            image = skimage.io.imread(path_to_image)
            #image = dataset.load_image(image_id)
        #     # Detect objects
            r = model.detect([image], verbose=0)[0]
            # Encode image to RLE. Returns a string of multiple lines
            source_id = image_id # dataset.image_info[image_id] #["id"]
            rle = mask_to_rle(source_id, r["masks"], r["scores"])
            submission.append(rle)

            #Append bounding boxes to list (old version)
            #box = utils.extract_bboxes(r["masks"])
            #box = ','.join(str(v) for v in box)
            #box = image_id+", "+box
            #boxes.append(box)

            #Append bounding boxes to list (new version)
            box = utils.extract_bboxes(r["masks"])
            print(type(box))
            if not box.any():
                print(image_id, " empty!")
                x_min = 0
                y_min = 0
                x_max = 0
                y_max = 0

                x_mins.append(x_min)
                y_mins.append(y_min)
                x_maxs.append(x_max)
                y_maxs.append(y_max)

                img_ids.append(image_id)
            else:   
                for b in box:
                    x_min = b[1]
                    y_min = b[0]
                    x_max = b[3]
                    y_max = b[2]

                    x_mins.append(x_min)
                    y_mins.append(y_min)
                    x_maxs.append(x_max)
                    y_maxs.append(y_max)
                    
                    img_ids.append(image_id)

        #print("Box:", box)
        #bbs.assign(bboxes=box)
        #bbs['imageID'] = image_id
        #print(bbs)
        #box = ','.join(str(v) for v in box)
        #box = image_id+", "+box
        
        #box = [image_id+","+ b for b in box]
        #print("After lc:",box)
        #boxes.append(box)

        # Save image with masks




            #visualize.display_instances(
            #    image, r['rois'], r['masks'], r['class_ids'],
            #    dataset.class_names, r['scores'],
            #    show_bbox=True, show_mask=False,
             #   title="Predictions")
           # plt.savefig("{}_Prediction.JPG".format(path_to_image))
           # plt.close('all')
            end = time.time()
            print("Processing time for the image: ", end-start)
            proc_times.append(end-start)
            #img_ids.append(image_id)







        tot_end = time.time()
        print("Total prossesing time: ", tot_end-tot_start)
        print("Prossesing time per image: ", (tot_end-tot_start)/(len(test_ids)-1))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)

    file_path_proc = os.path.join(submit_dir, "proc_times_per_image.csv")
    with open(file_path_proc, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(proc_times,img_ids))

    #New saver for bboxes
    file_path_proc = os.path.join(submit_dir, "submit_boxes_NewFormat.csv")
    with open(file_path_proc, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(img_ids, x_mins, y_mins, x_maxs, y_maxs))
  #  file_path_proc = os.path.join(submit_dir, "proc_times_per_image.csv")
   # with open(file_path_proc, "w") as f:
    #    f.write(proc_times_per_image)

    boxes = "ImageId,Boxes\n" + "\n".join(boxes)
    file_path = os.path.join(submit_dir, "submit_boxes.csv")
    with open(file_path, "w") as f:
        f.write(str(boxes))
    print("Saved to ", submit_dir)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect flowers.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dryas/dataset/",
                        help='Directory of the dryas dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DryasConfig()
    else:
        class InferenceConfig(DryasConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
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
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash' or 'detect'".format(args.command))
