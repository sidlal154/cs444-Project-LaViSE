import json
import os
import pickle
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

# Define the ImageNet normalization parameters.
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
unnormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

# Define the transforms for the different datasets.
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]),
}


class _DatasetAbstract(Dataset):
    """Abstract class for a dataset to be used for the LaViSE framework."""

    def __init__(self, root: str, cat_mappings_file: str,
                 transform: Optional[Callable],
                 filter_width: int, filter_height: int, cat_frac: float = 1.0):
        """
        Initialize an object which can load images and annotations.

        Args:
            root (str): The root directory where the dataset's images were
                downloaded to.
            cat_mappings_file (str): Path to the pickle file containing the
                category mappings.
            transform (Optional[Callable]): Transform to apply to the images.
            filter_width (int): Width that the segmentation mask should be
                resized to.
            filter_height (int): Height that the segmentation mask should be
                resized to.
            cat_frac (float, optional): The fraction of categories that should
                be used. Defaults to 1.0.
        """
        self.root = root
        self.transform = transform
        self.mask_transform = self._create_mask_transform(filter_width,
                                                          filter_height)

        # Load the categories. `self.cat_mappings` is a dictionary that
        # contains the following entries:
        # - "stoi" (String TO Index): a dictionary that maps a category token
        #   to its GloVe index in the embedding matrix.
        # - "itos" (Index TO String): a dictionary that maps a GloVe index in
        #   the embedding matrix to its category token.
        with open(cat_mappings_file, "rb") as f:
            self.cat_mappings = pickle.load(f)

        # Select a subset of the categories.
        self._select_category_subset(cat_frac)

        # Define a dict that maps a category token to its index in the
        # one-hot target vector.
        self.cat_token_to_vector_idx = {
            cat_token: vector_idx
            for vector_idx, cat_token in enumerate(
                sorted(self.cat_mappings["stoi"]))
        }

    def _create_mask_transform(self, filter_width: int, filter_height: int) \
            -> transforms.Compose:
        """
        Create a transform to resize a mask.

        Args:
            filter_width (int): Width of the mask.
            filter_height (int): Height of the mask.

        Returns:
            transforms.Compose: Mask transform.
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop(224),
            transforms.Resize([filter_height, filter_width],
                              interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def _select_category_subset(self, cat_frac: float = 1.0):
        """
        Select a subset of the categories.

        Args:
            cat_frac (float, optional): The fraction of categories that should
                be used. Defaults to 1.0.
        """
        if cat_frac < 1.0:
            cats_selected = np.random.choice(
                list(self.cat_mappings["stoi"]),
                round(len(self.cat_mappings["stoi"]) * cat_frac),
                replace=False
            )
            self.cat_mappings["stoi"] = {
                cat_token: glove_idx
                for cat_token, glove_idx in self.cat_mappings["stoi"].items()
                if cat_token in cats_selected
            }
            self.cat_mappings["itos"] = {
                glove_idx: cat_token
                for glove_idx, cat_token in self.cat_mappings["itos"].items()
                if cat_token in cats_selected
            }


class _VisualGenomeAbstract(_DatasetAbstract):
    """Abstract class for the Visual Genome dataset."""

    def __init__(self, objs_file: str, root: str, cat_mappings_file: str,
                 transform: Optional[Callable],
                 filter_width: int, filter_height: int, cat_frac: float = 1.0):
        """
        Initialize an object which can load VG images and annotations.

        Args:
            objs_file (str): Path to the json file containing the preprocessed
                object data.
            root (str): The root directory where the VG images were
                downloaded to.
            cat_mappings_file (str): Path to the pickle file containing the
                category mappings.
            transform (Optional[Callable]): Transform to apply to the images.
            filter_width (int): Width that the segmentation mask should be
                resized to.
            filter_height (int): Height that the segmentation mask should be
                resized to.
            cat_frac (float, optional): The fraction of categories that should
                be used. Defaults to 1.0.
        """
        super().__init__(root, cat_mappings_file, transform,
                         filter_width, filter_height, cat_frac)

        # Load the VG samples.
        with open(objs_file, "r") as f:
            self.samples = json.load(f)

        # Filter the samples to only include categories that have a GloVe index
        # associated with them in the category mappings dictionary.
        samples_processed = []
        for sample in self.samples:
            objects = {cat_token: bboxes
                       for cat_token, bboxes in sample["objects"].items()
                       if cat_token in self.cat_mappings["stoi"]}
            if len(objects) > 0:
                samples_processed.append({"image_id": sample["image_id"],
                                          "objects": objects})
        self.samples = samples_processed


class VisualGenomeImages(_VisualGenomeAbstract):
    """Visual Genome dataset that returns images."""

    def __len__(self) -> int:
        """
        Get the amount of samples in the dataset.

        Returns:
            int: Amount of images in the VG dataset.
        """
        return len(self.samples)

    def __getitem__(self, index: int) \
            -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        """
        Get a single image from the dataset, along with all its instances.
        An "instance" refers to a specific occurrence of an object in an image.

        Args:
            index (int): Index of the image to be returned.

        Returns:
            Tuple:
                Image.Image: Image from the dataset.
                    Shape: [3, 224, 224].
                torch.Tensor: GloVe category index for each instance.
                    Shape: [num_instances].
                torch.Tensor: Mask for each instance.
                    Shape: [num_instances, 1, filter_width, filter_height].
        """
        # Get the image and apply augmentations.
        path = f"VG_100K/{self.samples[index]['image_id']}.jpg"
        img_og = Image.open(os.path.join(self.root, path)).convert("RGB")
        img = self.transform(img_og) if self.transform is not None else img_og

        # Create the target category vector and instance masks.
        targets = []
        masks = []
        for cat_token, bboxes in self.samples[index]["objects"].items():
            for bbox in bboxes:
                # Add the GloVe index to the list of targets.
                targets.append(self.cat_mappings["stoi"][cat_token])
                # Add the mask to the list of masks.
                mask = torch.zeros(img_og.size)
                mask[bbox["y"]:bbox["y"] + bbox["h"],
                     bbox["x"]:bbox["x"] + bbox["w"]] = 1
                masks.append(self.mask_transform(mask))
        targets = torch.tensor(targets)
        masks = torch.stack(masks)

        return img, targets, masks


class VisualGenomeInstances(_VisualGenomeAbstract):
    """Visual Genome dataset that returns instances."""

    def __init__(self, objs_file: str, root: str, cat_mappings_file: str,
                 transform: Optional[Callable],
                 filter_width: int, filter_height: int, cat_frac: float = 1.0):
        """
        Initialize an object which can load VG images and annotations.

        Args:
            objs_file (str): Path to the json file containing the preprocessed
                object data.
            root (str): The root directory where the VG images were
                downloaded to.
            cat_mappings_file (str): Path to the pickle file containing the
                category mappings.
            transform (Optional[Callable]): Transform to apply to the images.
            filter_width (int): Width that the segmentation mask should be
                resized to.
            filter_height (int): Height that the segmentation mask should be
                resized to.
            cat_frac (float, optional): The fraction of categories that should
                be used. Defaults to 1.0.
        """
        super().__init__(objs_file, root, cat_mappings_file, transform,
                         filter_width, filter_height, cat_frac)

        # Create a mapping from instances to tuples of indices.
        self.instance2idx = []
        for sample_idx, sample in enumerate(self.samples):
            for cat_token, bboxes in sample["objects"].items():
                for bbox_idx in range(len(bboxes)):
                    self.instance2idx.append((sample_idx, cat_token, bbox_idx))

    def __len__(self) -> int:
        """
        Get the amount of samples in the dataset.

        Returns:
            int: Amount of instances in the VG dataset.
        """
        return len(self.instance2idx)

    def __getitem__(self, index: int) \
            -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        """
        Get a single instance from the dataset.
        An "instance" refers to a specific occurrence of an object in an image.

        Args:
            index (int): Index of the instance to be returned.

        Returns:
            Tuple:
                Image.Image: Image in which the instance is found.
                    Shape: [3, 224, 224].
                torch.Tensor: One-hot target category vector.
                    Shape: [num_categories].
                torch.Tensor: Instance mask.
                    Shape: [1, filter_width, filter_height].
        """
        sample_idx, cat_token, bbox_idx = self.instance2idx[index]

        # Get the image and apply augmentations.
        path = f"VG_100K/{self.samples[sample_idx]['image_id']}.jpg"
        img_og = Image.open(os.path.join(self.root, path)).convert("RGB")
        img = self.transform(img_og) if self.transform is not None else img_og

        # Create a one-hot target vector for the instance.
        target = torch.zeros((len(self.cat_mappings["stoi"]),))
        target[self.cat_token_to_vector_idx[cat_token]] = 1

        # Load the segmentation mask for the instance.
        bbox = self.samples[sample_idx]["objects"][cat_token][bbox_idx]
        mask = torch.zeros(img_og.size)
        mask[bbox["y"]:bbox["y"] + bbox["h"],
             bbox["x"]:bbox["x"] + bbox["w"]] = 1
        mask = self.mask_transform(mask)

        return img, target, mask


class _CocoAbstract(_DatasetAbstract):
    """Abstract class for the COCO dataset."""

    def __init__(self, ann_file: str, root: str, cat_mappings_file: str,
                 transform: Optional[Callable],
                 filter_width: int, filter_height: int, cat_frac: float = 1.0):
        """
        Initialize an object which can load COCO images and annotations.

        Args:
            ann_file (str): Path to json file containing instance annotations.
            root (str): The root directory where the dataset's images were
                downloaded to.
            cat_mappings_file (str): Path to the pickle file containing the
                category mappings.
            transform (Optional[Callable]): Transform to apply to the images.
            filter_width (int): Width that the segmentation mask should be
                resized to.
            filter_height (int): Height that the segmentation mask should be
                resized to.
            cat_frac (float, optional): The fraction of categories that should
                be used. Defaults to 1.0.
        """
        super().__init__(root, cat_mappings_file, transform,
                         filter_width, filter_height, cat_frac)

        # Load the COCO samples.
        self.coco = COCO(ann_file)


class CocoImages(_CocoAbstract):
    """COCO dataset that returns images."""

    def __init__(self, ann_file: str, root: str, cat_mappings_file: str,
                 transform: Optional[Callable],
                 filter_width: int, filter_height: int, cat_frac: float = 1.0):
        """
        Initialize an object which can load COCO images and annotations.

        Args:
            ann_file (str): Path to json file containing instance annotations.
            root (str): The root directory where the dataset's images were
                downloaded to.
            cat_mappings_file (str): Path to the pickle file containing the
                category mappings.
            transform (Optional[Callable]): Transform to apply to the images.
            filter_width (int): Width that the segmentation mask should be
                resized to.
            filter_height (int): Height that the segmentation mask should be
                resized to.
            cat_frac (float, optional): The fraction of categories that should
                be used. Defaults to 1.0.
        """
        super().__init__(ann_file, root, cat_mappings_file, transform,
                         filter_width, filter_height, cat_frac)

        # Get the IDs of all images in the dataset.
        self.img_ids = sorted(self.coco.imgs)

        # Remove samples that don't have any annotations.
        self.img_ids = [img_id for img_id in self.img_ids
                        if len(self.coco.getAnnIds(img_id)) > 0]

        # Filter the samples to only include categories that have a GloVe index
        # associated with them in the category mappings dictionary.
        self.img_ids = [
            img_id
            for img_id in self.img_ids
            if any(
                cat_token in self.cat_mappings["stoi"]
                for cat in self.coco.loadCats([
                    ann["category_id"]
                    for ann in self.coco.loadAnns(self.coco.getAnnIds(img_id))
                ])
                for cat_token in cat["name"].split()
            )
        ]

    def __len__(self) -> int:
        """
        Get the amount of samples in the dataset.

        Returns:
            int: Amount of images in the dataset.
        """
        return len(self.img_ids)

    def __getitem__(self, index: int) \
            -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        """
        Get a single image from the dataset, along with all its instances.
        An "instance" refers to a specific occurrence of an object in an image.

        Args:
            index (int): Index of the image to be returned.

        Returns:
            Tuple:
                Image.Image: Image from the dataset.
                    Shape: [3, 224, 224].
                torch.Tensor: GloVe category index for each instance.
                    Shape: [num_instances].
                torch.Tensor: Mask for each instance.
                    Shape: [num_instances, 1, filter_width, filter_height].
        """
        # Get the instance annotations from the dataset. An instance annotation
        # is a dictionary that contains (among others) the following entries:
        # - "image_id": the id of the image that contains the instance.
        # - "category_id": the id of the category that the instance belongs to.
        anns = self.coco.loadAnns(self.coco.getAnnIds(self.img_ids[index]))

        # Get the image and apply augmentations.
        path = self.coco.loadImgs(self.img_ids[index])[0]["file_name"]
        img_og = Image.open(os.path.join(self.root, path)).convert("RGB")
        img = self.transform(img_og) if self.transform is not None else img_og

        # Create the target category vector and instance masks.
        targets = []
        masks = []
        for ann in anns:
            mask = self.mask_transform(self.coco.annToMask(ann))
            cat = self.coco.loadCats(ann["category_id"])[0]
            for cat_token in cat["name"].split():
                if cat_token in self.cat_mappings["stoi"]:
                    # Add the GloVe index to the list of targets.
                    targets.append(self.cat_mappings["stoi"][cat_token])
                    # Add the mask to the list of masks.
                    masks.append(mask)
        targets = torch.tensor(targets)
        masks = torch.stack(masks)

        return img, targets, masks


class CocoInstances(_CocoAbstract):
    """COCO dataset that returns instances."""

    def __init__(self, ann_file: str, root: str, cat_mappings_file: str,
                 transform: Optional[Callable],
                 filter_width: int, filter_height: int, cat_frac: float = 1.0):
        """
        Initialize an object which can load COCO images and annotations.

        Args:
            ann_file (str): Path to json file containing instance annotations.
            root (str): The root directory where the dataset's images were
                downloaded to.
            cat_mappings_file (str): Path to the pickle file containing the
                category mappings.
            transform (Optional[Callable]): Transform to apply to the images.
            filter_width (int): Width that the segmentation mask should be
                resized to.
            filter_height (int): Height that the segmentation mask should be
                resized to.
            cat_frac (float, optional): The fraction of categories that should
                be used. Defaults to 1.0.
        """
        super().__init__(ann_file, root, cat_mappings_file, transform,
                         filter_width, filter_height, cat_frac)

        # Get the IDs of all instances in the dataset.
        self.instance_ids = sorted(self.coco.anns)

        # Filter the samples to only include categories that have a GloVe index
        # associated with them in the category mappings dictionary.
        self.instance_ids = [
            instance_id
            for instance_id in self.instance_ids
            if any(
                cat_token in self.cat_mappings["stoi"]
                for cat_token in self.coco.loadCats(
                    self.coco.loadAnns(instance_id)[0]["category_id"]
                )[0]["name"].split()
            )
        ]

    def __len__(self) -> int:
        """
        Get the amount of samples in the dataset.

        Returns:
            int: Amount of instances in the dataset.
        """
        return len(self.instance_ids)

    def __getitem__(self, index: int) \
            -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        """
        Get a single instance from the dataset.
        An "instance" refers to a specific occurrence of an object in an image.

        Args:
            index (int): Index of the instance to be returned.

        Returns:
            Tuple:
                Image.Image: Image in which the instance is found.
                    Shape: [3, 224, 224].
                torch.Tensor: Multiple-hot target category vector.
                    Shape: [num_categories].
                torch.Tensor: Instance mask.
                    Shape: [1, filter_width, filter_height].
        """
        # Get the instance annotation from the dataset. An instance annotation
        # is a dictionary that contains (among others) the following entries:
        # - "image_id": the id of the image that contains the instance.
        # - "category_id": the id of the category that the instance belongs to.
        ann = self.coco.loadAnns(self.instance_ids[index])[0]

        # Get the image and apply augmentations.
        path = self.coco.loadImgs(ann["image_id"])[0]["file_name"]
        img_og = Image.open(os.path.join(self.root, path)).convert("RGB")
        img = self.transform(img_og) if self.transform is not None else img_og

        # Create a multiple-hot target vector for the instance.
        target = torch.zeros((len(self.cat_mappings["stoi"]),))
        cat = self.coco.loadCats(ann["category_id"])[0]
        for cat_token in cat["name"].split():
            if cat_token in self.cat_mappings["stoi"]:
                target[self.cat_token_to_vector_idx[cat_token]] = 1

        # Load the segmentation mask for the instance.
        mask = self.mask_transform(self.coco.annToMask(ann))

        return img, target, mask
