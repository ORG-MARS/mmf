# Copyright (c) Facebook, Inc. and its affiliates.

"""
  Run with for example:
  python3 mmf/tools/scripts/features/frcnn/extract_features_frcnn.py \
  --model_file model.bin --config_file config.yaml --image_dir \
  ./example_images --output_folder ./output_features
"""

import argparse
import copy
import glob
import logging
import math
import os

import numpy as np
import torch

from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess
from utils import Config


class FeatureExtractor:

    MODEL_URL = {"LXMERT": "unc-nlp/frcnn-vg-finetuned"}

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.frcnn, self.frcnn_cfg = self._build_detection_model()

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model_name",
            default="LXMERT",
            type=str,
            help="Model to use for detection",
        )
        parser.add_argument(
            "--model_file",
            default=None,
            type=str,
            help="Huggingface model file. This overrides the model_name param.",
        )
        parser.add_argument(
            "--config_file", default=None, type=str, help="Huggingface config file"
        )
        parser.add_argument(
            "--start_index", default=0, type=int, help="Index to start from "
        )
        parser.add_argument("--end_index", default=None, type=int, help="")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument(
            "--num_features",
            type=int,
            default=100,
            help="Number of features to extract.",
        )
        parser.add_argument(
            "--output_folder", type=str, default="./output", help="Output folder"
        )
        parser.add_argument("--image_dir", type=str, help="Image directory or file")
        # TODO add functionality for this flag
        parser.add_argument(
            "--feature_name",
            type=str,
            help="The name of the feature to extract",
            default="fc6",
        )
        parser.add_argument(
            "--exclude_list",
            type=str,
            help="List of images to be excluded from feature conversion. "
            + "Each image on a new line",
            default="./list",
        )
        parser.add_argument(
            "--confidence_threshold",
            type=float,
            default=0,
            help="Threshold of detection confidence above which boxes will be selected",
        )
        # TODO finish background flag
        parser.add_argument(
            "--background",
            action="store_true",
            help="The model will output predictions for the background class when set",
        )
        parser.add_argument(
            "--padding",
            type=str,
            default=None,
            help="You can set your padding, i.e. 'max_detections'",
        )
        parser.add_argument(
            "--visualize",
            type=bool,
            default=False,
            help="Add this flag to save the extra file used for visualization",
        )
        parser.add_argument(
            "--partition",
            type=int,
            default=None,
            help="Add this flag to save the extra file used for visualization",
        )
        parser.add_argument(
            "--max_partition",
            type=int,
            default=None,
            help="Add this flag to save the extra file used for visualization",
        )
        return parser

    def _build_detection_model(self):
        if self.args.config_file:
            frcnn_cfg = Config.from_pretrained(self.args.config_file)
        else:
            frcnn_cfg = Config.from_pretrained(
                self.MODEL_URL.get(self.args.model_name, self.args.model_name)
            )
        if self.args.model_file:
            frcnn = GeneralizedRCNN.from_pretrained(
                self.args.model_file, config=frcnn_cfg
            )
        else:
            frcnn = GeneralizedRCNN.from_pretrained(
                self.MODEL_URL.get(self.args.model_name, self.args.model_name),
                config=frcnn_cfg,
            )

        return frcnn, frcnn_cfg

    def get_frcnn_features(self, image_paths):
        image_preprocess = Preprocess(self.frcnn_cfg)

        images, sizes, scales_yx = image_preprocess(image_paths)

        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding=None,
            max_detections=self.frcnn_cfg.max_detections,
            return_tensors="pt",
        )

        return output_dict

    def _save_feature(self, file_name, full_features, feat_list, info_list):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        full_feature_base_name = file_base_name + "_full.npy"
        feat_list_base_name = file_base_name + ".npy"
        info_list_base_name = file_base_name + "_info.npy"
        if self.args.visualize:
            np.save(
                os.path.join(self.args.output_folder, full_feature_base_name),
                full_features,
            )
        np.save(
            os.path.join(self.args.output_folder, feat_list_base_name),
            feat_list.cpu().numpy(),
        )
        np.save(os.path.join(self.args.output_folder, info_list_base_name), info_list)

    def _process_features(self, features, index):
        feature_keys = [
            "obj_ids",
            "obj_probs",
            "attr_ids",
            "attr_probs",
            "boxes",
            "sizes",
            "preds_per_image",
            "roi_features",
            "normalized_boxes",
        ]
        single_features = dict()

        for key in feature_keys:
            single_features[key] = features[key][index]

        confidence = self.args.confidence_threshold
        idx = 0
        while idx < single_features["obj_ids"].size()[0]:
            removed = False
            if (
                single_features["obj_probs"][idx] < confidence
                or single_features["attr_probs"][idx] < confidence
            ):
                single_features["obj_ids"] = torch.cat(
                    [
                        single_features["obj_ids"][0:idx],
                        single_features["obj_ids"][idx + 1 :],
                    ]
                )
                single_features["obj_probs"] = torch.cat(
                    [
                        single_features["obj_probs"][0:idx],
                        single_features["obj_probs"][idx + 1 :],
                    ]
                )
                single_features["attr_ids"] = torch.cat(
                    [
                        single_features["attr_ids"][0:idx],
                        single_features["attr_ids"][idx + 1 :],
                    ]
                )
                single_features["attr_probs"] = torch.cat(
                    [
                        single_features["attr_probs"][0:idx],
                        single_features["attr_probs"][idx + 1 :],
                    ]
                )
                single_features["boxes"] = torch.cat(
                    [
                        single_features["boxes"][0:idx, :],
                        single_features["boxes"][idx + 1 :, :],
                    ]
                )
                single_features["preds_per_image"] = (
                    single_features["preds_per_image"] - 1
                )
                single_features["roi_features"] = torch.cat(
                    [
                        single_features["roi_features"][0:idx, :],
                        single_features["roi_features"][idx + 1 :, :],
                    ]
                )
                single_features["normalized_boxes"] = torch.cat(
                    [
                        single_features["normalized_boxes"][0:idx, :],
                        single_features["normalized_boxes"][idx + 1 :, :],
                    ]
                )
                removed = True
            if not removed:
                idx += 1

        feat_list = single_features["roi_features"]

        boxes = single_features["boxes"][: self.args.num_features].cpu().numpy()
        num_boxes = self.args.num_features
        objects = single_features["obj_ids"][: self.args.num_features].cpu().numpy()
        probs = single_features["obj_probs"][: self.args.num_features].cpu().numpy()
        width = single_features["sizes"][1].item()
        height = single_features["sizes"][0].item()
        info_list = {
            "bbox": boxes,
            "num_boxes": num_boxes,
            "objects": objects,
            "cls_prob": probs,
            "image_width": width,
            "image_height": height,
        }

        return single_features, feat_list, info_list

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size], i

    def extract_features(self):
        image_dir = self.args.image_dir

        if os.path.isfile(image_dir):
            features = self.get_frcnn_features([image_dir])
            self._save_feature(image_dir, features[0])
        else:
            files = glob.glob(os.path.join(image_dir, "*.png"))
            files.extend(glob.glob(os.path.join(image_dir, "*.jpg")))
            files.extend(glob.glob(os.path.join(image_dir, "*.jpeg")))

            files = set(files)
            exclude = {}

            if os.path.exists(self.args.exclude_list):
                with open(self.args.exclude_list) as f:
                    lines = f.readlines()
                    for line in lines:
                        exclude[
                            line.strip("\n").split(os.path.sep)[-1].split(".")[0]
                        ] = 1

            for f in list(files):
                file_name = f.split(os.path.sep)[-1].split(".")[0]
                if file_name in exclude:
                    files.pop(f)

            files = list(files)
            files = sorted(files)

            if self.args.partition is not None and self.args.max_partition is not None:
                interval = math.floor(len(files) / self.args.max_partition)
                if self.args.partition == self.args.max_partition:
                    files = files[self.args.partition * interval :]
                else:
                    files = files[
                        self.args.partition
                        * interval : (self.args.partition + 1)
                        * interval
                    ]

            file_names = copy.deepcopy(files)

            start_index = self.args.start_index
            end_index = self.args.end_index
            if end_index is None:
                end_index = len(files)

            finished = 0
            total = len(files[start_index:end_index])
            failed = 0
            failedNames = []

            for chunk, begin_idx in self._chunks(
                files[start_index:end_index], self.args.batch_size
            ):
                try:
                    features = self.get_frcnn_features(chunk)
                    for idx, file_name in enumerate(chunk):
                        full_features, feat_list, info_list = self._process_features(
                            features, idx
                        )
                        self._save_feature(
                            file_names[begin_idx + idx],
                            full_features,
                            feat_list,
                            info_list,
                        )
                    finished += len(chunk)

                    if finished % 200 == 0:
                        print(f"Processed {finished}/{total}")
                except Exception:
                    failed += len(chunk)
                    for idx, file_name in enumerate(chunk):
                        failedNames.append(file_names[begin_idx + idx])
                    logging.exception("message")
            if self.args.partition is not None:
                print("Partition " + str(self.args.partition) + " done.")
            print("Failed: " + str(failed))
            print("Failed Names: " + str(failedNames))


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_features()
