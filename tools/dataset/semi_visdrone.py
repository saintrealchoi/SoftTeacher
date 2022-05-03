# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
"""Generate labeled and unlabeled dataset for coco train.

Example:
python tools/coco_semi.py
"""

import argparse
import numpy as np
import json
import os


def prepare_visdrone_data(seed=1, percent=10.0, version=2021, seed_offset=0):
    """Prepare VISDRONE dataset for Semi-supervised learning
    Args:
      seed: random seed for dataset split
      percent: percentage of labeled dataset
      version: VISDRONE dataset version
    """

    def _save_anno(name, images, annotations):
        """Save annotation."""
        print(
            ">> Processing dataset {}.json saved ({} images {} annotations)".format(
                name, len(images), len(annotations)
            )
        )
        new_anno = {}
        new_anno["images"] = images
        new_anno["annotations"] = annotations
        # visdrone annotation doesn't have licenses, info key
        # new_anno["licenses"] = anno["licenses"]
        # new_anno["info"] = anno["info"]
        new_anno["categories"] = anno["categories"]
        path = "{}/{}".format(VDANNDIR, "semi_supervised")
        if not os.path.exists(path):
            os.mkdir(path)

        with open(
            "{root}/{folder}/{save_name}.json".format(
                save_name=name, root=VDANNDIR, folder="semi_supervised"
            ),
            "w",
        ) as f:
            json.dump(new_anno, f)
        print(
            ">> Data {}.json saved ({} images {} annotations)".format(
                name, len(images), len(annotations)
            )
        )

    np.random.seed(seed + seed_offset)
    VDANNDIR = os.path.join(DATA_DIR, "annotations")

    anno = json.load(
        open(os.path.join(VDANNDIR, "train.json"))
    )

    image_list = anno["images"]
    labeled_tot = int(percent / 100.0 * len(image_list))
    labeled_ind = np.random.choice(
        range(len(image_list)), size=labeled_tot, replace=False
    )
    labeled_id = []
    labeled_images = []
    unlabeled_images = []
    labeled_ind = set(labeled_ind)
    for i in range(len(image_list)):
        if i in labeled_ind:
            labeled_images.append(image_list[i])
            labeled_id.append(image_list[i]["id"])
        else:
            unlabeled_images.append(image_list[i])

    # get all annotations of labeled images
    labeled_id = set(labeled_id)
    labeled_annotations = []
    unlabeled_annotations = []
    for an in anno["annotations"]:
        if an["image_id"] in labeled_id:
            labeled_annotations.append(an)
        else:
            unlabeled_annotations.append(an)

    # save labeled and unlabeled
    save_name = "instances_train{version}.{seed}@{tot}".format(
        version=version, seed=seed, tot=int(percent)
    )
    _save_anno(save_name, labeled_images, labeled_annotations)
    save_name = "instances_train{version}.{seed}@{tot}-unlabeled".format(
        version=version, seed=seed, tot=int(percent)
    )
    _save_anno(save_name, unlabeled_images, unlabeled_annotations)
    #construct 120k unlabeled data
    unlabeled_ann_file = os.path.join(VDANNDIR, "instances_unlabeled{}.json".format(version))
    if not os.path.exists(unlabeled_ann_file):
        unlabeled_info = json.load(
            open(os.path.join(VDANNDIR, "image_info_unlabeled{}.json".format(version)))
        )
        unlabeled_info["annotations"] = []
        unlabeled_info["categories"] = anno["categories"]
        print(">> Data {}.json saved({} images {} annotations)".format(unlabeled_ann_file,len(unlabeled_info["images"]),len(unlabeled_info["annotations"])))
        json.dump(unlabeled_info,open(os.path.join(VDANNDIR, "instances_unlabeled{}.json".format(version)),'w'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--percent", type=float, default=10)
    parser.add_argument("--version", type=int, default=2021)
    parser.add_argument("--seed", type=int, help="seed", default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    print(args)
    DATA_DIR = args.data_dir
    prepare_visdrone_data(args.seed, args.percent, args.version, args.seed_offset)
