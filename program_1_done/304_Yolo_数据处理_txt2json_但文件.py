import json
import cv2
import os
import numpy as np


def convert_txt_to_json_batch(txt_folder, img_folder, json_folder):
    categories = [{
        "id": 0,
        "name": "person",
        "supercategory": "",
        "color": "#75e955",
        "metadata": {},
        "keypoint_colors": [],
        "keypoints": [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ],
        "skeleton": [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
        ]
    }]

    txt_files = os.listdir(txt_folder)

    for txt_file in txt_files:
        print(txt_file[0:-4])
        # print(type(txt_file[0:-4]))
        txt_path = os.path.join(txt_folder, txt_file)
        image_name = os.path.splitext(txt_file)[0] + ".jpg"
        image_path = os.path.join(img_folder, image_name)
        json_name = os.path.splitext(txt_file)[0] + ".json"
        json_path = os.path.join(json_folder, json_name)

        images = []
        annotations = []

        img = cv2.imread(image_path)
        image_height, image_width = img.shape[:2]


    # 写入数据
        # 基本信息
        image_id = 0
        annotation_id = 0

        image_info = {
            "file_name": str(image_name),
            "height": image_height,
            "width": image_width,
            "id": image_id,
            "path": "/datasets/wider/" + str(image_name),
            "dataset_id": "3"
        }
        images.append(image_info)
        with open(txt_path, 'r') as txt_file:
            lines = txt_file.readlines()

            # 遍历每一行
            for line in lines:
                data = line.strip().split()
                class_index = int(data[0])
                x, y, width, height = map(float, data[1:5])
                keypoints_info = list(map(float, data[5:]))

                bbox = [x * image_width - width * image_width / 2, y * image_height - height * image_height / 2,
                        width * image_width, height * image_height]
                segmentation = [[int(coord) for coord in seg] for seg in [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1],
                                                                           bbox[0] + bbox[2], bbox[1] + bbox[3],
                                                                           bbox[0],
                                                                           bbox[1] + bbox[3]]]]
                keypoints = []
                for i in range(0, len(keypoints_info), 3):
                    px, py, _ = keypoints_info[i:i + 3]
                    keypoints.extend([px * image_width, py * image_height, 2])

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_index,
                    "iscrowd": False,
                    "isbbox": True,
                    "color": "#588C31",
                    "area": bbox[2] * bbox[3],
                    "bbox": [round(x, 1) for x in bbox],
                    "segmentation": [list(map(round, seg)) for seg in segmentation],
                    "keypoints": keypoints,
                    "metadata": {},
                    "num_keypoints": len(keypoints) // 3
                }
                annotations.append(annotation)

                image_id += 1
                annotation_id += 1

        output_data = {
            "categories": categories,
            "images": images,
            "annotations": annotations
        }

        with open(json_path, 'w') as json_file:
            json.dump(output_data, json_file, indent=2)


# 使用示例
if __name__ == '__main__':
    txt_folder = "../txt"
    img_folder = "../img"
    json_folder = "../json"

    convert_txt_to_json_batch(txt_folder, img_folder, json_folder)
