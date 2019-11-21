import torch
from torch.utils.data import  Dataset
import pandas as pd
import numpy as np
import json
import cv2


def gaussian2D(shape, sigma=(1, 1), scale=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h =  scale * np.exp(-(x * x / sigma[1] **2 + y * y / sigma[0] ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(image, rect, scale=1):
    image[rect[0]:rect[2], rect[1]:rect[3]] = np.maximum(image[rect[0]:rect[2], rect[1]:rect[3]],
                                                         gaussian2D((rect[2] - rect[0], rect[3] - rect[1]),
                                                                    ((rect[2] - rect[0]) / 3, (rect[3] - rect[1]) / 3),
                                                                    scale))
    return image


class TolokaDataset(Dataset):
    num_classes = 6
    label_dict = {"Boat": 0, "Bouy": 1, "Vessel": 2, "Millitary": 3, "Ice": 4, "Other": 5}

    def __init__(self, path, path_mapping, augment=True, rotate=0, down_ratio=4, output_dim=512):
        self.data = pd.read_csv(path, sep="\t")
        self.path_mappping = path_mapping
        self.down_ratio = down_ratio
        self.output_dim = output_dim
        self.augment = augment
        self.rotate = rotate
        self.pad = 31
        cv2.startWindowThread()
        #cv2.namedWindow("123")

    def summary(self):
        stats = np.zeros((self.num_classes,))
        for index, row in self.data.iterrows():
            try:
                anno = json.loads(row[3])
                for rect in anno:
                    try:
                        stats[self.label_dict.get(rect["annotation"], 5)] += 1
                    except KeyError:
                        pass
            except TypeError:
                pass
        return stats

    def relative_bbox2absolute(self, rect, image_shape):
        box = (np.clip(rect["p1"]["y"], 0, 1) * image_shape[0],
                np.clip(rect["p1"]["x"], 0, 1) * image_shape[1],
                np.clip(rect["p2"]["y"], 0, 1) * image_shape[0],
                np.clip(rect["p2"]["x"], 0, 1) * image_shape[1])
        if box[0] > box [2]:
            box = box[2], box[1], box[0], box[3]
        if box[1] > box[3]:
            box = box[0], box[3], box[2], box[1]
        return box

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = cv2.imread(self.data.iloc[index, 0].replace(self.path_mappping[0], self.path_mappping[1]))
        try:
            annotation = json.loads(self.data.iloc[index, 3])
        except TypeError:
            annotation = []
        num_objs = len(annotation)

        # for anno in annotation:
        #     box = self.relative_bbox2absolute(anno["data"], img.shape)
        #     cv2.rectangle(img, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (255, 0, 0), 10)
        # cv2.imshow("123", img)
        # cv2.waitKey()

        if self.augment:
            image_w = self.output_dim
            image_h = self.output_dim
            w_border = self._get_border(128, img.shape[1])
            h_border = self._get_border(128, img.shape[0])
            c = np.zeros((2,))

            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            angle = self.rotate * np.random.uniform(-1, 1)
            rotate_mat = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
            rect = cv2.transform(
                np.array([[[0, 0]], [[img.shape[1], 0]], [[0, img.shape[0]]], [[img.shape[1], img.shape[0]]]],
                         dtype=np.float32), rotate_mat)[:, 0, :]
            shift = np.min(rect, axis=0)
            rotate_mat[:, 2] -= shift

            choice = np.random.randint(0, num_objs + 1)
            border = int(self.output_dim/(2*np.cos(np.deg2rad(angle))) + 0.5)

            if choice < num_objs:
                ann = annotation[choice]
                bbox = self.relative_bbox2absolute(ann['data'], img.shape)
                sigma = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                c = ((bbox[2] + bbox[0]) / 2, (bbox[3] - bbox[1]) / 2)
                center_point = np.random.normal(c, sigma, (2,))
                center_point[0] = np.clip(center_point[0], border , img.shape[0] - border - 1)
                center_point[1] = np.clip(center_point[1], border, img.shape[1] - border - 1)
                center_point = cv2.transform(center_point, rotate_mat.reshape(1, 2, 3)).astype(np.int32)
                center_point = center_point[:, 0]
                s_low = max(abs(c[0] - center_point[0]), abs(c[1] - center_point[1]), self.output_dim // 2)
                s_high = min(img.shape[0] - center_point[0] - 1,
                             img.shape[1] - center_point[1] - 1, center_point[0], center_point[1])
                size = s_high if s_low >= s_high else np.random.randint(s_low,  int(s_high))
            else:
                c = (img.shape[0] / 2, img.shape[1] / 2)
                sigma = (max((img.shape[0] - 2 * border), 0) / 6, max(0, (img.shape[1] - 2 * border)) / 6)
                center_point = np.random.normal(c, sigma, (2,))
                center_point[0] = np.clip(center_point[0], border, img.shape[0] - border - 1)
                center_point[1] = np.clip(center_point[1], border, img.shape[1] - border - 1)
                center_point = cv2.transform(center_point, rotate_mat.reshape(1, 2, 3)).astype(np.int32)
                center_point = center_point[:, 0]
                s_low = self.output_dim // 2
                s_high = min(img.shape[0] - center_point[0] - 1,
                             img.shape[1] - center_point[1] - 1, center_point[0], center_point[1])
                size = s_high if s_low >= s_high else np.random.randint(s_low,  int(s_high))

            rect = cv2.transform(
                np.array([[[0, 0]], [[img.shape[1], 0]], [[0, img.shape[0]]], [[img.shape[1], img.shape[0]]]],
                         dtype=np.float32), rotate_mat)[:, 0, :]
            image = cv2.warpAffine(img, rotate_mat, (np.max(rect[:, 0]), np.max(rect[:, 1])), flags=cv2.INTER_LINEAR)
            sample = img[center_point[0] - size:center_point[0] + size,
                     center_point[1] - size: center_point[1] + size, :].copy()
            image = cv2.resize(sample, (self.output_dim, self.output_dim))#.transpose(2, 0, 1)
            trans_output = cv2.getAffineTransform(np.array([[center_point[0] - size, center_point[1] - size],
                                                            [center_point[0] + size, center_point[1] - size],
                                                            [center_point[0] - size, center_point[1] + size]],
                                                           dtype=np.float32),
                                                  np.array([[0, 0], [self.output_dim, 0], [0, self.output_dim]],
                                                           dtype=np.float32))
        else:
            trans_output = cv2.getRotationMatrix2D((0,0), 0, 1)
            image_w = (img.shape[0] | self.pad) + 1
            image_h = img.shape[1]
            image = img


        # for anno in annotation:
        #     bbox = self.relative_bbox2absolute(anno["data"], img.shape)
        #     center_point_g = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        #     center_point_g = cv2.transform(center_point_g.reshape(1,1,2), trans_output)
        #     c = np.sqrt(trans_output[0,0] ** 2 + trans_output[0,1] ** 2 )
        #     w, h = (bbox[3] - bbox[1]) / 2 * c, (bbox[2] - bbox[0]) / 2 * c
        #     bbox = [center_point_g[0,0,0] - h, center_point_g[0,0,1] -w, center_point_g[0,0,0] + h,
        #             center_point_g[0,0,1] + w]
        #     #bbox = [np.min(box[:, 0, 0]), np.min(box[:, 0, 1]), np.max(box[:, 0, 0]), np.max(box[:, 0, 1])]
        #     cv2.rectangle(image, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), (255, 0, 0), 10)
        #     cv2.circle(image, (center_point_g[0,0,1], center_point_g[0,0,0]), 40, (0, 255, 0), 10)
        # cv2.imshow("123", image)
        # cv2.waitKey()

        center_map = np.zeros((self.num_classes, image_w // self.down_ratio, image_h // self.down_ratio))
        wh_map = np.zeros((2, image_w // self.down_ratio, image_h // self.down_ratio))
        for anno in annotation:
            bbox = self.relative_bbox2absolute(anno["data"], img.shape)
            center_point_g = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            center_point_g = cv2.transform(center_point_g.reshape(1, 1, 2), trans_output)
            c = np.sqrt(trans_output[0, 0] ** 2 + trans_output[0, 1] ** 2)
            w, h = (bbox[3] - bbox[1]) / 2 * c, (bbox[2] - bbox[0]) / 2 * c
            bbox = [center_point_g[0, 0, 0] - h, center_point_g[0, 0, 1] - w, center_point_g[0, 0, 0] + h,
                    center_point_g[0, 0, 1] + w]
            bbox[0::2] = np.clip(bbox[0::2], 0, image_w)
            bbox[1::2] = np.clip(bbox[1::2], 0, image_h)
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= 400 and (bbox[2] - bbox[0]) > 4 and (bbox[3] - bbox[1]) > 4:
                bbox = [int(b / self.down_ratio) for b in bbox]
                class_index = self.label_dict.get(anno["annotation"], 5)
                center_map[class_index, :, :] = draw_gaussian(center_map[class_index, :, :], bbox)
                wh_map[0, :, :] = draw_gaussian(wh_map[0, :, :], bbox, w)
                wh_map[1, :, :] = draw_gaussian(wh_map[1, :, :], bbox, h)
                #center_point = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

        # map = wh_map.max(0)
        # map = cv2.resize((map).astype(np.uint8), (image_h, image_w))
        # cv2.imshow("321", map)
        # cv2.waitKey()
        return {"input": (image.transpose(2,0,1) - 128).astype(np.float32) / 255, "center": center_map,
                "wh": wh_map.astype(np.float32)}





