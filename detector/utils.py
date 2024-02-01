import copy
import time

import cv2
import networkx as nx
import numpy as np
import pgmpy
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from pgmpy.base import DAG
from pgmpy.models import BayesianNetwork

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def print_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000.0
        print(f"{func.__name__} took {execution_time_ms:.0f} ms to execute")
        return result

    return wrapper


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def merge_image_with_overlay(image, boxes, scores, class_ids, mask_alpha=0.4):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]

        draw_box(det_img, box, color)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
             thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image: np.ndarray, text: str, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
              font_size: float = 0.001, text_thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1),
                  (x1 + tw, y1 - th), color, -1)

    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness,
                       cv2.LINE_AA)


def draw_masks(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


def merge_lists_of_dicts(list1, list2):
    merged_list = []
    for dict1, dict2 in zip(list1, list2):
        merged_dict = {**dict1, **dict2}
        merged_list.append(merged_dict)
    return merged_list


def merge_single_dicts(dict1, dict2):
    return {**dict1, **dict2}


def export_BN_to_graph(bn: BayesianNetwork or pgmpy.base.DAG, root=None, try_visualization=False, vis_ls=None,
                       save=False,
                       name=None, show=True, color_map=None):
    if vis_ls is None:
        vis_ls = ["fdp"]
    else:
        vis_ls = vis_ls

    if name is None:
        name = root

    if try_visualization:
        vis_ls = ['neato', 'dot', 'twopi', 'fdp', 'sfdp', 'circo']

    for s in vis_ls:
        pos = graphviz_layout(bn, root=root, prog=s)
        nx.draw(
            bn, pos, with_labels=True, arrowsize=20, node_size=1500,  # alpha=1.0, font_weight="bold",
            node_color=color_map
        )
        if save:
            plt.box(False)
            plt.savefig(f"{name}.png", dpi=400, bbox_inches="tight")  # default dpi is 100
        if show:
            plt.show()


def get_mbs_as_bn(model: DAG or BayesianNetwork, center: [str]):
    mb_list = []
    for node in center:
        mb_list.extend(model.get_markov_blanket(node))
    mb = copy.deepcopy(model)

    mb_list.extend(center)
    for n in model.nodes:
        if n not in mb_list:
            mb.remove_node(n)

    return mb


def sort_and_join(s1, s2):
    sorted_strings = sorted([s1, s2], reverse=False)
    return '-'.join(sorted_strings)


def get_true(param):
    if len(param.variables) > 2:
        raise Exception("How come?")
    if len(param.variables) == 2:
        if param.values.shape == (1, 1):
            if (param.__getattribute__("state_names")[param.variables[0]][0] == 'True' and
                    param.__getattribute__("state_names")[param.variables[1]][0] == 'True'):
                return 1
            else:
                return 0
        elif param.values.shape == (2, 1):
            if (param.__getattribute__("state_names")[param.variables[0]][0] == 'True' or
                    param.__getattribute__("state_names")[param.variables[1]][0] == 'True'):
                return param.values[1][0]
            else:
                return 0
        elif param.values.shape == (1, 2):
            if (param.__getattribute__("state_names")[param.variables[0]][0] == 'True' or
                    param.__getattribute__("state_names")[param.variables[1]][0] == 'True'):
                return param.values[0][1]
            else:
                return 0
        elif param.values.shape == (2, 2):
            return param.values[1][1]
        else:
            return param.values[1]
    elif len(param.variables) == 1:
        if param.values.shape == (2, 1):
            return param.values[1]
        elif param.__getattribute__("state_names")[param.variables[0]][0] == True:
            return param.values[0]
        else:
            return param.values[1]


def get_latency_for_devices(d1, d2):
    translate_dict = {'Xavier': 0, 'Orin': 1, 'PC': 2}
    distance = np.array([[1, 5, 25],
                         [5, 5, 15],
                         [25, 15, 10]])

    a = translate_dict[d1]
    b = translate_dict[d2]

    return distance[a, b]