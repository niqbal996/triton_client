
import numpy as np
import struct
import os
import math
import time
import torch
import torchvision

dir_path = os.path.dirname(os.path.realpath(__file__))

def deserialize_bytes_float(encoded_tensor):
    strs = list()
    offset = 0
    val_buf = encoded_tensor
    datatype = 'f'
    l = struct.calcsize(datatype)
    while offset < len(val_buf):
        sb = struct.unpack_from(datatype, val_buf, offset)[0]
        offset += l
        strs.append(sb)
    return (np.array(strs, dtype=np.object_))

def deserialize_bytes_int(encoded_tensor):
    strs = list()
    offset = 0
    val_buf = encoded_tensor
    datatype = 'i'
    l = struct.calcsize(datatype)
    while offset < len(val_buf):
        sb = struct.unpack_from(datatype, val_buf, offset)[0]
        offset += l
        strs.append(sb)
    return (np.array(strs, dtype=np.object_))

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)

def extract_boxes_yolov5(prediction, conf_thres=0.6, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    boxes = deserialize_bytes_float(prediction.raw_output_contents[0])
    boxes = np.reshape(boxes, prediction.outputs[0].shape)
    prediction = boxes

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    assert np.any(np.array(xc)) == True
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    t = time.time()
    output = [torch.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4]).astype(np.float32)

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            x = torch.tensor(x.astype(np.float64))
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((torch.from_numpy(box), conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def extract_boxes_triton(response):
    """
    Post-process response to show classifications.
    """
    conf_thresh = 0.4
    nms_thresh = 0.6

    confs = deserialize_bytes_tensor(response.raw_output_contents[0])
    confs_contents = np.reshape(confs, response.outputs[0].shape)
    boxes = deserialize_bytes_tensor(response.raw_output_contents[1])
    boxes_contents = np.reshape(boxes, response.outputs[1].shape)

    box_array = boxes_contents
    # [batch, num, num_classes]
    confs = confs_contents

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append(
                        [ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3],
                         ll_max_conf[k],
                         ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)

    return bboxes_batch


def extract_boxes_onnx(conf_thresh, nms_thresh, output):
    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]


    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append(
                        [ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k],
                         ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)

    return bboxes_batch

def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img

def extract_boxes_detectron(prediction, conf_thres=0.6, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    boxes = deserialize_bytes_float(prediction.raw_output_contents[0])
    boxes = np.reshape(boxes, prediction.outputs[0].shape)
    # TODO class_ids are all zeros? BUG in unpacking or inferencing?
    # class_ids = deserialize_bytes_int(prediction.raw_output_contents[1])
    # class_ids = np.reshape(class_ids, prediction.outputs[1].shape * 2) # TODO DEBUG: Remove after debug *2
    scores = deserialize_bytes_float(prediction.raw_output_contents[2])
    scores = np.reshape(scores, prediction.outputs[2].shape)
    class_ids = [0] * len(scores)
    # prediction = boxes

    # nc = prediction.shape[2] - 5  # number of classes
    # xc = prediction[..., 4] > conf_thres  # candidates
    # assert np.any(np.array(xc)) == True
    # # Checks
    # assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    # assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    #
    # # Settings
    # min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    # max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    # time_limit = 10.0  # seconds to quit after
    # redundant = True  # require redundant detections
    # multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # merge = False  # use merge-NMS
    # t = time.time()
    # output = [torch.zeros((0, 6))] * prediction.shape[0]
    # for xi, x in enumerate(prediction):  # image index, image inference
    #     # Apply constraints
    #     # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
    #     x = x[xc[xi]]  # confidence
    #
    #     # Cat apriori labels if autolabelling
    #     if labels and len(labels[xi]):
    #         l = labels[xi]
    #         v = torch.zeros((len(l), nc + 5), device=x.device)
    #         v[:, :4] = l[:, 1:5]  # box
    #         v[:, 4] = 1.0  # conf
    #         v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
    #         x = torch.cat((x, v), 0)
    #
    #     # If none remain process next image
    #     if not x.shape[0]:
    #         continue
    #
    #     # Compute conf
    #     x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
    #
    #     # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    #     box = xywh2xyxy(x[:, :4]).astype(np.float32)
    #
    #     # Detections matrix nx6 (xyxy, conf, cls)
    #     if multi_label:
    #         i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
    #         x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
    #     else:  # best class only
    #         x = torch.tensor(x.astype(np.float64))
    #         conf, j = x[:, 5:].max(1, keepdim=True)
    #         x = torch.cat((torch.from_numpy(box), conf, j.float()), 1)[conf.view(-1) > conf_thres]
    #
    #     # Filter by class
    #     if classes is not None:
    #         x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
    #
    #     # Apply finite constraint
    #     # if not torch.isfinite(x).all():
    #     #     x = x[torch.isfinite(x).all(1)]
    #
    #     # Check shape
    #     n = x.shape[0]  # number of boxes
    #     if not n:  # no boxes
    #         continue
    #     elif n > max_nms:  # excess boxes
    #         x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
    #
    #     # Batched NMS
    #     c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
    #     boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
    #     i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    #     if i.shape[0] > max_det:  # limit detections
    #         i = i[:max_det]
    #     if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
    #         # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
    #         iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
    #         weights = iou * scores[None]  # box weights
    #         x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
    #         if redundant:
    #             i = i[iou.sum(1) > 1]  # require redundancy
    #
    #     output[xi] = x[i]
    #     if (time.time() - t) > time_limit:
    #         print(f'WARNING: NMS time limit {time_limit}s exceeded')
    #         break  # time limit exceeded

    return boxes, class_ids, scores

def load_class_names(namesfile='{}/../data/crop.names'.format(dir_path)):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names