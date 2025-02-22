import numpy as np
import onnxruntime
import onnx
import numpy
import cv2
import torch
import torchvision
import time

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # # https://github.com/ultralytics/yolov5/blob/master/utils/general.py
    gain_x = img0_shape[0] / img1_shape[0]
    gain_y = img0_shape[1] / img1_shape[1]

    boxes[:, [0, 2]] *= gain_y
    boxes[:, [1, 3]] *= gain_x
    return boxes


def xywh2xyxy(x):
    # https://github.com/ultralytics/yolov5/blob/master/utils/general.py
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2, eps=1e-7):
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

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

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
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
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
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

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
        # if (time.time() - t) > time_limit:
        #     print(f'WARNING: NMS time limit {time_limit}s exceeded')
        #     break  # time limit exceeded

    return output

model_path = './weights/yolov5m_syn.onnx'
exec_providers = onnxruntime.get_available_providers()
exec_provider = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in exec_providers else ['CPUExecutionProvider']


session = onnxruntime.InferenceSession(model_path, sess_options=None, providers=exec_provider)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

if True:
    vidcap = cv2.VideoCapture('/home/niqbal/Downloads/0001-1440.mkv')
    success, image = vidcap.read()
    count = 0
    while success:
        success, img = vidcap.read()
        orig_size = img.shape
        # print('Read a new frame: ', success)
        count += 1
        orig = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640), cv2.INTER_LINEAR)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)

        pred = session.run([output_name], {input_name: img})[0]
        pred = torch.Tensor(pred).to('cpu')
        pred = non_max_suppression(pred)[0]
        # pred = scale_boxes((640, 640), pred, (1535, 2047))
        pred = scale_boxes((640, 640), pred, (orig_size[0], orig_size[1]))
        pred = pred.cpu().numpy()

        for obj in range(pred.shape[0]):
            box = pred[obj, :]
            if box[-1] == 0:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            cv2.rectangle(orig,
                          pt1=(int(box[0]), int(box[1])),
                          pt2=(int(box[2]), int(box[3])),
                          color=color,
                          thickness=2)
            # cv2.putText(orig,
            #             '{:.2f} {}'.format(object[-2], self.class_names[int(object[-1])]),
            #             org=(int(box[0]), int(box[1] - 10)),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=0.5,
            #             thickness=2,
            #             color=self.color)
        cv2.imwrite("./output_data/{:04}.png".format(count), orig)
        # cv2.imshow('figure', orig)
        # cv2.waitKey()
        print(count)

else:
    img = cv2.imread('/home/niqbal/git/yolov5_v6.0/crop_test.png')

