from .base_postprocess import Postprocess
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Tuple
try:
    from mmcv.ops import nms, nms_rotated
except ImportError as e:
    print("[ERROR] {}".format(e))
class PointPillarPostprocess(Postprocess):
    def __init__(self):
        self.use_sigmoid_cls = True
        self.use_rotate_nms = True
        self.feat_map_size = torch.Size([248, 216])
        self.rotations = [0, 1.5707963]
        self.box_code_size = 7          # TODO change 7  to dynamic box_code_size
        self.anchors = self.single_level_grid_anchors(featmap_size=self.feat_map_size,
                                                      scale=1)        

    def postprocess(self):
        pass

    def load_class_names(self, namesfile='./data/crop.names'):
        class_names = []
        with open(namesfile, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        return class_names

    def extract_boxes(self, prediction):
        """Runs Non-Maximum Suppression (NMS) on inference results

            Returns:
                 list of detections, on (n,6) tensor per image [xyxy, conf, cls]
            """
        self.boxes = self.deserialize_bytes_float(prediction.raw_output_contents[0])
        self.boxes = np.reshape(self.boxes, prediction.outputs[0].shape)
        self.class_ids = self.deserialize_bytes_float(prediction.raw_output_contents[1])
        self.class_ids = np.reshape(self.class_ids, prediction.outputs[1].shape)
        self.scores = self.deserialize_bytes_float(prediction.raw_output_contents[2])
        self.scores = np.reshape(self.scores, prediction.outputs[2].shape)

        self.boxes, self.class_ids, self.scores = self.nms_3d()

        return self.boxes, self.class_ids, self.scores


    def box3d_multiclass_nms(self, mlvl_bboxes,
                         mlvl_bboxes_for_nms,
                         mlvl_scores,
                         score_thr,
                         max_num,
                         cfg,
                         mlvl_dir_scores=None,
                         mlvl_attr_scores=None,
                         mlvl_bboxes2d=None):
        """Multi-class NMS for 3D boxes. The IoU used for NMS is defined as the 2D
        IoU between BEV boxes.

        Args:
            mlvl_bboxes (torch.Tensor): Multi-level boxes with shape (N, M).
                M is the dimensions of boxes.
            mlvl_bboxes_for_nms (torch.Tensor): Multi-level boxes with shape
                (N, 5) ([x1, y1, x2, y2, ry]). N is the number of boxes.
                The coordinate system of the BEV boxes is counterclockwise.
            mlvl_scores (torch.Tensor): Multi-level boxes with shape
                (N, C + 1). N is the number of boxes. C is the number of classes.
            score_thr (float): Score threshold to filter boxes with low
                confidence.
            max_num (int): Maximum number of boxes will be kept.
            cfg (dict): Configuration dict of NMS.
            mlvl_dir_scores (torch.Tensor, optional): Multi-level scores
                of direction classifier. Defaults to None.
            mlvl_attr_scores (torch.Tensor, optional): Multi-level scores
                of attribute classifier. Defaults to None.
            mlvl_bboxes2d (torch.Tensor, optional): Multi-level 2D bounding
                boxes. Defaults to None.

        Returns:
            tuple[torch.Tensor]: Return results after nms, including 3D
                bounding boxes, scores, labels, direction scores, attribute
                scores (optional) and 2D bounding boxes (optional).
        """
        # do multi class nms
        # the fg class id range: [0, num_classes-1]
        num_classes = mlvl_scores.shape[1] - 1
        bboxes = []
        scores = []
        labels = []
        dir_scores = []
        attr_scores = []
        bboxes2d = []
        for i in range(0, num_classes):
            # get bboxes and scores of this class
            cls_inds = mlvl_scores[:, i] > score_thr
            if not cls_inds.any():
                continue

            _scores = mlvl_scores[cls_inds, i]
            _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]

            if self.use_rotate_nms:
                nms_func = self.nms_bev #TODO HARD_CODED
            else:
                nms_func = self.nms_normal_bev

            selected = nms_func(_bboxes_for_nms, _scores, 0.01)
            _mlvl_bboxes = mlvl_bboxes[cls_inds, :]
            bboxes.append(_mlvl_bboxes[selected])
            scores.append(_scores[selected])
            cls_label = mlvl_bboxes.new_full((len(selected), ),
                                            i,
                                            dtype=torch.long)
            labels.append(cls_label)

            if mlvl_dir_scores is not None:
                _mlvl_dir_scores = mlvl_dir_scores[cls_inds]
                dir_scores.append(_mlvl_dir_scores[selected])
            if mlvl_attr_scores is not None:
                _mlvl_attr_scores = mlvl_attr_scores[cls_inds]
                attr_scores.append(_mlvl_attr_scores[selected])
            if mlvl_bboxes2d is not None:
                _mlvl_bboxes2d = mlvl_bboxes2d[cls_inds]
                bboxes2d.append(_mlvl_bboxes2d[selected])

        if bboxes:
            bboxes = torch.cat(bboxes, dim=0)
            scores = torch.cat(scores, dim=0)
            labels = torch.cat(labels, dim=0)
            if mlvl_dir_scores is not None:
                dir_scores = torch.cat(dir_scores, dim=0)
            if mlvl_attr_scores is not None:
                attr_scores = torch.cat(attr_scores, dim=0)
            if mlvl_bboxes2d is not None:
                bboxes2d = torch.cat(bboxes2d, dim=0)
            if bboxes.shape[0] > max_num:
                _, inds = scores.sort(descending=True)
                inds = inds[:max_num]
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                scores = scores[inds]
                if mlvl_dir_scores is not None:
                    dir_scores = dir_scores[inds]
                if mlvl_attr_scores is not None:
                    attr_scores = attr_scores[inds]
                if mlvl_bboxes2d is not None:
                    bboxes2d = bboxes2d[inds]
        else:
            bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
            scores = mlvl_scores.new_zeros((0, ))
            labels = mlvl_scores.new_zeros((0, ), dtype=torch.long)
            if mlvl_dir_scores is not None:
                dir_scores = mlvl_scores.new_zeros((0, ))
            if mlvl_attr_scores is not None:
                attr_scores = mlvl_scores.new_zeros((0, ))
            if mlvl_bboxes2d is not None:
                bboxes2d = mlvl_scores.new_zeros((0, 4))

        results = (bboxes, scores, labels)

        if mlvl_dir_scores is not None:
            results = results + (dir_scores, )
        if mlvl_attr_scores is not None:
            results = results + (attr_scores, )
        if mlvl_bboxes2d is not None:
            results = results + (bboxes2d, )

        return results

    def single_level_grid_anchors(self, featmap_size, scale, device='cuda'):
        """Generate grid anchors of a single level feature map.

        This function is usually called by method ``self.grid_anchors``.

        Args:
            featmap_size (tuple[int]): Size of the feature map.
            scale (float): Scale factor of the anchors in the current level.
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature map.
        """
        # We reimplement the anchor generator using torch in cuda
        # torch: 0.6975 s for 1000 times
        # numpy: 4.3345 s for 1000 times
        # which is ~5 times faster than the numpy implementation
        # if not self.size_per_range:
        #     return self.anchors_single_range(
        #         featmap_size,
        #         self.ranges[0],
        #         scale,
        #         self.sizes,
        #         self.rotations,
        #         device=device)
        # TODO hard coded values
        self.ranges = [[0, -39.68, -0.6, 69.12, 39.68, -0.6], 
                        [0, -39.68, -0.6, 69.12, 39.68, -0.6], 
                        [0, -39.68, -1.78, 69.12, 39.68, -1.78]]
        self.sizes=[[0.8, 0.6, 1.73],
                    [1.76, 0.6, 1.73],
                    [3.9, 1.6, 1.56]]
        mr_anchors = []
        for anchor_range, anchor_size in zip(self.ranges, self.sizes):
            mr_anchors.append(
                self.anchors_single_range(
                    featmap_size,
                    anchor_range,
                    scale,
                    anchor_size,
                    self.rotations,
                    device=device))
        mr_anchors = torch.cat(mr_anchors, dim=-3)
        return mr_anchors

    def anchors_single_range(self,
                             feature_size,
                             anchor_range,
                             scale=1,
                             sizes=[[3.9, 1.6, 1.56]],
                             rotations=[0, 1.5707963],
                             device='cuda'):
        """Generate anchors in a single range.

        Args:
            feature_size (list[float] | tuple[float]): Feature map size. It is
                either a list of a tuple of [D, H, W](in order of z, y, and x).
            anchor_range (torch.Tensor | list[float]): Range of anchors with
                shape [6]. The order is consistent with that of anchors, i.e.,
                (x_min, y_min, z_min, x_max, y_max, z_max).
            scale (float | int, optional): The scale factor of anchors.
                Defaults to 1.
            sizes (list[list] | np.ndarray | torch.Tensor, optional):
                Anchor size with shape [N, 3], in order of x, y, z.
                Defaults to [[3.9, 1.6, 1.56]].
            rotations (list[float] | np.ndarray | torch.Tensor, optional):
                Rotations of anchors in a single feature grid.
                Defaults to [0, 1.5707963].
            device (str): Devices that the anchors will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors with shape
                [*feature_size, num_sizes, num_rots, 7].
        """
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = torch.tensor(anchor_range, device=device)
        z_centers = torch.linspace(
            anchor_range[2], anchor_range[5], feature_size[0], device=device)
        y_centers = torch.linspace(
            anchor_range[1], anchor_range[4], feature_size[1], device=device)
        x_centers = torch.linspace(
            anchor_range[0], anchor_range[3], feature_size[2], device=device)
        sizes = torch.tensor(sizes, device=device).reshape(-1, 3) * scale
        rotations = torch.tensor(rotations, device=device)

        # torch.meshgrid default behavior is 'id', np's default is 'xy'
        rets = torch.meshgrid(x_centers, y_centers, z_centers, rotations)
        # torch.meshgrid returns a tuple rather than list
        rets = list(rets)
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).repeat(tile_shape).unsqueeze(-1)

        sizes = sizes.reshape([1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)
        rets.insert(3, sizes)

        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])
        # [1, 200, 176, N, 2, 7] for kitti after permute

        # if len(self.custom_values) > 0:
        #     custom_ndim = len(self.custom_values)
        #     custom = ret.new_zeros([*ret.shape[:-1], custom_ndim])
        #     # custom[:] = self.custom_values
        #     ret = torch.cat([ret, custom], dim=-1)
        #     # [1, 200, 176, N, 2, 9] for nus dataset after permute
        return ret

# This function duplicates functionality of mmcv.ops.iou_3d.nms_bev
# from mmcv<=1.5, but using cuda ops from mmcv.ops.nms.nms_rotated.
# Nms api will be unified in mmdetection3d one day.
    def nms_bev(self, boxes, scores, thresh, pre_max_size=None, post_max_size=None):
        """NMS function GPU implementation (for BEV boxes). The overlap of two
        boxes for IoU calculation is defined as the exact overlapping area of the
        two boxes. In this function, one can also set ``pre_max_size`` and
        ``post_max_size``.

        Args:
            boxes (torch.Tensor): Input boxes with the shape of [N, 5]
                ([x1, y1, x2, y2, ry]).
            scores (torch.Tensor): Scores of boxes with the shape of [N].
            thresh (float): Overlap threshold of NMS.
            pre_max_size (int, optional): Max size of boxes before NMS.
                Default: None.
            post_max_size (int, optional): Max size of boxes after NMS.
                Default: None.

        Returns:
            torch.Tensor: Indexes after NMS.
        """
        assert boxes.size(1) == 5, 'Input boxes shape should be [N, 5]'
        order = scores.sort(0, descending=True)[1]
        if pre_max_size is not None:
            order = order[:pre_max_size]
        boxes = boxes[order].contiguous()
        scores = scores[order]

        # xyxyr -> back to xywhr
        # note: better skip this step before nms_bev call in the future
        boxes = torch.stack(
            ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
                boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1], boxes[:, 4]),
            dim=-1)

        keep = nms_rotated(boxes, scores, thresh)[1]
        keep = order[keep]
        if post_max_size is not None:
            keep = keep[:post_max_size]
        return keep


    # This function duplicates functionality of mmcv.ops.iou_3d.nms_normal_bev
    # from mmcv<=1.5, but using cuda ops from mmcv.ops.nms.nms.
    # Nms api will be unified in mmdetection3d one day.
    def nms_normal_bev(self, boxes, scores, thresh):
        """Normal NMS function GPU implementation (for BEV boxes). The overlap of
        two boxes for IoU calculation is defined as the exact overlapping area of
        the two boxes WITH their yaw angle set to 0.

        Args:
            boxes (torch.Tensor): Input boxes with shape (N, 5).
            scores (torch.Tensor): Scores of predicted boxes with shape (N).
            thresh (float): Overlap threshold of NMS.

        Returns:
            torch.Tensor: Remaining indices with scores in descending order.
        """
        assert boxes.shape[1] == 5, 'Input boxes shape should be [N, 5]'
        return nms(boxes[:, :-1], scores, thresh)[1]

    def xywhr2xyxyr(self, boxes_xywhr):
        """Convert a rotated boxes in XYWHR format to XYXYR format.

        Args:
            boxes_xywhr (torch.Tensor | np.ndarray): Rotated boxes in XYWHR format.

        Returns:
            (torch.Tensor | np.ndarray): Converted boxes in XYXYR format.
        """
        boxes = torch.zeros_like(boxes_xywhr)
        half_w = boxes_xywhr[..., 2] / 2
        half_h = boxes_xywhr[..., 3] / 2

        boxes[..., 0] = boxes_xywhr[..., 0] - half_w
        boxes[..., 1] = boxes_xywhr[..., 1] - half_h
        boxes[..., 2] = boxes_xywhr[..., 0] + half_w
        boxes[..., 3] = boxes_xywhr[..., 1] + half_h
        boxes[..., 4] = boxes_xywhr[..., 4]
        return boxes

    def decode(self, anchors, deltas):
        """Apply transformation `deltas` (dx, dy, dz, dx_size, dy_size,
        dz_size, dr, dv*) to `boxes`.

        Args:
            anchors (torch.Tensor): Parameters of anchors with shape (N, 7).
            deltas (torch.Tensor): Encoded boxes with shape
                (N, 7+n) [x, y, z, x_size, y_size, z_size, r, velo*].

        Returns:
            torch.Tensor: Decoded boxes.
        """
        cas, cts = [], []
        box_ndim = anchors.shape[-1]
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(deltas, 1, dim=-1)
        else:
            xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)

        za = za + ha / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)

    def nms_3d(self):

        # cfg = self.test_cfg if cfg is None else cfg
        # assert len(cls_self.scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []

        # for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
        #         cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
        self.scores = torch.from_numpy(np.vstack(self.scores).astype(np.float))
        self.class_ids = torch.from_numpy(np.vstack(self.class_ids).astype(np.float))
        self.boxes = torch.from_numpy(np.vstack(self.boxes).astype(np.float))
        assert self.scores.size()[-2:] == self.boxes.size()[-2:]
        assert self.scores.size()[-2:] == self.class_ids.size()[-2:]
        self.class_ids = self.class_ids.permute(1, 2, 0).reshape(-1, 2)
        dir_cls_score = torch.max(self.class_ids, dim=-1)[1]

        self.scores = self.scores.permute(1, 2,
                                        0).reshape(-1, 3)   # TODO change 3  to dynamic Number of classes
        if self.use_sigmoid_cls:
            scores = self.scores.sigmoid()
        else:
            scores = self.scores.softmax(-1)
        self.boxes = self.boxes.permute(1, 2,
                                        0).reshape(-1, self.box_code_size)   
        anchors = self.anchors.reshape(-1, self.box_code_size).cpu()
        
        # nms_pre = cfg.get('nms_pre', -1)
        nms_pre = 100
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            if self.use_sigmoid_cls:
                max_scores, _ = scores.max(dim=1)
            else:
                max_scores, _ = scores[:, :-1].max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            anchors = anchors[topk_inds, :]
            self.boxes = self.boxes[topk_inds, :]
            self.scores = self.scores[topk_inds, :]
            self.class_ids = self.class_ids[topk_inds]

        bboxes = self.decode(anchors, self.boxes)
        mlvl_bboxes.append(self.boxes)
        mlvl_scores.append(self.scores)
        mlvl_dir_scores.append(self.class_ids)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = self.xywhr2xyxyr(mlvl_bboxes[:, [0, 1, 3, 4, 6]])
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = 0.1 # TODO remove hard coded
        cfg = None
        results = self.box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_scores, score_thr,50,
                                       cfg, mlvl_dir_scores,)
        bboxes, scores, labels, dir_scores = results
        # if bboxes.shape[0] > 0:
        #     dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
        #                            self.dir_limit_offset, np.pi)
        #     bboxes[..., 6] = (
        #         dir_rot + self.dir_offset +
        #         np.pi * dir_scores.to(bboxes.dtype))
        # bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
        return bboxes, scores, labels

    def nms_rotated(self,
                dets: Tensor,
                scores: Tensor,
                iou_threshold: float,
                labels: Optional[Tensor] = None,
                clockwise: bool = True) -> Tuple[Tensor, Tensor]:
        """Performs non-maximum suppression (NMS) on the rotated boxes according to
        their intersection-over-union (IoU).

        Rotated NMS iteratively removes lower scoring rotated boxes which have an
        IoU greater than iou_threshold with another (higher scoring) rotated box.

        Args:
            dets (torch.Tensor):  Rotated boxes in shape (N, 5).
                They are expected to be in
                (x_ctr, y_ctr, width, height, angle_radian) format.
            scores (torch.Tensor): scores in shape (N, ).
            iou_threshold (float): IoU thresh for NMS.
            labels (torch.Tensor, optional): boxes' label in shape (N,).
            clockwise (bool): flag indicating whether the positive angular
                orientation is clockwise. default True.
                `New in version 1.4.3.`

        Returns:
            tuple: kept dets(boxes and scores) and indice, which is always the
            same data type as the input.
        """
        if dets.shape[0] == 0:
            return dets, None
        if not clockwise:
            flip_mat = dets.new_ones(dets.shape[-1])
            flip_mat[-1] = -1
            dets_cw = dets * flip_mat
        else:
            dets_cw = dets
        multi_label = labels is not None
        if multi_label:
            dets_wl = torch.cat((dets_cw, labels.unsqueeze(1)), 1)  # type: ignore
        else:
            dets_wl = dets_cw
        _, order = scores.sort(0, descending=True)
        dets_sorted = dets_wl.index_select(0, order)

        if torch.__version__ == 'parrots':
            keep_inds = ext_module.nms_rotated(
                dets_wl,
                scores,
                order,
                dets_sorted,
                iou_threshold=iou_threshold,
                multi_label=multi_label)
        else:
            keep_inds = torch.ops.detectron2.nms_rotated(dets_wl, scores, order, dets_sorted,
                                            iou_threshold, multi_label)
        dets = torch.cat((dets[keep_inds], scores[keep_inds].reshape(-1, 1)),
                        dim=1)
        return dets, keep_inds