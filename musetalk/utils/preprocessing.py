import sys
from face_detection import FaceAlignment,LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import torch
from tqdm import tqdm

# Add insightface imports
import onnxruntime as ort

# initialize the mmpose model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
model = init_model(config_file, checkpoint_file, device=device)

# initialize the face detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType._2D, flip_input=False,device=device)

# maker if the bbox is not sufficient 
coord_placeholder = (0.0,0.0,0.0,0.0)

# InsightFace models
class InsightFaceSCRFD:
    """SCRFD face detection model from InsightFace"""
    def __init__(self, model_path="../insightface/buffalo_l/det_10g.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_size = (640, 640)
        self.det_thresh = 0.5
        self.nms_thresh = 0.4
        self.center_cache = {}
        self._init_model()
        
    def _init_model(self):
        """Initialize ONNX model"""
        if os.path.exists(self.model_path):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get model info
            input_cfg = self.session.get_inputs()[0]
            self.input_name = input_cfg.name
            input_shape = input_cfg.shape
            
            # Check if model supports dynamic input size
            if isinstance(input_shape[2], str):
                self.input_size = (640, 640)  # Default size
            else:
                self.input_size = tuple(input_shape[2:4][::-1])
                
            outputs = self.session.get_outputs()
            self.output_names = [o.name for o in outputs]
            
            # Determine model architecture
            self.use_kps = len(outputs) in [9, 15]  # Models with keypoints
            if len(outputs) in [6, 9]:
                self.fmc = 3
                self._feat_stride_fpn = [8, 16, 32]
                self._num_anchors = 2
            else:
                self.fmc = 5
                self._feat_stride_fpn = [8, 16, 32, 64, 128]
                self._num_anchors = 1
                
            self.input_mean = 127.5
            self.input_std = 128.0
            print(f"SCRFD model loaded: {self.model_path}")
        else:
            print(f"Warning: SCRFD model not found at {self.model_path}")
            
    def detect(self, img, max_num=0):
        """Detect faces in image"""
        if self.session is None:
            return np.array([]), None
            
        # Prepare input
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(self.input_size[1]) / self.input_size[0]
        
        if im_ratio > model_ratio:
            new_height = self.input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = self.input_size[0]
            new_height = int(new_width * im_ratio)
            
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        
        # Run inference
        blob = cv2.dnn.blobFromImage(det_img, 1.0/self.input_std, self.input_size, 
                                   (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        
        # Process outputs
        scores_list, bboxes_list, kpss_list = self._process_outputs(net_outs, det_img.shape[:2])
        
        if not scores_list:
            return np.array([]), None
            
        # Combine results
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        
        if self.use_kps and kpss_list:
            kpss = np.vstack(kpss_list) / det_scale
        else:
            kpss = None
            
        # NMS
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self._nms(pre_det)
        det = pre_det[keep, :]
        
        if kpss is not None:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
            
        # Limit number of faces
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
                
        return det, kpss
        
    def _process_outputs(self, net_outs, input_shape):
        """Process model outputs to get detections"""
        scores_list = []
        bboxes_list = []
        kpss_list = []
        
        input_height, input_width = input_shape
        fmc = self.fmc
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc] * stride
            
            if self.use_kps:
                kps_preds = net_outs[idx + fmc * 2] * stride
                
            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers
                    
            pos_inds = np.where(scores >= self.det_thresh)[0]
            if len(pos_inds) > 0:
                bboxes = self._distance2bbox(anchor_centers, bbox_preds)
                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]
                scores_list.append(pos_scores)
                bboxes_list.append(pos_bboxes)
                
                if self.use_kps:
                    kpss = self._distance2kps(anchor_centers, kps_preds)
                    kpss = kpss.reshape((kpss.shape[0], -1, 2))
                    pos_kpss = kpss[pos_inds]
                    kpss_list.append(pos_kpss)
                    
        return scores_list, bboxes_list, kpss_list
        
    def _distance2bbox(self, points, distance):
        """Convert distance predictions to bounding boxes"""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)
        
    def _distance2kps(self, points, distance):
        """Convert distance predictions to keypoints"""
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)
        
    def _nms(self, dets):
        """Non-Maximum Suppression"""
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


class InsightFaceLandmark:
    """1k3d68 landmark model from InsightFace"""
    def __init__(self, model_path="../insightface/buffalo_l/1k3d68.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_size = (192, 192)  # Default input size for 1k3d68
        self._init_model()
        
    def _init_model(self):
        """Initialize ONNX model"""
        if os.path.exists(self.model_path):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get model info
            input_cfg = self.session.get_inputs()[0]
            self.input_name = input_cfg.name
            input_shape = input_cfg.shape
            self.input_size = tuple(input_shape[2:4][::-1])
            
            outputs = self.session.get_outputs()
            self.output_names = [o.name for o in outputs]
            
            # Check model type by output shape
            output_shape = outputs[0].shape
            if output_shape[1] == 3309:  # 3D landmarks
                self.lmk_dim = 3
                self.lmk_num = 68
            else:
                self.lmk_dim = 2
                self.lmk_num = output_shape[1] // 2
                
            # Set normalization parameters
            self.input_mean = 127.5
            self.input_std = 128.0
            
            print(f"Landmark model loaded: {self.model_path}, landmarks: {self.lmk_num}D-{self.lmk_dim}")
        else:
            print(f"Warning: Landmark model not found at {self.model_path}")
            
    def get_landmarks(self, img, bbox):
        """Get facial landmarks from bounding box - FIXED: match official InsightFace implementation"""
        if self.session is None:
            return None
            
        # Calculate face center and dimensions (matching official InsightFace)
        x1, y1, x2, y2 = bbox[:4]
        w, h = x2 - x1, y2 - y1
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Use official InsightFace scaling calculation
        scale = self.input_size[0] / (max(w, h) * 1.5)
        rotate = 0
        
        # Apply transformation using official method
        aimg, M = self._transform(img, center, self.input_size[0], scale, rotate)
        
        # Prepare input blob (matching official InsightFace)
        blob = cv2.dnn.blobFromImage(aimg, 1.0/self.input_std, self.input_size,
                                   (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        
        # Run inference
        pred = self.session.run(self.output_names, {self.input_name: blob})[0][0]
        
        # Process output exactly like official InsightFace landmark.py
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
            
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num * -1:, :]
            
        # CRITICAL FIX: Use official InsightFace coordinate processing
        # Add 1 to shift from [-1,1] to [0,2] range
        pred[:, 0:2] += 1
        # Scale to input image coordinates
        pred[:, 0:2] *= (self.input_size[0] // 2)
        # For 3D landmarks, scale Z coordinate too
        if pred.shape[1] == 3:
            pred[:, 2] *= (self.input_size[0] // 2)
            
        # Transform back to original image coordinates
        IM = cv2.invertAffineTransform(M)
        pred = self._trans_points(pred, IM)
        
        return pred
        
    def get_all_landmarks(self, img, bbox):
        """Get ALL landmarks from the model (not just 68) for better bbox calculation"""
        if self.session is None:
            return None
            
        # Calculate face center and dimensions (matching official InsightFace)
        x1, y1, x2, y2 = bbox[:4]
        w, h = x2 - x1, y2 - y1
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Use official InsightFace scaling calculation
        scale = self.input_size[0] / (max(w, h) * 1.5)
        rotate = 0
        
        # Apply transformation using official method
        aimg, M = self._transform(img, center, self.input_size[0], scale, rotate)
        
        # Prepare input blob (matching official InsightFace)
        blob = cv2.dnn.blobFromImage(aimg, 1.0/self.input_std, self.input_size,
                                   (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        
        # Run inference
        pred = self.session.run(self.output_names, {self.input_name: blob})[0][0]
        
        # Process output exactly like official InsightFace landmark.py
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
            
        # DON'T trim to just 68 landmarks - use ALL landmarks
        print(f"Model output: {pred.shape[0]} landmarks")
            
        # CRITICAL FIX: Use official InsightFace coordinate processing
        # Add 1 to shift from [-1,1] to [0,2] range
        pred[:, 0:2] += 1
        # Scale to input image coordinates
        pred[:, 0:2] *= (self.input_size[0] // 2)
        # For 3D landmarks, scale Z coordinate too
        if pred.shape[1] == 3:
            pred[:, 2] *= (self.input_size[0] // 2)
            
        # Transform back to original image coordinates
        IM = cv2.invertAffineTransform(M)
        pred = self._trans_points(pred, IM)
        
        return pred
        
    def _transform(self, img, center, output_size, scale, rotation):
        """Apply transformation matching official InsightFace face_align.transform"""
        from skimage import transform as trans
        
        scale_ratio = scale
        rot = float(rotation) * np.pi / 180.0
        
        # Use the exact transformation sequence from official InsightFace
        t1 = trans.SimilarityTransform(scale=scale_ratio)
        cx = center[0] * scale_ratio
        cy = center[1] * scale_ratio
        t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
        t3 = trans.SimilarityTransform(rotation=rot)
        t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
        t = t1 + t2 + t3 + t4
        M = t.params[0:2]
        
        # Apply transformation
        cropped = cv2.warpAffine(img, M, (output_size, output_size), borderValue=0.0)
        return cropped, M
        
    def _trans_points(self, pts, M):
        """Transform points using transformation matrix - matching official InsightFace trans_points"""
        if pts.shape[1] == 2:
            return self._trans_points2d(pts, M)
        else:
            return self._trans_points3d(pts, M)
    
    def _trans_points2d(self, pts, M):
        """Transform 2D points - matching official InsightFace"""
        new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
        for i in range(pts.shape[0]):
            pt = pts[i]
            new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
            new_pt = np.dot(M, new_pt)
            new_pts[i] = new_pt[0:2]
        return new_pts

    def _trans_points3d(self, pts, M):
        """Transform 3D points - matching official InsightFace"""
        scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
        new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
        for i in range(pts.shape[0]):
            pt = pts[i]
            new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
            new_pt = np.dot(M, new_pt)
            new_pts[i][0:2] = new_pt[0:2]
            new_pts[i][2] = pts[i][2] * scale
        return new_pts


# Initialize InsightFace models
scrfd_detector = InsightFaceSCRFD("../insightface/buffalo_l/det_10g.onnx")
landmark_model = InsightFaceLandmark("../insightface/buffalo_l/1k3d68.onnx")

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_bbox_range(img_list, upperbondrange=0, use_insightface=True):
    """
    Get bbox range information for parameter adjustment.
    
    Args:
        img_list: List of image paths
        upperbondrange: Bbox adjustment parameter
        use_insightface: If True, use InsightFace models, else use MMPose
    """
    if use_insightface:
        return get_bbox_range_insightface(img_list, upperbondrange)
    else:
        return get_bbox_range_mmpose(img_list, upperbondrange)


def get_bbox_range_insightface(img_list, upperbondrange=0):
    """InsightFace-based bbox range calculation"""
    frames = read_imgs(img_list)
    
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with InsightFace, bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with InsightFace, default value')
    
    average_range_minus = []
    average_range_plus = []
    
    for frame in tqdm(frames):
        # Detect faces using SCRFD
        det_results, kps = scrfd_detector.detect(frame, max_num=1)
        
        if len(det_results) == 0:
            continue
            
        # Get the best detection
        bbox = det_results[0]
        
        # Get detailed landmarks using 1k3d68 model
        landmarks = landmark_model.get_landmarks(frame, bbox)
        
        if landmarks is not None and landmarks.shape[0] >= 68:
            face_land_mark = landmarks[:68].astype(np.int32)
            
            # Use nose area landmarks for adjustment
            range_minus = (face_land_mark[31] - face_land_mark[30])[1] if len(face_land_mark) > 31 else 5
            range_plus = (face_land_mark[30] - face_land_mark[29])[1] if len(face_land_mark) > 29 else 5
            
            average_range_minus.append(abs(range_minus))
            average_range_plus.append(abs(range_plus))
    
    if average_range_minus and average_range_plus:
        avg_minus = int(sum(average_range_minus) / len(average_range_minus))
        avg_plus = int(sum(average_range_plus) / len(average_range_plus))
        text_range = f"Total frame:「{len(frames)}」 Manually adjust range : [ -{avg_minus}~{avg_plus} ] , the current value: {upperbondrange}"
    else:
        text_range = f"Total frame:「{len(frames)}」 Current bbox_shift value: {upperbondrange}"
    
    return text_range


def get_bbox_range_mmpose(img_list, upperbondrange=0):
    """Original MMPose-based bbox range calculation"""
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with MMPose, bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with MMPose, default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark= keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        
        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                continue
            
            half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30]- face_land_mark[29])[1]
            range_plus = (face_land_mark[29]- face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）

    text_range=f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    return text_range
    

def get_landmark_and_bbox_insightface(img_list, upperbondrange=0, debug_dir=None):
    """HYBRID: SCRFD detection + face-aligned 1k3d68 landmarks (ONNX-only, Unity-ready)"""
    frames = read_imgs(img_list)
    coords_list = []
    
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with HYBRID (SCRFD+1k3d68), bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with HYBRID (SCRFD+1k3d68), default value')
    
    average_range_minus = []
    average_range_plus = []
    
    for idx, frame in enumerate(tqdm(frames)):
        # Detect faces using SCRFD
        det_results, kps = scrfd_detector.detect(frame, max_num=1)  # Get the best face
        
        if len(det_results) == 0:
            coords_list.append(coord_placeholder)
            continue
            
        # Get the best detection
        bbox = det_results[0]  # [x1, y1, x2, y2, score]
        x1, y1, x2, y2 = bbox[:4].astype(int)
        original_bbox = (x1, y1, x2, y2)
        
        # IMPROVED APPROACH: Use SCRFD detection + proper 1k3d68 landmarks
        # The key insight: 1k3d68 needs face-aligned cropping, not just bbox cropping
        
        # Step 1: Get proper 1k3d68 landmarks using SCRFD keypoints for better face alignment
        landmarks_68 = None
        if kps is not None and len(kps) > 0:
            scrfd_kps = kps[0]
            if scrfd_kps.shape[0] >= 5:
                # Use SCRFD keypoints to create a better face-aligned crop for 1k3d68
                # This is the key fix: align the face properly before landmark extraction
                left_eye = scrfd_kps[0]
                right_eye = scrfd_kps[1]
                nose_tip = scrfd_kps[2]
                left_mouth = scrfd_kps[3]
                right_mouth = scrfd_kps[4]
                
                # Calculate face center and scale using eye distance (like face recognition)
                eye_center = (left_eye + right_eye) / 2
                eye_distance = np.linalg.norm(right_eye - left_eye)
                
                # Create a more accurate bbox for 1k3d68 based on facial geometry
                face_scale = eye_distance * 2.2  # Empirical factor for good face crop
                
                # Center the crop around eye center but shift down slightly for better coverage
                crop_center_x = eye_center[0]
                crop_center_y = eye_center[1] + eye_distance * 0.3  # Shift down for mouth coverage
                
                # Create square crop for 1k3d68 (it expects square input)
                half_size = face_scale / 2
                crop_x1 = max(0, int(crop_center_x - half_size))
                crop_y1 = max(0, int(crop_center_y - half_size))
                crop_x2 = min(frame.shape[1], int(crop_center_x + half_size))
                crop_y2 = min(frame.shape[0], int(crop_center_y + half_size))
                
                # Make it square by taking the minimum dimension
                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1
                crop_size = min(crop_w, crop_h)
                
                # Recalculate with square dimensions
                crop_x1 = max(0, int(crop_center_x - crop_size/2))
                crop_y1 = max(0, int(crop_center_y - crop_size/2))
                crop_x2 = min(frame.shape[1], crop_x1 + crop_size)
                crop_y2 = min(frame.shape[0], crop_y1 + crop_size)
                
                aligned_bbox = (crop_x1, crop_y1, crop_x2, crop_y2)
                
                print(f"SCRFD keypoints: eyes dist={eye_distance:.1f}, face_scale={face_scale:.1f}")
                print(f"Aligned bbox: {aligned_bbox} = {crop_x2-crop_x1}x{crop_y2-crop_y1}")
                
                # Extract landmarks using the aligned bbox
                landmarks_68 = landmark_model.get_landmarks(frame, aligned_bbox)
        
        # Step 2: Process landmarks and create final bbox
        if landmarks_68 is not None and landmarks_68.shape[0] >= 68:
            # Use the properly extracted 68-point landmarks
            face_land_mark = landmarks_68[:68].astype(np.int32)
            
            # Use standard MMPose-compatible landmark indices
            nose_tip = face_land_mark[30]  # Standard nose tip
            nose_bridge_top = face_land_mark[27]  # Top of nose bridge
            
            # Calculate ranges like MMPose
            range_minus = abs((face_land_mark[30] - face_land_mark[29])[1]) if len(face_land_mark) > 30 else 20
            range_plus = abs((face_land_mark[29] - face_land_mark[28])[1]) if len(face_land_mark) > 29 else 20
            
            half_face_coord = face_land_mark[29].copy()  # Use landmark 29 like MMPose
            
            print(f"1k3d68 landmarks: nose_tip={nose_tip}, landmark_29={half_face_coord}")
            print(f"Calculated ranges: minus={range_minus}, plus={range_plus}")
            
            # Apply bbox shift if specified
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange + half_face_coord[1]
            
            # Create face bbox using landmark center but SCRFD-like size for proper blending
            # Use landmark center for accurate positioning, but maintain reasonable face coverage
            
            # Get landmark center and bounds
            landmark_center_x = np.mean(face_land_mark[:, 0])
            landmark_center_y = np.mean(face_land_mark[:, 1])
            
            # Use SCRFD detection size as reference for proper face coverage
            fx1, fy1, fx2, fy2 = original_bbox
            scrfd_w = fx2 - fx1
            scrfd_h = fy2 - fy1
            
            # Create bbox centered on landmarks but with SCRFD-like dimensions
            # This ensures we have enough face area for blending while being landmark-accurate
            face_w = int(scrfd_w * 0.9)  # Slightly smaller than SCRFD for precision
            face_h = int(scrfd_h * 0.9)
            
            lx1 = max(0, int(landmark_center_x - face_w / 2))
            ly1 = max(0, int(landmark_center_y - face_h / 2))
            lx2 = min(frame.shape[1], lx1 + face_w)
            ly2 = min(frame.shape[0], ly1 + face_h)
            
            landmark_bbox = (lx1, ly1, lx2, ly2)
            
            # Validate landmark bbox
            if ly2 - ly1 <= 0 or lx2 - lx1 <= 0 or lx1 < 0:
                print(f"Invalid landmark bbox: {landmark_bbox}, using SCRFD bbox")
                # Fallback to SCRFD with minimal expansion
                fx1, fy1, fx2, fy2 = original_bbox
                expansion_factor = 1.05
                center_x, center_y = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                scrfd_w, scrfd_h = fx2 - fx1, fy2 - fy1
                new_w, new_h = scrfd_w * expansion_factor, scrfd_h * expansion_factor
                lx1 = max(0, int(center_x - new_w / 2))
                ly1 = max(0, int(center_y - new_h / 2))
                lx2 = min(frame.shape[1], int(center_x + new_w / 2))
                ly2 = min(frame.shape[0], int(center_y + new_h / 2))
            
            f_landmark = (lx1, ly1, lx2, ly2)
            
        else:
            print("Failed to extract 1k3d68 landmarks, using SCRFD bbox")
            # Fallback to SCRFD approach
            fx1, fy1, fx2, fy2 = original_bbox
            expansion_factor = 1.05
            center_x, center_y = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            scrfd_w, scrfd_h = fx2 - fx1, fy2 - fy1
            new_w, new_h = scrfd_w * expansion_factor, scrfd_h * expansion_factor
            lx1 = max(0, int(center_x - new_w / 2))
            ly1 = max(0, int(center_y - new_h / 2))
            lx2 = min(frame.shape[1], int(center_x + new_w / 2))
            ly2 = min(frame.shape[0], int(center_y + new_h / 2))
            f_landmark = (lx1, ly1, lx2, ly2)
            
            # Use SCRFD keypoints for range calculation
            if kps is not None and len(kps) > 0:
                scrfd_kps = kps[0]
                if scrfd_kps.shape[0] >= 5:
                    nose_tip = scrfd_kps[2].astype(int)
                    mouth_center = ((scrfd_kps[3] + scrfd_kps[4]) / 2).astype(int)
                    nose_to_mouth_dist = abs(mouth_center[1] - nose_tip[1])
                    range_minus = range_plus = max(10, nose_to_mouth_dist // 3)
                    half_face_coord = nose_tip.copy()
                else:
                    range_minus = range_plus = 20
                    half_face_coord = np.array([int(center_x), int(center_y)])
            else:
                range_minus = range_plus = 20
                half_face_coord = np.array([int(center_x), int(center_y)])
        
        average_range_minus.append(abs(range_minus))
        average_range_plus.append(abs(range_plus))
                
        # Debug: Save comparison images
        if debug_dir and idx == 0:  # Only debug first frame
            debug_frame = frame.copy()
            # Draw original detection bbox in red
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(debug_frame, 'SCRFD Det', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw aligned crop bbox in blue (if available)
            if 'aligned_bbox' in locals():
                ax1, ay1, ax2, ay2 = aligned_bbox
                cv2.rectangle(debug_frame, (ax1, ay1), (ax2, ay2), (255, 0, 0), 2)
                cv2.putText(debug_frame, 'Aligned Crop', (ax1, ay1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw final landmark bbox in green  
            cv2.rectangle(debug_frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
            cv2.putText(debug_frame, 'Final Bbox', (lx1, ly1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw the 5 SCRFD keypoints
            if kps is not None and len(kps) > 0:
                scrfd_kps = kps[0]
                keypoint_names = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
                for i, (kp, name) in enumerate(zip(scrfd_kps, keypoint_names)):
                    cv2.circle(debug_frame, (int(kp[0]), int(kp[1])), 4, (0, 255, 255), -1)  # Yellow circles
                    cv2.putText(debug_frame, name, (int(kp[0])+5, int(kp[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Draw 1k3d68 landmarks if available
            if 'face_land_mark' in locals() and landmarks_68 is not None:
                # Draw key 68-point landmarks
                key_landmarks = [8, 27, 28, 29, 30, 33, 36, 39, 42, 45, 48, 54, 57, 64]  # Key facial points
                for pt_idx in key_landmarks:
                    if pt_idx < len(face_land_mark):
                        pt = face_land_mark[pt_idx]
                        cv2.circle(debug_frame, (int(pt[0]), int(pt[1])), 2, (255, 0, 255), -1)  # Magenta
                        cv2.putText(debug_frame, str(pt_idx), (int(pt[0])+3, int(pt[1])-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Save debug image
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"hybrid_debug_{idx}.jpg"), debug_frame)
            
            # Save cropped regions
            if ly2 - ly1 > 0 and lx2 - lx1 > 0 and lx1 >= 0 and ly1 >= 0:
                crop_final = frame[ly1:ly2, lx1:lx2]
                cv2.imwrite(os.path.join(debug_dir, f"hybrid_crop_final_{idx}.jpg"), crop_final)
            
            crop_original = frame[y1:y2, x1:x2] 
            cv2.imwrite(os.path.join(debug_dir, f"hybrid_crop_original_{idx}.jpg"), crop_original)
            
            # Save aligned crop if available
            if 'aligned_bbox' in locals():
                ax1, ay1, ax2, ay2 = aligned_bbox
                if ay2 - ay1 > 0 and ax2 - ax1 > 0:
                    crop_aligned = frame[ay1:ay2, ax1:ax2]
                    cv2.imwrite(os.path.join(debug_dir, f"hybrid_crop_aligned_{idx}.jpg"), crop_aligned)
            
            print(f"DEBUG Hybrid SCRFD+1k3d68 - Frame {idx}:")
            print(f"  Original SCRFD bbox: {original_bbox}")
            if 'aligned_bbox' in locals():
                print(f"  Aligned crop bbox: {aligned_bbox}")
            print(f"  Final landmark bbox: {f_landmark}")
            if 'landmarks_68' in locals() and landmarks_68 is not None:
                print(f"  Successfully extracted {len(landmarks_68)} landmarks")
            print(f"  Half face coord: {half_face_coord}")
        
        # Validate landmark bbox
        if ly2 - ly1 <= 0 or lx2 - lx1 <= 0 or lx1 < 0:
            print(f"Error landmark bbox: {f_landmark}, using detection bbox")
            coords_list.append(tuple(bbox[:4]))
        else:
            coords_list.append(f_landmark)
    
    # Print adjustment information
    print("********************************************bbox_shift parameter adjustment (InsightFace)**********************************************************")
    if average_range_minus and average_range_plus:
        avg_minus = int(sum(average_range_minus) / len(average_range_minus))
        avg_plus = int(sum(average_range_plus) / len(average_range_plus))
        print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{avg_minus}~{avg_plus} ] , the current value: {upperbondrange}")
    else:
        print(f"Total frame:「{len(frames)}」 Current bbox_shift value: {upperbondrange}")
    print("*************************************************************************************************************************************")
    
    return coords_list, frames


def get_landmark_and_bbox(img_list, upperbondrange=0, use_insightface=True, debug_dir=None):
    """
    Main function for face detection and landmarking.
    
    Args:
        img_list: List of image paths
        upperbondrange: Bbox adjustment parameter  
        use_insightface: If True, use InsightFace models (SCRFD + 1k3d68), 
                        else use MMPose + face_detection
        debug_dir: Directory to save debug images (optional)
    """
    if use_insightface:
        return get_landmark_and_bbox_insightface(img_list, upperbondrange, debug_dir)
    else:
        return get_landmark_and_bbox_mmpose(img_list, upperbondrange, debug_dir)


def get_landmark_and_bbox_mmpose(img_list, upperbondrange=0, debug_dir=None):
    """Original MMPose-based implementation (kept for reference)"""
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with MMPose, bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with MMPose, default value')
    average_range_minus = []
    average_range_plus = []
    frame_idx = 0
    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark= keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        
        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                continue
            
            original_bbox = f
            half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30]- face_land_mark[29])[1]
            range_plus = (face_land_mark[29]- face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）
            half_face_dist = np.max(face_land_mark[:,1]) - half_face_coord[1]
            upper_bond = half_face_coord[1]-half_face_dist
            
            f_landmark = (np.min(face_land_mark[:, 0]),int(upper_bond),np.max(face_land_mark[:, 0]),np.max(face_land_mark[:,1]))
            x1, y1, x2, y2 = f_landmark
            
            # Debug: Save comparison images  
            if debug_dir and frame_idx == 0:  # Only debug first frame
                frame = fb[0]
                debug_frame = frame.copy()
                
                # Draw original detection bbox in red
                fx1, fy1, fx2, fy2 = f
                cv2.rectangle(debug_frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
                cv2.putText(debug_frame, 'Face Det', (fx1, fy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw landmark-based bbox in green
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_frame, 'MMPose Landmark Bbox', (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw key landmarks
                key_points = [28, 29, 30]  # Nose area used by MMPose
                for pt_idx in key_points:
                    if pt_idx < len(face_land_mark):
                        pt = face_land_mark[pt_idx]
                        cv2.circle(debug_frame, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)
                        cv2.putText(debug_frame, str(pt_idx), (int(pt[0])+5, int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Draw ALL MMPose face landmarks
                for pt_idx, pt in enumerate(face_land_mark):
                    cv2.circle(debug_frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 255), -1)
                
                # Save debug image
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"mmpose_debug_{frame_idx}.jpg"), debug_frame)
                
                # Save cropped regions
                if y2 - y1 > 0 and x2 - x1 > 0 and x1 >= 0 and y1 >= 0:
                    crop_landmark = frame[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(debug_dir, f"mmpose_crop_landmark_{frame_idx}.jpg"), crop_landmark)
                
                crop_original = frame[fy1:fy2, fx1:fx2]
                cv2.imwrite(os.path.join(debug_dir, f"mmpose_crop_original_{frame_idx}.jpg"), crop_original)
                
                print(f"DEBUG MMPose - Frame {frame_idx}:")
                print(f"  Original Face Detection bbox: {original_bbox}")
                print(f"  Landmark bbox: {f_landmark}")
                print(f"  Half face coord (landmark 29): {half_face_coord}")
                print(f"  Key landmarks 28,29,30: {face_land_mark[28]}, {face_land_mark[29]}, {face_land_mark[30]}")
                print(f"  Face landmarks shape: {face_land_mark.shape}")
            
            if y2-y1<=0 or x2-x1<=0 or x1<0: # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                w,h = f[2]-f[0], f[3]-f[1]
                print("error bbox:",f)
            else:
                coords_list += [f_landmark]
            
            frame_idx += 1
    
    print("********************************************bbox_shift parameter adjustment (MMPose)**********************************************************")
    print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}")
    print("*************************************************************************************************************************************")
    return coords_list,frames
    

if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png","./results/lyria/00001.png","./results/lyria/00002.png","./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list,full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)
        
    for bbox, frame in zip(coords_list,full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print('Cropped shape', crop_frame.shape)
        
        #cv2.imwrite(path.join(save_dir, '{}.png'.format(i)),full_frames[i][0][y1:y2, x1:x2])
    print(coords_list)
