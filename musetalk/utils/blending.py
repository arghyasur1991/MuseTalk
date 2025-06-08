from PIL import Image
import numpy as np
import cv2
import copy


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s


def face_seg(image, mode="raw", fp=None):
    """
    对图像进行面部解析，生成面部区域的掩码。

    Args:
        image (PIL.Image): 输入图像。

    Returns:
        PIL.Image: 面部区域的掩码图像。
    """
    seg_image = fp(image, mode=mode)  # 使用 FaceParsing 模型解析面部
    if seg_image is None:
        print("error, no person_segment")  # 如果没有检测到面部，返回错误
        return None

    seg_image = seg_image.resize(image.size)  # 将掩码图像调整为输入图像的大小
    return seg_image


def get_image(image, face, face_box, upper_boundary_ratio=0.5, expand=1.5, mode="raw", fp=None, debug_dir=None, frame_idx=None):
    """
    将裁剪的面部图像粘贴回原始图像，并进行一些处理。

    Args:
        image (numpy.ndarray): 原始图像（身体部分）。
        face (numpy.ndarray): 裁剪的面部图像。
        face_box (tuple): 面部边界框的坐标 (x, y, x1, y1)。
        upper_boundary_ratio (float): 用于控制面部区域的保留比例。
        expand (float): 扩展因子，用于放大裁剪框。
        mode: 融合mask构建方式 
        debug_dir: 调试目录，保存中间结果
        frame_idx: 帧索引，用于调试文件命名

    Returns:
        numpy.ndarray: 处理后的图像。
    """
    import os
    
    # 将 numpy 数组转换为 PIL 图像
    body = Image.fromarray(image[:, :, ::-1])  # 身体部分图像(整张图)
    face = Image.fromarray(face[:, :, ::-1])  # 面部图像

    x, y, x1, y1 = face_box  # 获取面部边界框的坐标
    crop_box, s = get_crop_box(face_box, expand)  # 计算扩展后的裁剪框
    x_s, y_s, x_e, y_e = crop_box  # 裁剪框的坐标
    face_position = (x, y)  # 面部在原始图像中的位置

    # 从身体图像中裁剪出扩展后的面部区域（下巴到边界有距离）
    face_large = body.crop(crop_box)
        
    ori_shape = face_large.size  # 裁剪后图像的原始尺寸

    # Debug: Save face_large crop
    if debug_dir and frame_idx is not None and frame_idx < 3:
        os.makedirs(debug_dir, exist_ok=True)
        face_large_np = np.array(face_large)[:, :, ::-1]  # Convert to BGR for OpenCV
        cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_idx:03d}_face_large_crop.jpg"), face_large_np)

    # 对裁剪后的面部区域进行面部解析，生成掩码
    mask_image = face_seg(face_large, mode=mode, fp=fp)
    
    # Debug: Save face segmentation mask
    if debug_dir and frame_idx is not None and frame_idx < 3 and mask_image is not None:
        mask_np = np.array(mask_image)
        cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_idx:03d}_face_seg_mask.jpg"), mask_np)
    
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 裁剪出面部区域的掩码
    
    # Debug: Save small mask
    if debug_dir and frame_idx is not None and frame_idx < 3:
        mask_small_np = np.array(mask_small)
        cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_idx:03d}_mask_small.jpg"), mask_small_np)
    
    mask_image = Image.new('L', ori_shape, 0)  # 创建一个全黑的掩码图像
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 将面部掩码粘贴到全黑图像上
    
    # Debug: Save full mask before boundary modification
    if debug_dir and frame_idx is not None and frame_idx < 3:
        mask_full_np = np.array(mask_image)
        cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_idx:03d}_mask_full.jpg"), mask_full_np)
    
    # 保留面部区域的上半部分（用于控制说话区域）
    # For InsightFace, use more aggressive boundary to match MMPose coverage
    width, height = mask_image.size
    
    # Adjust boundary ratio based on mode for better quality matching
    if mode == "jaw":  # v15 mode
        adjusted_ratio = upper_boundary_ratio * 0.8  # More aggressive for v15
    else:
        adjusted_ratio = upper_boundary_ratio * 0.85  # Slightly more aggressive for v1
        
    top_boundary = int(height * adjusted_ratio)  # 计算上半部分的边界
    modified_mask_image = Image.new('L', ori_shape, 0)  # 创建一个新的全黑掩码图像
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))  # 粘贴上半部分掩码
    
    # Debug: Save modified mask after boundary cut
    if debug_dir and frame_idx is not None and frame_idx < 3:
        modified_mask_np = np.array(modified_mask_image)
        cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_idx:03d}_mask_modified_boundary.jpg"), modified_mask_np)
        print(f"DEBUG: Frame {frame_idx} - Top boundary at {top_boundary} (adjusted ratio {adjusted_ratio:.3f}, orig {upper_boundary_ratio}), mask size: {width}x{height}")
    
    # 对掩码进行高斯模糊，使边缘更平滑
    # Increase blur kernel size for InsightFace to match MMPose smoothness
    base_blur_factor = 0.08 if mode == "jaw" else 0.06  # Larger blur for better blending
    blur_kernel_size = int(base_blur_factor * ori_shape[0] // 2 * 2) + 1  # 计算模糊核大小
    blur_kernel_size = max(blur_kernel_size, 15)  # Minimum blur kernel size
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)  # 高斯模糊
    #mask_array = np.array(modified_mask_image)
    mask_image = Image.fromarray(mask_array)  # 将模糊后的掩码转换回 PIL 图像
    
    # Debug: Save final blurred mask
    if debug_dir and frame_idx is not None and frame_idx < 3:
        final_mask_np = np.array(mask_image)
        cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_idx:03d}_mask_final_blurred.jpg"), final_mask_np)
        print(f"DEBUG: Frame {frame_idx} - Blur kernel size: {blur_kernel_size}")
    
    # 将裁剪的面部图像粘贴回扩展后的面部区域
    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    
    # Debug: Save face_large with pasted face
    if debug_dir and frame_idx is not None and frame_idx < 3:
        face_large_with_face_np = np.array(face_large)[:, :, ::-1]
        cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_idx:03d}_face_large_with_pasted_face.jpg"), face_large_with_face_np)
    
    body.paste(face_large, crop_box[:2], mask_image)
    
    body = np.array(body)  # 将 PIL 图像转换回 numpy 数组

    return body[:, :, ::-1]  # 返回处理后的图像（BGR 转 RGB）


def get_image_blending(image, face, face_box, mask_array, crop_box):
    body = Image.fromarray(image[:,:,::-1])
    face = Image.fromarray(face[:,:,::-1])

    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = body.crop(crop_box)

    mask_image = Image.fromarray(mask_array)
    mask_image = mask_image.convert("L")
    face_large.paste(face, (x-x_s, y-y_s, x1-x_s, y1-y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:,:,::-1]


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.5, fp=None, mode="raw"):
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box
