"""显微镜局部图拼接。

采用“按拍摄顺序配准 + 参考图择优 + 覆盖羽化”的保守策略，而不是完全依赖
``cv2.Stitcher``。PCB 上存在大量重复焊盘，纯自动全景容易错位；本实现要求相邻
图有 20%~30% 重叠，并对每张图与前序图做 ORB/RANSAC 验证。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from pcb_ar.core.geo import as_homography, is_valid_similarity


class StitchError(RuntimeError):
    pass


@dataclass
class PairMatch:
    index_a: int
    index_b: int
    matrix_b_to_a: np.ndarray
    inliers: int
    matches: int
    score: float
    overlap: float = 0.0
    correlation: float = 0.0


@dataclass
class StitchResult:
    image: np.ndarray
    transforms: list[np.ndarray]
    pairs: list[PairMatch] = field(default_factory=list)
    output_scale: float = 1.0
    canvas_size: tuple[int, int] = (0, 0)


ProgressCb = Callable[[int, int, str], None]


def _gray_work(image: np.ndarray, max_width: int = 1300) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_width / float(w))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray, scale


def estimate_pair_transform(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    ratio: float = 0.75,
    min_matches: int = 22,
    min_inliers: int = 16,
    ransac_px: float = 4.0,
) -> PairMatch | None:
    """估计图像 B -> 图像 A 的相似变换。"""
    gray_a, scale_a = _gray_work(image_a)
    gray_b, scale_b = _gray_work(image_b)
    # SIFT 对 PCB 上重复的焊盘/丝印更稳健；OpenCV 4.4+ 已内置。
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02, edgeThreshold=12)
        norm = cv2.NORM_L2
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        ratio_threshold = 0.7
    else:  # pragma: no cover - 仅极旧 OpenCV 回退
        detector = cv2.ORB_create(nfeatures=4000, fastThreshold=10)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        ratio_threshold = min(ratio, 0.68)
    kp_a, des_a = detector.detectAndCompute(gray_a, None)
    kp_b, des_b = detector.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None or len(kp_a) < min_matches or len(kp_b) < min_matches:
        return None
    knn = matcher.knnMatch(des_b, des_a, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good.append(m)
    if len(good) < min_matches:
        return None
    src = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 2)
    matrix_work, mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_px,
        maxIters=5000,
        confidence=0.999,
        refineIters=12,
    )
    if matrix_work is None or mask is None:
        return None
    inliers = int(mask.sum())
    if inliers < min_inliers:
        return None
    h = np.eye(3, dtype=np.float64)
    h[:2, :] = matrix_work
    # B 工作坐标 -> B 原图坐标：缩放；A 原图坐标 -> A 工作坐标：逆缩放
    inv_b = np.diag([1.0 / scale_b, 1.0 / scale_b, 1.0])
    a_scale = np.diag([scale_a, scale_a, 1.0])
    full = a_scale @ h @ inv_b
    matrix = full[:2, :].astype(np.float32)
    if not is_valid_similarity(matrix):
        return None
    # 重叠面积（B 四角映射到 A 后与 A 画面的交集）
    hb, wb = image_b.shape[:2]
    corners = np.float32([[0, 0], [wb, 0], [wb, hb], [0, hb]]).reshape(-1, 1, 2)
    mapped = cv2.transform(corners, matrix).reshape(-1, 2)
    ha, wa = image_a.shape[:2]
    frame = np.float32([[0, 0], [wa, 0], [wa, ha], [0, ha]])
    area, _ = cv2.intersectConvexConvex(mapped, frame)
    overlap = float(area) / float(max(1, wb * hb))
    if overlap < 0.08:
        return None
    correlation = _overlap_correlation(image_a, image_b, matrix)
    if correlation < 0.30:
        return None
    score = (
        inliers / max(len(good), 1) * min(1.0, inliers / 80.0)
        + min(0.2, overlap)
        + max(0.0, correlation) * 0.35
    )
    return PairMatch(
        -1,
        -1,
        matrix,
        inliers,
        len(good),
        float(score),
        overlap=overlap,
        correlation=correlation,
    )


def _overlap_correlation(image_a: np.ndarray, image_b: np.ndarray, matrix: np.ndarray) -> float:
    """在重叠区域比较梯度图相关性，剔除“几何上像但内容不对”的重复纹理。"""
    ha, wa = image_a.shape[:2]
    warped = cv2.warpAffine(
        image_b,
        matrix,
        (wa, ha),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask = cv2.warpAffine(
        np.full(image_b.shape[:2], 255, dtype=np.uint8),
        matrix,
        (wa, ha),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = mask > 0
    # 排除因图像本身黑边造成的伪重叠
    valid &= image_a.mean(axis=2) > 5
    valid &= warped.mean(axis=2) > 5
    if valid.sum() < 1000:
        return 0.0
    ga = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    ga = cv2.Laplacian(ga, cv2.CV_32F, ksize=3)
    gb = cv2.Laplacian(gb, cv2.CV_32F, ksize=3)
    a = ga[valid].astype(np.float64)
    b = gb[valid].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


def _transform_corners(matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    # 统一用 2x3 仿射变换；3x3 齐次矩阵直接传给 cv2.transform 会返回 3 个分量，
    # reshape 后变成错误数量的点，导致拼图画布范围不可信。
    return cv2.transform(corners, as_homography(matrix)[:2, :]).reshape(-1, 2)


def stitch_images(
    images: list[np.ndarray],
    progress: ProgressCb | None = None,
    *,
    max_canvas_side: int = 12000,
    max_canvas_pixels: int = 90_000_000,
    feather_px: int = 36,
) -> StitchResult:
    if not images:
        raise StitchError("没有可拼接的图像")
    if len(images) == 1:
        img = images[0].copy()
        return StitchResult(img, [np.eye(3, dtype=np.float32)], output_scale=1.0, canvas_size=(img.shape[1], img.shape[0]))
    for idx, image in enumerate(images):
        if image is None or image.size == 0:
            raise StitchError(f"第 {idx + 1} 张图像为空")

    total = len(images)
    if progress:
        progress(0, total, "提取并匹配特征…")

    transforms: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
    pairs: list[PairMatch] = []
    for i in range(1, total):
        best: PairMatch | None = None
        best_j = -1
        # 优先与前一张配准，失败时回溯更早的图（应对拍摄时跳过/回退）
        order = [i - 1] + [j for j in range(i - 2, -1, -1)]
        for j in order:
            candidate = estimate_pair_transform(images[j], images[i])
            if candidate is None:
                continue
            if best is None or candidate.score > best.score:
                best = candidate
                best_j = j
            if candidate.inliers >= 90:
                break
        if best is None or best_j < 0:
            raise StitchError(
                f"第 {i + 1} 张图无法与前序图像配准。请确保相邻画面有约 25% 重叠，"
                "并避免运动模糊。"
            )
        best.index_a = best_j
        best.index_b = i
        h_b = as_homography(best.matrix_b_to_a) @ as_homography(transforms[best_j])
        transforms.append(h_b)
        pairs.append(best)
        if progress:
            progress(i, total, f"第 {i + 1}/{total} 张配准完成（内点 {best.inliers}）")

    result = render_mosaic(
        images,
        transforms,
        progress,
        max_canvas_side=max_canvas_side,
        max_canvas_pixels=max_canvas_pixels,
        feather_px=feather_px,
    )
    result.pairs = pairs
    return result


def render_mosaic(
    images: list[np.ndarray],
    transforms: list[np.ndarray],
    progress: ProgressCb | None = None,
    *,
    max_canvas_side: int = 12000,
    max_canvas_pixels: int = 90_000_000,
    feather_px: int = 36,
) -> StitchResult:
    """按给定的 图像->世界 变换渲染拼图。

    供“视频扫描建图”复用：帧间位姿已由扫描会话累加，无需再次匹配。
    """
    if not images:
        raise StitchError("没有可拼接的图像")
    if len(images) != len(transforms):
        raise StitchError("图像数量与变换矩阵数量不一致")
    if len(images) == 1:
        img = images[0].copy()
        return StitchResult(
            img,
            [np.eye(3, dtype=np.float32)],
            output_scale=1.0,
            canvas_size=(img.shape[1], img.shape[0]),
        )
    for idx, image in enumerate(images):
        if image is None or image.size == 0:
            raise StitchError(f"第 {idx + 1} 张图像为空")

    total = len(images)
    all_corners = []
    for image, matrix in zip(images, transforms):
        h, w = image.shape[:2]
        all_corners.append(_transform_corners(matrix, w, h))
    corners = np.vstack(all_corners)
    min_x, min_y = np.floor(corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(corners.max(axis=0)).astype(int)
    margin = 12
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin
    canvas_w = int(max_x - min_x)
    canvas_h = int(max_y - min_y)
    if canvas_w <= 0 or canvas_h <= 0:
        raise StitchError("计算出的拼图画布尺寸无效")
    output_scale = 1.0
    if canvas_w > max_canvas_side or canvas_h > max_canvas_side:
        output_scale = min(max_canvas_side / canvas_w, max_canvas_side / canvas_h)
    if canvas_w * canvas_h * output_scale * output_scale > max_canvas_pixels:
        output_scale = min(output_scale, math.sqrt(max_canvas_pixels / (canvas_w * canvas_h)))
    output_scale = max(0.05, min(1.0, output_scale))
    canvas_w = max(1, int(round(canvas_w * output_scale)))
    canvas_h = max(1, int(round(canvas_h * output_scale)))
    if progress:
        progress(0, total, f"生成拼图 {canvas_w}x{canvas_h}…")

    offset = np.array(
        [
            [output_scale, 0, -min_x * output_scale],
            [0, output_scale, -min_y * output_scale],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    world = [offset @ as_homography(t) for t in transforms]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    covered = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    for i, (image, matrix) in enumerate(zip(images, world)):
        h, w = image.shape[:2]
        mapped = _transform_corners(matrix, w, h)
        x0 = max(0, int(math.floor(mapped[:, 0].min())) - 2)
        y0 = max(0, int(math.floor(mapped[:, 1].min())) - 2)
        x1 = min(canvas_w, int(math.ceil(mapped[:, 0].max())) + 2)
        y1 = min(canvas_h, int(math.ceil(mapped[:, 1].max())) + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        bw, bh = x1 - x0, y1 - y0
        local = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64) @ as_homography(matrix)
        warped = cv2.warpAffine(
            image,
            local[:2, :],
            (bw, bh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mask = cv2.warpAffine(
            np.full((h, w), 255, dtype=np.uint8),
            local[:2, :],
            (bw, bh),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if not np.any(mask):
            continue
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        alpha = np.clip(dist / float(max(1, feather_px)), 0.0, 1.0)
        roi_covered = covered[y0:y1, x0:x1]
        unseen = (mask > 0) & (roi_covered == 0)
        alpha[unseen] = 1.0
        alpha = alpha[..., None]
        roi = canvas[y0:y1, x0:x1].astype(np.float32)
        blended = roi * (1.0 - alpha) + warped.astype(np.float32) * alpha
        canvas[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        covered[y0:y1, x0:x1] = np.maximum(roi_covered, (mask > 0).astype(np.uint8) * 255)
        if progress:
            progress(i + 1, total, f"合成第 {i + 1}/{total} 张…")

    # 裁掉完全未覆盖的边缘，但保留少量安全边
    ys, xs = np.where(covered > 0)
    if len(xs) and len(ys):
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        if x0 > 0 or y0 > 0 or x1 < canvas_w or y1 < canvas_h:
            canvas = canvas[y0:y1, x0:x1]
            shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
            world = [shift @ m for m in world]
    if progress:
        progress(total, total, "拼图完成")
    return StitchResult(
        image=canvas,
        transforms=world,
        output_scale=output_scale,
        canvas_size=(canvas.shape[1], canvas.shape[0]),
    )
