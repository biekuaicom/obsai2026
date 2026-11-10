"""结构特征树 Phase B1：稳定基元 + 相对关系图。

这里不保存原始像素，只保存：
* 元件、焊盘、定位孔、长走线等稳定基元；
* 基元的尺寸/长宽比/方向；
* 基元之间的相对距离、角度、尺寸比。

它是后续“单元件 -> 多组件组合”结构匹配的数据基础。当前版本先做提取、
持久化和人工可查看的统计，不用它替换 ORB 定位。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pcb_ar.core.detector import detect_component_candidates
from pcb_ar.core.geo import as_homography

STRUCTURAL_MANIFEST = "structural_map.json"


@dataclass
class StructuralNode:
    id: int
    kind: str  # component / pad / hole / line
    bbox: tuple[int, int, int, int]
    area: int
    angle: float = 0.0
    source: str = "classical"

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0

    @property
    def width(self) -> int:
        return int(self.bbox[2])

    @property
    def height(self) -> int:
        return int(self.bbox[3])

    @property
    def aspect(self) -> float:
        return max(self.width, self.height) / float(max(1, min(self.width, self.height)))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "bbox": [int(v) for v in self.bbox],
            "area": int(self.area),
            "angle": float(self.angle),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "StructuralNode":
        bbox = tuple(int(v) for v in raw.get("bbox", [0, 0, 1, 1]))[:4]
        return cls(
            id=int(raw.get("id", 0)),
            kind=str(raw.get("kind", "component")),
            bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
            area=int(raw.get("area", 0)),
            angle=float(raw.get("angle", 0.0)),
            source=str(raw.get("source", "classical")),
        )


@dataclass
class StructuralEdge:
    source: int
    target: int
    distance: float
    angle: float
    size_ratio: float

    def to_dict(self) -> dict:
        return {
            "source": int(self.source),
            "target": int(self.target),
            "distance": float(self.distance),
            "angle": float(self.angle),
            "size_ratio": float(self.size_ratio),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "StructuralEdge":
        return cls(
            source=int(raw.get("source", 0)),
            target=int(raw.get("target", 0)),
            distance=float(raw.get("distance", 0.0)),
            angle=float(raw.get("angle", 0.0)),
            size_ratio=float(raw.get("size_ratio", 0.0)),
        )


@dataclass
class StructuralMap:
    width: int
    height: int
    nodes: list[StructuralNode] = field(default_factory=list)
    edges: list[StructuralEdge] = field(default_factory=list)
    version: int = 1

    @property
    def component_count(self) -> int:
        return sum(1 for n in self.nodes if n.kind == "component")

    @property
    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for node in self.nodes:
            result[node.kind] = result.get(node.kind, 0) + 1
        return result

    def describe(self) -> str:
        c = self.counts
        return (
            f"元件 {c.get('component', 0)} · 焊盘 {c.get('pad', 0)} · "
            f"定位孔 {c.get('hole', 0)} · 线段 {c.get('line', 0)} · "
            f"关系 {len(self.edges)}"
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "width": int(self.width),
            "height": int(self.height),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StructuralMap":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            width=int(raw.get("width", 0)),
            height=int(raw.get("height", 0)),
            nodes=[StructuralNode.from_dict(n) for n in raw.get("nodes", [])],
            edges=[StructuralEdge.from_dict(e) for e in raw.get("edges", [])],
            version=int(raw.get("version", 1)),
        )


def _node_id(nodes: list[StructuralNode]) -> int:
    return max((n.id for n in nodes), default=-1) + 1


def _inside_existing(nodes: list[StructuralNode], bbox: tuple[int, int, int, int], kind: str) -> bool:
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    for n in nodes:
        if n.kind != kind:
            continue
        nx, ny, nw, nh = n.bbox
        if nx <= cx <= nx + nw and ny <= cy <= ny + nh:
            return True
    return False


def _append_components(nodes: list[StructuralNode], image: np.ndarray) -> None:
    for det in detect_component_candidates(image):
        b = det.bbox
        nodes.append(
            StructuralNode(
                id=_node_id(nodes),
                kind="component",
                bbox=(int(b.x), int(b.y), int(b.w), int(b.h)),
                area=int(b.w * b.h),
                angle=0.0,
                source="detector",
            )
        )


def _append_holes(nodes: list[StructuralNode], image: np.ndarray) -> None:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    min_r = max(3, int(min(h, w) * 0.004))
    max_r = max(min_r + 2, int(min(h, w) * 0.035))
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, min_r * 2),
        param1=90,
        param2=24,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return
    for c in np.round(circles[0]).astype(int)[:40]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        bbox = (max(0, cx - r), max(0, cy - r), max(1, 2 * r), max(1, 2 * r))
        if _inside_existing(nodes, bbox, "hole"):
            continue
        nodes.append(
            StructuralNode(
                id=_node_id(nodes),
                kind="hole",
                bbox=bbox,
                area=int(math.pi * r * r),
                angle=0.0,
                source="hough",
            )
        )


def _append_pads(nodes: list[StructuralNode], image: np.ndarray) -> None:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    mx = image.max(axis=2)
    mn = image.min(axis=2)
    spread = mx.astype(np.int16) - mn.astype(np.int16)
    # 金属焊盘：亮、低饱和、RGB 三通道接近；绿色阻焊/彩色元件会被排除。
    mask = ((gray > 185) & (sat < 80) & (spread < 35)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if count >= 120:
            break
        area = float(cv2.contourArea(contour))
        if area < 25 or area > 0.002 * w * h:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if min(bw, bh) < 3 or max(bw, bh) > min(w, h) * 0.18:
            continue
        if max(bw, bh) / float(max(1, min(bw, bh))) > 7.0:
            continue
        extent = area / float(max(1, bw * bh))
        if extent < 0.45:
            continue
        bbox = (x, y, bw, bh)
        if _inside_existing(nodes, bbox, "pad"):
            continue
        nodes.append(
            StructuralNode(
                id=_node_id(nodes),
                kind="pad",
                bbox=bbox,
                area=int(area),
                angle=0.0,
                source="bright",
            )
        )
        count += 1


def _append_lines(nodes: list[StructuralNode], image: np.ndarray) -> None:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 130)
    diagonal = math.hypot(w, h)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=55,
        minLineLength=int(diagonal * 0.045),
        maxLineGap=5,
    )
    if lines is None:
        return
    kept: list[tuple[int, int, int, int]] = []
    for raw in lines[:, 0, :][:100]:
        x1, y1, x2, y2 = (int(v) for v in raw)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        duplicate = False
        for a, b, c, d in kept:
            if abs(a - x1) + abs(b - y1) + abs(c - x2) + abs(d - y2) < 18:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append((x1, y1, x2, y2))
        bbox = (min(x1, x2), min(y1, y2), max(1, abs(x2 - x1)), max(1, abs(y2 - y1)))
        nodes.append(
            StructuralNode(
                id=_node_id(nodes),
                kind="line",
                bbox=bbox,
                area=int(math.hypot(x2 - x1, y2 - y1)),
                angle=angle,
                source="hough",
            )
        )
        if len(kept) >= 60:
            break


def _build_edges(nodes: list[StructuralNode], width: int, height: int) -> list[StructuralEdge]:
    if len(nodes) < 2:
        return []
    diagonal = max(1.0, math.hypot(width, height))
    edges: list[StructuralEdge] = []
    seen: set[tuple[int, int]] = set()
    for i, a in enumerate(nodes):
        ax, ay = a.center
        distances = []
        for j, b in enumerate(nodes):
            if i == j:
                continue
            bx, by = b.center
            d = math.hypot(bx - ax, by - ay)
            if d > diagonal * 0.6:
                continue
            distances.append((d, j, bx, by))
        distances.sort(key=lambda item: item[0])
        for d, j, bx, by in distances[:3]:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            b = nodes[j]
            area_ratio = a.area / float(max(1, b.area))
            if area_ratio < 1.0:
                area_ratio = 1.0 / area_ratio
            edges.append(
                StructuralEdge(
                    source=i,
                    target=j,
                    distance=float(d / diagonal),
                    angle=float(math.degrees(math.atan2(by - ay, bx - ax))),
                    size_ratio=float(area_ratio),
                )
            )
    return edges


@dataclass
class StructuralMatch:
    ok: bool
    confidence: float
    inliers: int
    support: int
    scale: float = 1.0
    rotation: float = 0.0
    matrix: np.ndarray | None = None
    message: str = ""


def transform_structural_map(source: StructuralMap, matrix: np.ndarray) -> StructuralMap:
    """把结构图坐标（例如关键帧像素）变换到目标坐标（例如总览像素）。"""
    m = as_homography(matrix)
    scale = math.sqrt(max(1e-9, abs(float(np.linalg.det(m[:2, :2])))))
    rotation = math.degrees(math.atan2(float(m[1, 0]), float(m[0, 0])))
    nodes: list[StructuralNode] = []
    for n in source.nodes:
        cx, cy = n.center
        p = m @ np.array([cx, cy, 1.0])
        if abs(p[2]) < 1e-9:
            continue
        nx, ny = float(p[0] / p[2]), float(p[1] / p[2])
        nw = max(1, int(round(n.width * scale)))
        nh = max(1, int(round(n.height * scale)))
        nodes.append(
            StructuralNode(
                id=n.id,
                kind=n.kind,
                bbox=(int(round(nx - nw / 2)), int(round(ny - nh / 2)), nw, nh),
                area=max(1, int(round(n.area * scale * scale))),
                angle=n.angle + rotation,
                source=n.source,
            )
        )
    for idx, n in enumerate(nodes):
        n.id = idx
    return StructuralMap(
        width=int(round(source.width * scale)),
        height=int(round(source.height * scale)),
        nodes=nodes,
        edges=[],
    )


def _component_nodes(structural: StructuralMap, limit: int) -> list[StructuralNode]:
    comps = [n for n in structural.nodes if n.kind == "component"]
    comps.sort(key=lambda n: n.area, reverse=True)
    return comps[:limit]


def _apply_similarity(point: tuple[float, float], matrix: np.ndarray) -> tuple[float, float]:
    p = matrix @ np.array([point[0], point[1], 1.0])
    if abs(p[2]) < 1e-9:
        return -1e9, -1e9
    return float(p[0] / p[2]), float(p[1] / p[2])


def _score_hypothesis(
    reference: StructuralMap,
    query: StructuralMap,
    ref_components: list[StructuralNode],
    qry_components: list[StructuralNode],
    matrix: np.ndarray,
    tolerance: float,
) -> tuple[int, int, float]:
    """返回 (component inliers, auxiliary support, mean distance)。"""
    ref_pts = np.float32([r.center for r in ref_components])
    ref_areas = np.float32([max(1, r.area) for r in ref_components])
    q_pts_h = np.hstack(
        [np.float32([q.center for q in qry_components]), np.ones((len(qry_components), 1), np.float32)]
    )
    q_pts = (q_pts_h @ matrix.T.astype(np.float32))[:, :2]
    dist = np.linalg.norm(q_pts[:, None, :] - ref_pts[None, :, :], axis=2)
    q_areas = np.float32([max(1, q.area) for q in qry_components])
    ratio = q_areas[:, None] / ref_areas[None, :]
    valid = (dist <= tolerance) & (ratio >= 0.35) & (ratio <= 2.8)
    used = np.zeros(len(ref_components), dtype=bool)
    inliers = 0
    distances: list[float] = []
    for i in range(len(qry_components)):
        candidates = np.where(valid[i] & ~used)[0]
        if candidates.size == 0:
            continue
        j = int(candidates[np.argmin(dist[i, candidates])])
        used[j] = True
        inliers += 1
        distances.append(float(dist[i, j]))
    if inliers < 3:
        return inliers, 0, 0.0
    # 焊盘/孔/线段作为辅助证据，权重低于元件本体；限制数量防止匹配爆炸。
    support = 0
    aux = [n for n in query.nodes if n.kind != "component"][:18]
    ref_aux = [n for n in reference.nodes if n.kind != "component"][:24]
    aux_h = np.hstack([np.float32([q.center for q in aux]), np.ones((len(aux), 1), np.float32)]) if aux else np.zeros((0, 3), np.float32)
    aux_pts = (aux_h @ matrix.T.astype(np.float32))[:, :2] if len(aux) else np.zeros((0, 2), np.float32)
    for i, q in enumerate(aux):
        for r in ref_aux:
            if r.kind != q.kind:
                continue
            rx, ry = r.center
            if math.hypot(float(aux_pts[i, 0]) - rx, float(aux_pts[i, 1]) - ry) <= tolerance * 0.6:
                support += 1
                break
    mean_d = sum(distances) / len(distances) if distances else 0.0
    return inliers, support, mean_d


def match_structural_maps(
    reference: StructuralMap,
    query: StructuralMap,
    *,
    max_components: int = 10,
) -> StructuralMatch:
    """用元件中心的两两对应生成相似变换假设，做 RANSAC 式结构匹配。

    reference 是整板结构图，query 是局部关键帧/另一张结构图。返回
    query 坐标 -> reference 坐标的相似变换。
    """
    ref_components = _component_nodes(reference, max_components)
    qry_components = _component_nodes(query, max_components)
    if len(ref_components) < 3 or len(qry_components) < 3:
        return StructuralMatch(False, 0.0, 0, 0, message="元件不足 3 个，无法结构匹配")
    diag_ref = max(1.0, math.hypot(reference.width, reference.height))
    tolerance = max(8.0, diag_ref * 0.035)
    best: tuple[int, int, float, np.ndarray] | None = None
    q_pairs = [(i, j) for i in range(len(qry_components)) for j in range(i + 1, len(qry_components))]
    r_pairs = [(i, j) for i in range(len(ref_components)) for j in range(i + 1, len(ref_components))]
    for qi, qj in q_pairs:
        qa, qb = qry_components[qi], qry_components[qj]
        qax, qay = qa.center
        qbx, qby = qb.center
        dq = math.hypot(qbx - qax, qby - qay)
        if dq < 4.0:
            continue
        aq = math.atan2(qby - qay, qbx - qax)
        for ri, rj in r_pairs:
            ra, rb = ref_components[ri], ref_components[rj]
            rax, ray = ra.center
            rbx, rby = rb.center
            dr = math.hypot(rbx - rax, rby - ray)
            if dr < 4.0:
                continue
            scale = dr / dq
            if scale < 0.25 or scale > 4.0:
                continue
            angle = math.atan2(rby - ray, rbx - rax) - aq
            ca, sa = math.cos(angle), math.sin(angle)
            # qa 对齐到 ra
            matrix = np.array(
                [
                    [scale * ca, -scale * sa, rax - scale * (ca * qax - sa * qay)],
                    [scale * sa, scale * ca, ray - scale * (sa * qax + ca * qay)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            inliers, support, mean_d = _score_hypothesis(
                reference, query, ref_components, qry_components, matrix, tolerance
            )
            if inliers < 3:
                continue
            if best is None or inliers > best[0] or (inliers == best[0] and support > best[1]):
                best = (inliers, support, mean_d, matrix)
    if best is None:
        return StructuralMatch(False, 0.0, 0, 0, message="没有可靠的结构对应")
    inliers, support, mean_d, matrix = best
    # 用匹配上的元件中心做一次最小二乘精修。
    matched_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    used: set[int] = set()
    for q in qry_components:
        qx, qy = _apply_similarity(q.center, matrix)
        for j, r in enumerate(ref_components):
            if j in used:
                continue
            rx, ry = r.center
            if math.hypot(qx - rx, qy - ry) <= tolerance:
                matched_pairs.append((q.center, r.center))
                used.add(j)
                break
    if len(matched_pairs) >= 3:
        src = np.float32([p[0] for p in matched_pairs]).reshape(-1, 1, 2)
        dst = np.float32([p[1] for p in matched_pairs]).reshape(-1, 1, 2)
        refined, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if refined is not None:
            matrix = np.vstack([refined, [0, 0, 1]]).astype(np.float64)
            inliers, support, mean_d = _score_hypothesis(
                reference, query, ref_components, qry_components, matrix, tolerance
            )
    scale = math.sqrt(max(1e-9, abs(float(np.linalg.det(matrix[:2, :2])))))
    rotation = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    confidence = min(1.0, (inliers + 0.25 * support) / max(3.0, float(len(qry_components))))
    ok = inliers >= 3 and confidence >= 0.35
    return StructuralMatch(
        ok=ok,
        confidence=float(confidence),
        inliers=int(inliers),
        support=int(support),
        scale=float(scale),
        rotation=float(rotation),
        matrix=matrix,
        message=(
            f"结构匹配 {inliers} 内点 / {support} 辅助 / 置信度 {confidence:.0%}"
            if ok
            else f"结构匹配不足（{inliers} 内点，{confidence:.0%}）"
        ),
    )


def extract_structural_map(image: np.ndarray, *, max_nodes: int = 300) -> StructuralMap:
    if image is None or image.size == 0:
        return StructuralMap(width=0, height=0)
    h, w = image.shape[:2]
    nodes: list[StructuralNode] = []
    _append_components(nodes, image)
    _append_holes(nodes, image)
    _append_pads(nodes, image)
    _append_lines(nodes, image)
    # 组件优先保留；其余按类别截断，避免一张图生成过多噪点关系。
    components = [n for n in nodes if n.kind == "component"]
    others = [n for n in nodes if n.kind != "component"]
    max_others = max(0, max_nodes - len(components))
    nodes = components + others[:max_others]
    for idx, node in enumerate(nodes):
        node.id = idx
    edges = _build_edges(nodes, w, h)
    return StructuralMap(width=w, height=h, nodes=nodes, edges=edges)
