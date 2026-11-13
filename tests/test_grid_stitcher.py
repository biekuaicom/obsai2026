"""冻结拼图：相似变换、拒绝失败匹配、重叠像素不回写、星座压过周期焊盘。"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from pcb_ar.core.geo import invert_affine, matrix_rotation_deg
from pcb_ar.core.grid_stitcher import (
    _Shape,
    _apply,
    _assign_cells,
    _calibrate_grid,
    _compose,
    _fit_similarity,
    _match_constellation,
    _overlap_geometry,
    _rect_cover_count,
    _roi_intersection,
    _search_rough,
    stitch_grid_folder,
)

_BOARD = (36, 92, 40)
_INK = (26, 26, 30)
_TRACE = (48, 148, 72)
_PAD = (190, 210, 230)


def _tile_to_scene(origin: tuple[float, float], size: tuple[int, int], angle_deg: float) -> np.ndarray:
    width, height = size
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float64)
    angle = math.radians(angle_deg)
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = rotation
    matrix[:, 2] = np.asarray(origin, dtype=np.float64) + center - rotation @ center
    return matrix


def _cut_tile(scene: np.ndarray, origin: tuple[float, float], size: tuple[int, int], angle_deg: float) -> np.ndarray:
    matrix = _tile_to_scene(origin, size, angle_deg)
    width, height = size
    # warpAffine 默认把矩阵当 src→dst 再取逆。这里矩阵已经是 tile→scene，
    # 需要按 dst→src 直接采样，图块像素才对应真实板面位置。
    return cv2.warpAffine(
        scene,
        matrix.astype(np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_BOARD,
    )


def _paint_board(width: int = 760, height: int = 560) -> np.ndarray:
    scene = np.full((height, width, 3), _BOARD, np.uint8)
    # 重叠区里的唯一封装、安装孔和长走线。周期焊盘不能单独解释它们的相对位置。
    cv2.rectangle(scene, (210, 46), (318, 102), _INK, -1)
    cv2.circle(scene, (248, 156), 13, (0, 0, 0), -1)
    cv2.line(scene, (214, 122), (360, 136), _TRACE, 4)
    cv2.rectangle(scene, (430, 70), (498, 104), (22, 22, 40), -1)
    cv2.circle(scene, (470, 150), 11, (0, 0, 0), -1)
    cv2.line(scene, (40, 80), (150, 96), _TRACE, 4)
    cv2.rectangle(scene, (48, 150), (120, 198), (34, 34, 48), -1)
    for x in range(188, 430, 26):
        for y in range(188, 250, 26):
            cv2.rectangle(scene, (x, y), (x + 8, y + 8), _PAD, -1)
    for x in range(40, 180, 26):
        for y in range(210, 250, 26):
            cv2.rectangle(scene, (x, y), (x + 8, y + 8), _PAD, -1)
    return scene


def _write_tiles(directory: Path, tiles: list[np.ndarray]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, tile in enumerate(tiles):
        path = directory / f"tile_{index}.png"
        cv2.imwrite(str(path), tile)
        paths.append(path)
    return paths


def _relative(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    return _compose(invert_affine(pose_a).astype(np.float64), pose_b)


def _max_point_error(got: np.ndarray, expected: np.ndarray, width: int, height: int) -> float:
    points = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1], [width / 2, height / 2]],
        dtype=np.float64,
    )
    errs = [float(np.linalg.norm(_apply(got, p) - _apply(expected, p))) for p in points]
    return max(errs)


def test_rotated_pair_keeps_full_similarity(tmp_path: Path):
    size = (360, 260)
    origin_a = (24.0, 30.0)
    origin_b = (184.0, 30.0)
    angle = 2.0
    scene = _paint_board()
    tiles = [
        _cut_tile(scene, origin_a, size, 0.0),
        _cut_tile(scene, origin_b, size, angle),
    ]
    result = stitch_grid_folder(
        _write_tiles(tmp_path, tiles),
        rows=1,
        cols=2,
        serpentine=False,
        step=(160.0, 220.0),
        max_side=4000,
    )
    assert result.placed == 2, result.message
    assert result.poses is not None and result.poses[0] is not None and result.poses[1] is not None
    rotation = matrix_rotation_deg(result.poses[1]) - matrix_rotation_deg(result.poses[0])
    assert abs(rotation) > 0.5
    assert abs(abs(rotation) - angle) < 0.6
    # 线性部分必须带上旋转，不能只留平移。
    assert abs(float(result.poses[1][0, 1])) > 1e-3 or abs(float(result.poses[1][1, 0])) > 1e-3
    expected = _compose(
        invert_affine(_tile_to_scene(origin_a, size, 0.0)).astype(np.float64),
        _tile_to_scene(origin_b, size, angle),
    )
    got = _relative(result.poses[0], result.poses[1])
    error = _max_point_error(got, expected, size[0], size[1])
    assert error < 4.0, error


def test_failed_match_is_rejected_and_does_not_pollute_frozen(tmp_path: Path):
    size = (360, 260)
    scene = _paint_board()
    good = _cut_tile(scene, (24.0, 30.0), size, 0.0)
    rng = np.random.default_rng(0)
    bad = rng.integers(0, 40, size=(size[1], size[0], 3), dtype=np.uint8)
    bad[:] = (240, 240, 240)
    cv2.rectangle(bad, (30, 30), (150, 90), (40, 40, 220), -1)
    cv2.circle(bad, (220, 160), 30, (0, 180, 255), -1)
    bad[12:36, 180:210] = (255, 0, 255)
    result = stitch_grid_folder(
        _write_tiles(tmp_path, [good, bad]),
        rows=1,
        cols=2,
        step=(160.0, 220.0),
        max_side=4000,
    )
    assert result.placed == 1, result.message
    assert result.poses is not None and result.poses[1] is None
    magenta = (result.image[:, :, 0] > 240) & (result.image[:, :, 1] < 15) & (result.image[:, :, 2] > 240)
    assert not np.any(magenta)

    cells = _assign_cells(2, 1, 2, 160.0, 220.0, False)
    frozen, cover = _calibrate_grid(cells, [good, bad], 160.0, 220.0)
    assert cells[0].accepted and not cells[1].accepted
    xs = np.where(cover > 0)[1]
    assert int(xs.max() - xs.min() + 1) <= good.shape[1] + 2
    frozen_magenta = (frozen[:, :, 0] > 240) & (frozen[:, :, 1] < 15) & (frozen[:, :, 2] > 240)
    assert not np.any(frozen_magenta)


def test_overlap_pixels_of_frozen_tile_are_not_replaced(tmp_path: Path):
    size = (360, 260)
    origin_a = (24.0, 30.0)
    origin_b = (184.0, 30.0)
    scene = _paint_board()
    tile_a = _cut_tile(scene, origin_a, size, 0.0)
    tile_b = _cut_tile(scene, origin_b, size, 1.2)
    # 重叠区内部同一物理位置涂成不同颜色。冻结图必须留下第一张的红色。
    tile_a[210:222, 230:242] = (0, 0, 255)
    scene_pt = np.array([origin_a[0] + 236.0, origin_a[1] + 216.0, 1.0])
    inv_b = invert_affine(_tile_to_scene(origin_b, size, 1.2)).astype(np.float64)
    uv = inv_b @ scene_pt
    x = int(round(float(uv[0])))
    y = int(round(float(uv[1])))
    assert 8 < x < size[0] - 16 and 8 < y < size[1] - 16
    tile_b[y - 6 : y + 6, x - 6 : x + 6] = (255, 0, 255)

    cells = _assign_cells(2, 1, 2, 160.0, 220.0, False)
    frozen, _cover = _calibrate_grid(cells, [tile_a, tile_b], 160.0, 220.0)
    assert cells[1].accepted, "对照用的第二张应当能配上，否则看不出覆盖写"
    red = (frozen[:, :, 2] > 240) & (frozen[:, :, 1] < 12) & (frozen[:, :, 0] < 12)
    magenta = (frozen[:, :, 0] > 240) & (frozen[:, :, 1] < 12) & (frozen[:, :, 2] > 240)
    assert int(np.count_nonzero(red)) >= 20
    assert not np.any(magenta)

    result = stitch_grid_folder(
        _write_tiles(tmp_path, [tile_a, tile_b]),
        rows=1,
        cols=2,
        step=(160.0, 220.0),
        max_side=4000,
    )
    assert result.placed == 2
    assert result.poses is not None and result.poses[1] is not None
    assert abs(matrix_rotation_deg(result.poses[1])) > 0.4


def test_constellation_prefers_package_and_hole_over_pad_grid():
    pitch = 26.0
    src: list[_Shape] = []
    for i in range(8):
        for j in range(5):
            src.append(_Shape("pad", np.array([16.0 + i * pitch, 24.0 + j * pitch]), 8.0, 8.0, 0.0))
    component = _Shape("component", np.array([92.0, 48.0]), 88.0, 40.0, 0.02)
    hole = _Shape("hole", np.array([46.0, 132.0]), 26.0, 26.0, 0.0)
    trace = _Shape("trace", np.array([150.0, 96.0]), 84.0, 4.0, 0.15)
    src.extend([component, hole, trace])
    angle = math.radians(1.35)
    shift = np.array([14.0, -5.0])
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])

    def transform(point: np.ndarray) -> np.ndarray:
        return rotation @ point + shift

    dst = [
        _Shape(shape.kind, transform(shape.center), shape.major, shape.minor, shape.angle + angle)
        for shape in src
    ]
    # 最近焊盘会把 14px 平移吸到 0 或 26px。星座必须跟封装和孔走。
    match = _match_constellation(src, dst)
    assert match is not None
    assert match.anchor_hits >= 2
    assert all(src[i].kind != "pad" for i, _j in match.pairs)
    fitted = _fit_similarity(
        np.vstack([src[i].center for i, _j in match.pairs]),
        np.vstack([dst[j].center for _i, j in match.pairs]),
    )
    assert fitted is not None
    moved = _apply(fitted, component.center)
    truth = transform(component.center)
    error = float(np.linalg.norm(moved - truth))
    snapped = float(np.linalg.norm(moved - (truth + np.array([pitch, 0.0]))))
    assert error < 2.5, error
    assert error + 4.0 < snapped


def test_image_constellation_not_pulled_by_pad_pitch(tmp_path: Path):
    size = (360, 260)
    origin_a = (24.0, 30.0)
    # 相对网格多出来的平移不是焊盘间距 26 的整数倍。
    origin_b = (24.0 + 160.0 + 14.0, 30.0 - 5.0)
    angle = 1.5
    scene = _paint_board()
    tiles = [
        _cut_tile(scene, origin_a, size, 0.0),
        _cut_tile(scene, origin_b, size, angle),
    ]
    result = stitch_grid_folder(
        _write_tiles(tmp_path, tiles),
        rows=1,
        cols=2,
        serpentine=False,
        step=(160.0, 220.0),
        max_side=4000,
    )
    assert result.placed == 2, result.message
    assert result.poses is not None and result.poses[0] is not None and result.poses[1] is not None
    expected = _compose(
        invert_affine(_tile_to_scene(origin_a, size, 0.0)).astype(np.float64),
        _tile_to_scene(origin_b, size, angle),
    )
    got = _relative(result.poses[0], result.poses[1])
    error = _max_point_error(got, expected, size[0], size[1])
    pitch_shift = expected.copy()
    pitch_shift[0, 2] += 26.0
    snapped = _max_point_error(got, pitch_shift, size[0], size[1])
    assert error < 4.0, error
    assert error + 4.0 < snapped
    rotation = abs(matrix_rotation_deg(result.poses[1]) - matrix_rotation_deg(result.poses[0]))
    assert abs(rotation - angle) < 0.6


def test_serpentine_stays_opt_in(tmp_path: Path):
    size = (320, 240)
    origin = (16.0, 16.0)
    step = (150.0, 110.0)
    scene = _paint_board(680, 520)
    # 四张图都能看到中心的封装/孔；蛇形会把下排左右对调，核对应当拒绝。
    cv2.rectangle(scene, (190, 150), (286, 196), _INK, -1)
    cv2.circle(scene, (230, 214), 12, (0, 0, 0), -1)
    cv2.line(scene, (188, 204), (320, 214), _TRACE, 4)
    tiles = []
    for row in range(2):
        for col in range(2):
            ox = origin[0] + col * step[0]
            oy = origin[1] + row * step[1]
            tiles.append(_cut_tile(scene, (ox, oy), size, 0.0))
    paths = _write_tiles(tmp_path, tiles)
    forward = stitch_grid_folder(paths, rows=2, cols=2, serpentine=False, step=step, max_side=4000)
    snake = stitch_grid_folder(paths, rows=2, cols=2, serpentine=True, step=step, max_side=4000)
    assert forward.placed == 4, forward.message
    assert snake.placed < 4, snake.message


def _stamp_motif(scene: np.ndarray, x: int, y: int, tag: int) -> None:
    cv2.rectangle(scene, (x, y), (x + 64 + tag * 6, y + 34), _INK, -1)
    cv2.circle(scene, (x + 22, y + 56), 11, (0, 0, 0), -1)
    cv2.line(scene, (x, y + 46), (x + 88, y + 50), _TRACE, 4)


def test_measured_step_is_used_for_the_next_prediction():
    size = (280, 180)
    # 真实步长 200，overlap 猜测只有 140。第一对放下之后，第三张的预测必须改用测得步长。
    true_step = 200.0
    guess = 140.0
    origin = 16.0
    scene = np.full((260, 820, 3), _BOARD, np.uint8)
    _stamp_motif(scene, int(origin + true_step + 8), 28, 0)
    _stamp_motif(scene, int(origin + 2 * true_step + 8), 36, 2)
    tiles = [_cut_tile(scene, (origin + i * true_step, 20.0), size, 0.0) for i in range(3)]
    cells = _assign_cells(3, 1, 3, guess, 160.0, False)
    _calibrate_grid(cells, tiles, guess, 160.0)
    assert cells[0].accepted and cells[1].accepted and cells[2].accepted
    assert cells[2].predicted_step is not None
    measured = float(cells[2].predicted_step[0])
    assert abs(measured - true_step) < 16.0, measured
    assert abs(measured - guess) > 30.0
    # 同一行不应留下明显的上下错位。
    assert abs(float(cells[1].pose[1, 2] - cells[0].pose[1, 2])) < 2.0


def test_coarse_search_does_not_maximize_overlap_area():
    cover = np.zeros((150, 480), np.uint8)
    cover[:, 20:240] = 255
    width, height = 220, 120
    expected_w = 44.0
    rough_x = 240.0 - expected_w
    rough_y = 12.0
    found = _search_rough(
        cover,
        rough_x,
        rough_y,
        width,
        height,
        expected_overlap=(expected_w, float(height)),
        axis="x",
    )
    assert found is not None
    found_geom = _overlap_geometry(cover, found[0], found[1], width, height)
    assert found_geom is not None
    slide_x = 24.0
    slide_area = _rect_cover_count(cover, slide_x, rough_y, width, height)
    assert found_geom["area"] < slide_area
    assert abs(found_geom["ow"] - expected_w) < 16.0
    assert found[0] > slide_x + 80.0


def test_rejected_tile_does_not_overwrite_frozen_overlap():
    size = (280, 180)
    scene = np.full((240, 520, 3), _BOARD, np.uint8)
    _stamp_motif(scene, 150, 30, 1)
    good = _cut_tile(scene, (12.0, 16.0), size, 0.0)
    good[40:52, 150:162] = (0, 0, 255)
    bad = np.full_like(good, (240, 240, 240))
    bad[8:40, 8:40] = (255, 0, 255)
    cells = _assign_cells(2, 1, 2, 150.0, 160.0, False)
    frozen, _cover = _calibrate_grid(cells, [good, bad], 150.0, 160.0)
    assert cells[0].accepted and not cells[1].accepted
    red = (frozen[:, :, 2] > 240) & (frozen[:, :, 1] < 12) & (frozen[:, :, 0] < 12)
    magenta = (frozen[:, :, 0] > 240) & (frozen[:, :, 1] < 12) & (frozen[:, :, 2] > 240)
    assert int(np.count_nonzero(red)) >= 20
    assert not np.any(magenta)


def test_match_region_is_intersection_not_full_rectangle():
    height, width = 90, 120
    frozen = np.full((height, width * 2, 3), (10, 20, 30), np.uint8)
    cover = np.zeros((height, width * 2), np.uint8)
    cover[:, :70] = 255
    image = np.full((height, width, 3), (0, 0, 200), np.uint8)
    image[20:40, 10:30] = (0, 255, 0)
    rough = np.array([[1.0, 0.0, 40.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    packed = _roi_intersection(frozen, cover, image, rough)
    assert packed is not None
    frozen_roi, new_roi, mask, _origin, _local = packed
    assert int(np.count_nonzero(mask)) < mask.size
    assert new_roi.shape[1] < image.shape[1]
    assert mask.shape[0] * mask.shape[1] < image.shape[0] * image.shape[1]
    assert np.all(new_roi[~mask] == 0)
    assert np.all(frozen_roi[~mask] == 0)
    # 交集大约是 x=40..70，而不是整幅 120 宽的预测矩形。
    assert int(np.count_nonzero(mask)) < 50 * height
