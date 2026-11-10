from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pcb_ar.core.geo import invert_affine
from pcb_ar.core.models import Project
from pcb_ar.core.project import ProjectStore
from pcb_ar.core.sparse_capture import (
    SparseCaptureConfig,
    SparseCaptureSession,
    SparseCaptureResult,
)
from pcb_ar.core.sparse_localizer import HybridTracker, SparseLocalizer
from pcb_ar.core.sparse_map import SparseMap
from pcb_ar.core.synthetic import synthetic_view
from pcb_ar.core.tracker import GlobalLocalTracker


@pytest.fixture(scope="module")
def sparse_bundle(board, tmp_path_factory):
    session = SparseCaptureSession(SparseCaptureConfig(min_sharpness=5))
    truths = []
    for cx, cy in [(1000, 1000), (1550, 1000), (2100, 1000), (2100, 1450)]:
        frame, matrix = synthetic_view(board, (cx, cy), 0.9, 1.5, (1600, 1000), seed=0)
        for _ in range(3):
            session.preview(frame)
        acceptance = session.capture()
        assert acceptance.accepted, acceptance.message
        truths.append(matrix)
    result = session.finish()
    out_dir = tmp_path_factory.mktemp("sparse_project")
    manifest = result.sparse_map.save(out_dir / "assets")
    loaded = SparseMap.load(manifest)
    return result, loaded, truths, out_dir


def test_project_sparse_fields_roundtrip():
    project = Project(name="x", sparse_map="assets/sparse_map.json", map_mode="hybrid")
    loaded = Project.from_dict(project.to_dict())
    assert loaded.sparse_map == "assets/sparse_map.json"
    assert loaded.map_mode == "hybrid"


def test_sparse_capture_result(sparse_bundle):
    result, loaded, truths, _ = sparse_bundle
    assert result.keyframe_count == 4
    assert result.overview.size > 0
    assert loaded.count == 4
    assert loaded.total_points > 2000
    assert loaded.board_width == result.overview.shape[1]
    assert loaded.board_height == result.overview.shape[0]
    assert all(k.coverage.shape == (4, 2) for k in loaded.keyframes)
    assert any(k.red_area_ratio > 0 for k in loaded.keyframes)
    assert all(len(k.red_hu) == 7 for k in loaded.keyframes)


def test_sparse_localization_accuracy(board, sparse_bundle):
    result, sparse, truths, _ = sparse_bundle
    localizer = SparseLocalizer(sparse)
    frame, live_truth = synthetic_view(board, (1600, 1030), 0.9, 1.5, (1600, 1000), seed=0)
    track = localizer.track(frame)
    assert track.state == "locked"
    assert track.matrix is not None
    assert track.inliers >= 20
    # 用第二个关键帧的真值构造期望的板坐标->实时帧矩阵
    keyframe = sparse.sorted_keyframes()[1]
    skf = keyframe.shape[1] / 1600.0
    truth_live_to_kf = (
        np.diag([skf, skf, 1.0])
        @ np.vstack([truths[1], [0, 0, 1]])
        @ np.linalg.inv(np.vstack([live_truth, [0, 0, 1]]))
    )
    expected_board_to_live = invert_affine(
        (np.vstack([keyframe.pose, [0, 0, 1]]) @ truth_live_to_kf)[:2]
    )
    corners = np.float32(
        [[0, 0], [sparse.board_width, 0], [sparse.board_width, sparse.board_height], [0, sparse.board_height]]
    ).reshape(-1, 1, 2)
    a = cv2.transform(corners, expected_board_to_live).reshape(-1, 2)
    b = cv2.transform(corners, track.matrix).reshape(-1, 2)
    assert float(np.mean(np.linalg.norm(a - b, axis=1))) < 10.0


def test_overview_merges_immediately_after_each_capture(board):
    session = SparseCaptureSession(SparseCaptureConfig(min_sharpness=5))
    sizes = []
    versions = []
    for cx, cy in [(1000, 1000), (1550, 1000)]:
        frame, _ = synthetic_view(board, (cx, cy), 0.9, 1.5, (1200, 760), seed=0)
        for _ in range(3):
            session.preview(frame)
        assert session.capture().accepted
        assert session.overview_image is not None
        sizes.append(session.overview_image.shape)
        versions.append(session.overview_version)
        assert session.coverage_union_mask is not None
        assert int(np.count_nonzero(session.coverage_union_mask)) > 0
    assert versions[1] > versions[0]
    assert sizes[1][1] > sizes[0][1]
    assert session.keyframes[1].pose_source >= 0
    assert session.seam_refined_count >= 1
    result = session.finish()
    assert result.overview.shape[1] == result.sparse_map.board_width
    assert result.overview.shape[0] == result.sparse_map.board_height
    assert result.overview.shape[0] <= session.overview_image.shape[0]


def test_overlap_rejects_duplicate_capture(board):
    session = SparseCaptureSession(SparseCaptureConfig(min_sharpness=5))
    frame, _ = synthetic_view(board, (1200, 1000), 0.9, 1.5, (1600, 1000), seed=0)
    for _ in range(3):
        session.preview(frame)
    assert session.capture().accepted
    preview = session.preview(frame)
    assert preview.covered_fraction > 0.9
    assert not preview.can_capture
    assert preview.reason == "overlap_high"


def test_sparse_project_save_as_copies_assets(sparse_bundle, tmp_path):
    result, _loaded, _truths, _ = sparse_bundle
    store = ProjectStore.create(tmp_path / "a.pcbproj.json", "A", result.overview)
    manifest = result.sparse_map.save(store.assets_dir)
    store.project.sparse_map = manifest.relative_to(store.path.parent).as_posix()
    store.project.map_mode = "hybrid"
    store.save()

    new_path = tmp_path / "sub" / "b.pcbproj.json"
    store.save_as(new_path)
    loaded = ProjectStore.load(new_path)
    assert loaded.sparse_map_path is not None
    assert loaded.sparse_map_path.exists()
    assert (loaded.assets_dir / "sparse_features.npz").exists()
    assert any((loaded.assets_dir / "keyframes").glob("*.jpg"))
    assert SparseMap.load(loaded.sparse_map_path).count == 4


def test_hybrid_tracker_uses_sparse_and_full(sparse_bundle, board):
    result, sparse, _truths, _ = sparse_bundle
    localizer = SparseLocalizer(sparse)
    full = GlobalLocalTracker(result.overview)
    hybrid = HybridTracker(localizer, full, full_interval=3)
    frame, _ = synthetic_view(board, (1600, 1030), 0.9, 1.5, (1600, 1000), seed=0)
    track = hybrid.track(frame)
    assert track.state == "locked"
    assert track.diagnostics.get("source") in ("sparse", "full")
    checked = [track.diagnostics.get("full_checked", False)]
    for _ in range(4):
        track = hybrid.track(frame)
        assert track.state == "locked"
        checked.append(track.diagnostics.get("full_checked", False))
    assert any(checked)
