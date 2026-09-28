import numpy as np
import pytest

from roadwatch.config import LaneConfig
from roadwatch.perception.lanes import LaneModel, LanePipeline, LaneStateExtractor, _clusters

H, W = 720, 1280


def masks(lines=(), thickness=8):
    """lines: list of ((x_top, y_top), (x_bottom, y_bottom)) straight lane lines."""
    import cv2

    lane = np.zeros((H, W), np.uint8)
    for p0, p1 in lines:
        cv2.line(lane, p0, p1, 1, thickness)
    drivable = np.zeros((H, W), bool)
    drivable[H // 2:, 300:980] = True
    return lane.astype(bool), drivable


# Symmetric ego lane converging towards the top (perspective).
LEFT = ((560, 400), (240, 719))
RIGHT = ((720, 400), (1040, 719))


def test_clusters_merge_small_gaps():
    row = np.zeros(100, bool)
    row[10:14] = True
    row[16:18] = True  # gap 2 -> merged with the first run
    row[50:53] = True
    row[90] = True  # single pixel below min_px=2 -> dropped
    # Centre = mean of member pixels: (10+11+12+13+16+17)/6.
    assert _clusters(row, gap=4, min_px=2) == pytest.approx([79 / 6, 51.0])


def test_two_straight_lines():
    ext = LaneStateExtractor()
    st = ext.update(*masks([LEFT, RIGHT]))
    assert st.coverage == 1.0
    assert st.offset == pytest.approx(0.0, abs=0.02)
    assert all(x is not None for x in st.left_x + st.right_x)
    assert st.left_x[2] < st.left_x[0] and st.right_x[2] > st.right_x[0]  # wider near the car
    assert st.quality >= LaneConfig().good_quality
    assert st.drivable_ratio == pytest.approx(680 / 1280)


def test_empty_mask():
    st = LaneStateExtractor().update(*masks([]))
    assert st.coverage == 0.0
    assert st.offset is None
    assert st.quality < 0.2
    assert all(x is None for x in st.left_x + st.right_x)


def test_single_line_has_no_coverage():
    st = LaneStateExtractor().update(*masks([LEFT]))
    assert st.coverage == 0.0 and st.offset is None
    assert st.quality < LaneConfig().good_quality


def test_offset_sign_when_car_is_right_of_lane_center():
    shift = -200  # lane drawn 200 px to the left -> image centre is right of lane centre
    lines = [((a[0] + shift, a[1]), (b[0] + shift, b[1])) for a, b in (LEFT, RIGHT)]
    st = LaneStateExtractor().update(*masks(lines))
    assert st.offset is not None and st.offset > 0.2


def test_width_stability_drops_on_width_jump():
    ext = LaneStateExtractor()
    for _ in range(5):
        stable = ext.update(*masks([LEFT, RIGHT]))
    assert stable.width_stability == pytest.approx(1.0)
    wide = ext.update(*masks([((460, 400), (40, 719)), ((820, 400), (1240, 719))]))
    assert wide.width_stability < 0.8


def test_diverging_upper_border_rejected():
    # Right border goes outward going up (e.g. a barrier line) -> upper rows rejected.
    st = LaneStateExtractor().update(*masks([LEFT, ((1270, 480), (1040, 719))]))
    assert st.right_x[2] is not None  # bottom row fine
    assert st.right_x[0] is None  # 70% row diverges -> dropped


def test_ema_smooths_borders():
    ext = LaneStateExtractor(LaneConfig(border_ema_alpha=0.5))
    a = ext.update(*masks([LEFT, RIGHT]))
    shifted = [((x0 + 40, y0), (x1 + 40, y1)) for (x0, y0), (x1, y1) in (LEFT, RIGHT)]
    b = ext.update(*masks(shifted))
    assert b.left_x[2] == pytest.approx(a.left_x[2] + 20, abs=2)


# ---------------------------------------------------------------- integration

MODEL = LaneConfig().model_path


@pytest.mark.models
@pytest.mark.skipif(not MODEL.exists(), reason=f"model missing: {MODEL}")
def test_yolop_runs_and_maps_outputs():
    model = LaneModel(num_threads=2)
    assert "lane" in model.lane_output and "drive" in model.drivable_output
    img = np.full((H, W, 3), 90, np.uint8)
    lf = LanePipeline(model)(img)
    assert lf.lane_mask.shape == (H, W) and lf.drivable_mask.shape == (H, W)
    assert 0.0 <= lf.state.quality <= 1.0
