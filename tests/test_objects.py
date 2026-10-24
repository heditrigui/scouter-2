import numpy as np

from scouter.objects import COCO, GRIDS, GRID_STRIDES, decode, letterbox


def test_letterbox_keeps_aspect_ratio():
    img = np.zeros((480, 640, 3), np.uint8)
    padded, ratio = letterbox(img)
    assert padded.shape == (640, 640, 3)
    assert ratio == 1.0
    assert padded[500, 0, 0] == 114  # padding below the image


def test_decode_one_box():
    raw = np.full((len(GRIDS), 85), -10.0, np.float32)
    raw[:, 4] = 0.0  # objectness 0 everywhere
    # Put a 'cup' on the stride-32 cell at grid (5, 3): centre (5.5*32, 3.5*32) = (176, 112), 64x32 px.
    i = int(np.where((GRIDS[:, 0] == 5) & (GRIDS[:, 1] == 3) & (GRID_STRIDES[:, 0] == 32))[0][0])
    raw[i, :4] = [0.5, 0.5, np.log(2.0), np.log(1.0)]
    raw[i, 4] = 0.9
    raw[i, 5:] = 0.0
    raw[i, 5 + COCO.index("cup")] = 1.0

    dets = decode(raw, ratio=0.5, min_score=0.5, nms=0.5)
    assert len(dets) == 1
    d = dets[0]
    assert d.label == "cup" and abs(d.score - 0.9) < 1e-6
    # Box in the 640 input is x 144..208, y 96..128; dividing by the 0.5 ratio maps it back.
    assert d.box == (288, 192, 128, 64)


def test_decode_nothing_above_threshold():
    raw = np.zeros((len(GRIDS), 85), np.float32)
    assert decode(raw, 1.0, 0.5, 0.5) == []
