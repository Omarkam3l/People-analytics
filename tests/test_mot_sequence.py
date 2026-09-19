"""Tests for MOT17 sequence parser and data models."""

from pathlib import Path
import pytest

from people_analytics.dataset.models import (
    Detection,
    GroundTruthAnnotation,
    MOT17Class,
    SequenceInfo,
)
from people_analytics.dataset.mot_sequence import MOT17Sequence


@pytest.fixture
def mock_sequence_dir(tmp_path: Path) -> Path:
    """Create a temporary mock MOT17 sequence directory."""
    seq_dir = tmp_path / "MOT17-99-MOCK"
    seq_dir.mkdir()

    # seqinfo.ini
    ini_content = """[Sequence]
name=MOT17-99-MOCK
imDir=img1
frameRate=25
seqLength=10
imWidth=1280
imHeight=720
imExt=.jpg
"""
    (seq_dir / "seqinfo.ini").write_text(ini_content, encoding="utf-8")

    # img1 directory with 2 mock image files
    img_dir = seq_dir / "img1"
    img_dir.mkdir()
    (img_dir / "000001.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    (img_dir / "000002.jpg").write_bytes(b"\xff\xd8\xff\xe0")

    # gt/gt.txt
    gt_dir = seq_dir / "gt"
    gt_dir.mkdir()
    # Format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    gt_lines = [
        "1,1,100,200,50,150,1,1,0.9",   # Active pedestrian
        "1,2,300,400,60,120,0,7,0.5",   # Static person (inactive)
        "2,1,105,202,50,150,1,1,0.85",  # Active pedestrian, next frame
        "2,3,500,100,200,100,0,8,0.2",  # Distractor
    ]
    (gt_dir / "gt.txt").write_text("\n".join(gt_lines), encoding="utf-8")

    # det/det.txt
    det_dir = seq_dir / "det"
    det_dir.mkdir()
    # Format: frame, id, bb_left, bb_top, bb_width, bb_height, conf
    det_lines = [
        "1,-1,98.5,198.0,52.0,151.0,0.95",
        "1,-1,50.0,50.0,30.0,40.0,0.30",
        "2,-1,104.0,201.0,51.0,150.5,0.92",
    ]
    (det_dir / "det.txt").write_text("\n".join(det_lines), encoding="utf-8")

    return seq_dir


# ====================================================================
# Unit Tests (using mock fixtures)
# ====================================================================

def test_sequence_info_parsing(mock_sequence_dir: Path):
    seq = MOT17Sequence(mock_sequence_dir)
    info = seq.info
    assert info.name == "MOT17-99-MOCK"
    assert info.frame_rate == 25.0
    assert info.seq_length == 10
    assert info.im_width == 1280
    assert info.im_height == 720
    assert info.resolution == (1280, 720)
    assert info.im_ext == ".jpg"
    assert info.im_dir == "img1"


def test_frame_paths_and_existence(mock_sequence_dir: Path):
    seq = MOT17Sequence(mock_sequence_dir)
    assert seq.has_images() is True
    assert seq.frame_exists(1) is True
    assert seq.frame_exists(2) is True
    assert seq.frame_exists(3) is False

    frame_path_1 = seq.get_frame_path(1)
    assert frame_path_1.name == "000001.jpg"

    with pytest.raises(ValueError, match="Frame index must be >= 1"):
        seq.get_frame_path(0)


def test_ground_truth_parsing(mock_sequence_dir: Path):
    seq = MOT17Sequence(mock_sequence_dir)
    all_gt = seq.load_ground_truth()
    assert len(all_gt) == 4

    # Check first annotation
    g0 = all_gt[0]
    assert g0.frame == 1
    assert g0.track_id == 1
    assert g0.bb_left == 100.0
    assert g0.bb_top == 200.0
    assert g0.bb_width == 50.0
    assert g0.bb_height == 150.0
    assert g0.conf == 1.0
    assert g0.class_id == MOT17Class.PEDESTRIAN
    assert g0.visibility == 0.9
    assert g0.is_pedestrian is True
    assert g0.is_active is True
    assert g0.class_name == "Pedestrian"

    assert g0.bbox_xywh == (100.0, 200.0, 50.0, 150.0)
    assert g0.bbox_xyxy == (100.0, 200.0, 150.0, 350.0)


def test_ground_truth_filtering(mock_sequence_dir: Path):
    seq = MOT17Sequence(mock_sequence_dir)

    # Filter pedestrians only
    peds = seq.load_ground_truth(pedestrians_only=True)
    assert len(peds) == 2
    assert all(p.class_id == MOT17Class.PEDESTRIAN for p in peds)

    # Filter active only
    active = seq.load_ground_truth(active_only=True)
    assert len(active) == 2
    assert all(a.conf == 1.0 for a in active)

    # Filter by visibility threshold
    high_vis = seq.load_ground_truth(min_visibility=0.88)
    assert len(high_vis) == 1
    assert high_vis[0].visibility == 0.9


def test_detection_parsing_and_filtering(mock_sequence_dir: Path):
    seq = MOT17Sequence(mock_sequence_dir)
    assert seq.has_detections() is True

    all_dets = seq.load_detections()
    assert len(all_dets) == 3

    # Filter by confidence threshold
    high_conf = seq.load_detections(min_conf=0.5)
    assert len(high_conf) == 2
    assert all(d.conf >= 0.5 for d in high_conf)

    d0 = all_dets[0]
    assert d0.frame == 1
    assert d0.track_id == -1
    assert d0.conf == 0.95
    assert d0.bbox_xywh == (98.5, 198.0, 52.0, 151.0)


def test_sequence_summary(mock_sequence_dir: Path):
    seq = MOT17Sequence(mock_sequence_dir)
    summary = seq.get_summary()

    assert summary["name"] == "MOT17-99-MOCK"
    assert summary["seq_length"] == 10
    assert summary["fps"] == 25.0
    assert summary["resolution"] == "1280x720"
    assert summary["has_images"] is True
    assert summary["images_on_disk"] == 2
    assert summary["has_gt"] is True
    assert summary["total_gt_annotations"] == 4
    assert summary["active_pedestrian_annotations"] == 2
    assert summary["unique_pedestrian_tracks"] == 1
    assert summary["has_det"] is True
    assert summary["total_detections"] == 3


def test_missing_files_error_handling(tmp_path: Path):
    empty_dir = tmp_path / "empty_seq"
    empty_dir.mkdir()

    seq = MOT17Sequence(empty_dir)
    with pytest.raises(FileNotFoundError, match="seqinfo.ini not found"):
        _ = seq.info

    with pytest.raises(FileNotFoundError, match="Ground truth not found"):
        seq.load_ground_truth()

    with pytest.raises(FileNotFoundError, match="Detections not found"):
        seq.load_detections()


def test_malformed_gt_line(tmp_path: Path):
    bad_dir = tmp_path / "bad_gt_seq"
    bad_dir.mkdir()
    (bad_dir / "seqinfo.ini").write_text("[Sequence]\nname=BAD\nseqLength=1\nframeRate=30\nimWidth=100\nimHeight=100\n", encoding="utf-8")
    gt_dir = bad_dir / "gt"
    gt_dir.mkdir()
    (gt_dir / "gt.txt").write_text("1,1,10,20\n", encoding="utf-8")  # only 4 columns

    seq = MOT17Sequence(bad_dir)
    with pytest.raises(ValueError, match="expected >= 9"):
        seq.load_ground_truth()


def test_mot17_class_names():
    assert MOT17Class.get_name(1) == "Pedestrian"
    assert MOT17Class.get_name(7) == "Static Person"
    assert MOT17Class.get_name(999) == "Unknown (999)"


def test_detection_properties():
    det = Detection(
        frame=1,
        track_id=-1,
        bb_left=50.0,
        bb_top=100.0,
        bb_width=40.0,
        bb_height=80.0,
        conf=0.9,
    )
    assert det.bbox_xywh == (50.0, 100.0, 40.0, 80.0)
    assert det.bbox_xyxy == (50.0, 100.0, 90.0, 180.0)


def test_sequence_dir_not_found(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="Sequence directory does not exist"):
        MOT17Sequence(non_existent)


def test_seqinfo_missing_section(tmp_path: Path):
    seq_dir = tmp_path / "bad_ini_seq"
    seq_dir.mkdir()
    (seq_dir / "seqinfo.ini").write_text("[InvalidSection]\nfoo=bar\n", encoding="utf-8")
    seq = MOT17Sequence(seq_dir)
    with pytest.raises(ValueError, match="Missing \\[Sequence\\] section"):
        _ = seq.info


def test_malformed_det_line(tmp_path: Path):
    bad_dir = tmp_path / "bad_det_seq"
    bad_dir.mkdir()
    (bad_dir / "seqinfo.ini").write_text(
        "[Sequence]\nname=BAD\nseqLength=1\nframeRate=30\nimWidth=100\nimHeight=100\n",
        encoding="utf-8",
    )
    det_dir = bad_dir / "det"
    det_dir.mkdir()
    (det_dir / "det.txt").write_text("1,-1,10,20\n", encoding="utf-8")  # only 4 columns

    seq = MOT17Sequence(bad_dir)
    with pytest.raises(ValueError, match="expected >= 7"):
        seq.load_detections()


# ====================================================================
# Integration Test on Local MOT17 Dataset (if present)
# ====================================================================

LOCAL_MOT17_09 = Path("MOT17/MOT17/train/MOT17-09-FRCNN")


@pytest.mark.skipif(not LOCAL_MOT17_09.exists(), reason="Local MOT17-09-FRCNN sequence not found")
def test_local_mot17_09_sequence():
    seq = MOT17Sequence(LOCAL_MOT17_09)
    assert seq.info.name == "MOT17-09-FRCNN"
    assert seq.info.seq_length == 525
    assert seq.info.frame_rate == 30.0
    assert seq.info.resolution == (1920, 1080)
    assert seq.has_images() is True
    assert seq.frame_exists(1) is True
    assert seq.frame_exists(525) is True

    # Check GT
    active_peds = seq.load_ground_truth(pedestrians_only=True, active_only=True)
    assert len(active_peds) == 5325
    unique_tracks = set(g.track_id for g in active_peds)
    assert len(unique_tracks) == 26

    # Check frame 1 sample
    frame_1_peds = [g for g in active_peds if g.frame == 1]
    assert len(frame_1_peds) > 0
    first = frame_1_peds[0]
    assert first.track_id == 1
    assert first.bbox_xywh == (260.0, 450.0, 102.0, 262.0)

