"""Data models for MOT17 dataset sequences and annotations."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple


class MOT17Class(IntEnum):
    """Standard MOT17 class label identifiers."""
    PEDESTRIAN = 1
    PERSON_ON_VEHICLE = 2
    CAR = 3
    BICYCLE = 4
    MOTORBIKE = 5
    NON_MOTORIZED_VEHICLE = 6
    STATIC_PERSON = 7
    DISTRACTOR = 8
    OCCLUDER = 9
    OCCLUDER_ON_GROUND = 10
    OCCLUDER_FULL = 11
    REFLECTION = 12

    @classmethod
    def get_name(cls, class_id: int) -> str:
        """Return human-readable name for class ID, or 'UNKNOWN' if not found."""
        try:
            return cls(class_id).name.replace("_", " ").title()
        except ValueError:
            return f"Unknown ({class_id})"


@dataclass(frozen=True)
class SequenceInfo:
    """Metadata extracted from seqinfo.ini."""
    name: str
    im_dir: str
    frame_rate: float
    seq_length: int
    im_width: int
    im_height: int
    im_ext: str

    @property
    def resolution(self) -> Tuple[int, int]:
        """Sequence resolution as (width, height)."""
        return self.im_width, self.im_height


@dataclass(frozen=True)
class GroundTruthAnnotation:
    """Ground truth annotation record from gt.txt."""
    frame: int
    track_id: int
    bb_left: float
    bb_top: float
    bb_width: float
    bb_height: float
    conf: float
    class_id: int
    visibility: float

    @property
    def class_name(self) -> str:
        """Human-readable class name."""
        return MOT17Class.get_name(self.class_id)

    @property
    def is_pedestrian(self) -> bool:
        """True if annotation is a pedestrian (class_id == 1)."""
        return self.class_id == MOT17Class.PEDESTRIAN

    @property
    def is_active(self) -> bool:
        """True if annotation is active / considered in evaluation (conf == 1.0)."""
        return self.conf == 1.0

    @property
    def bbox_xywh(self) -> Tuple[float, float, float, float]:
        """Bounding box in (left, top, width, height) format."""
        return self.bb_left, self.bb_top, self.bb_width, self.bb_height

    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        """Bounding box in (x1, y1, x2, y2) format."""
        return self.bb_left, self.bb_top, self.bb_left + self.bb_width, self.bb_top + self.bb_height


@dataclass(frozen=True)
class Detection:
    """Public detection record from det.txt."""
    frame: int
    track_id: int
    bb_left: float
    bb_top: float
    bb_width: float
    bb_height: float
    conf: float

    @property
    def bbox_xywh(self) -> Tuple[float, float, float, float]:
        """Bounding box in (left, top, width, height) format."""
        return self.bb_left, self.bb_top, self.bb_width, self.bb_height

    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        """Bounding box in (x1, y1, x2, y2) format."""
        return self.bb_left, self.bb_top, self.bb_left + self.bb_width, self.bb_top + self.bb_height

