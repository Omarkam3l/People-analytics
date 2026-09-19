"""Parser and loader for an individual MOT17 sequence directory."""

import configparser
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from people_analytics.dataset.models import (
    Detection,
    GroundTruthAnnotation,
    MOT17Class,
    SequenceInfo,
)


class MOT17Sequence:
    """Represents a single MOT17 sequence directory."""

    def __init__(self, sequence_dir: Union[str, Path]):
        self.sequence_dir = Path(sequence_dir).resolve()
        if not self.sequence_dir.exists() or not self.sequence_dir.is_dir():
            raise FileNotFoundError(f"Sequence directory does not exist: {self.sequence_dir}")

        self._info: Optional[SequenceInfo] = None
        self._seqinfo_path = self.sequence_dir / "seqinfo.ini"

    @property
    def seqinfo_path(self) -> Path:
        """Path to seqinfo.ini."""
        return self._seqinfo_path

    @property
    def gt_path(self) -> Path:
        """Path to gt/gt.txt."""
        return self.sequence_dir / "gt" / "gt.txt"

    @property
    def det_path(self) -> Path:
        """Path to det/det.txt."""
        return self.sequence_dir / "det" / "det.txt"

    @property
    def img_dir_path(self) -> Path:
        """Path to image directory (e.g. img1)."""
        info = self.info
        return self.sequence_dir / info.im_dir

    @property
    def info(self) -> SequenceInfo:
        """Sequence metadata parsed from seqinfo.ini (cached)."""
        if self._info is None:
            self._info = self.load_seqinfo()
        return self._info

    def load_seqinfo(self) -> SequenceInfo:
        """Parse and return metadata from seqinfo.ini."""
        if not self._seqinfo_path.exists():
            raise FileNotFoundError(f"seqinfo.ini not found in {self.sequence_dir}")

        parser = configparser.ConfigParser()
        parser.read(self._seqinfo_path, encoding="utf-8")

        if "Sequence" not in parser:
            raise ValueError(f"Missing [Sequence] section in {self._seqinfo_path}")

        sec = parser["Sequence"]
        try:
            info = SequenceInfo(
                name=sec["name"],
                im_dir=sec.get("imDir", "img1"),
                frame_rate=float(sec["frameRate"]),
                seq_length=int(sec["seqLength"]),
                im_width=int(sec["imWidth"]),
                im_height=int(sec["imHeight"]),
                im_ext=sec.get("imExt", ".jpg"),
            )
        except KeyError as err:
            raise ValueError(f"Missing required field in seqinfo.ini: {err}") from err
        except ValueError as err:
            raise ValueError(f"Malformed value in seqinfo.ini: {err}") from err

        return info

    def get_frame_path(self, frame_index: int) -> Path:
        """Return expected filesystem path for a 1-based frame index."""
        if frame_index < 1:
            raise ValueError(f"Frame index must be >= 1 (1-based), got {frame_index}")
        info = self.info
        filename = f"{frame_index:06d}{info.im_ext}"
        return self.img_dir_path / filename

    def frame_exists(self, frame_index: int) -> bool:
        """Check whether the image file for a given frame index exists on disk."""
        return self.get_frame_path(frame_index).is_file()

    def has_images(self) -> bool:
        """Check whether the image directory exists and contains images."""
        img_dir = self.img_dir_path
        if not img_dir.is_dir():
            return False
        return any(img_dir.glob(f"*{self.info.im_ext}"))

    def has_ground_truth(self) -> bool:
        """Check whether ground truth gt.txt exists."""
        return self.gt_path.is_file()

    def has_detections(self) -> bool:
        """Check whether det.txt exists."""
        return self.det_path.is_file()

    def load_ground_truth(
        self,
        pedestrians_only: bool = False,
        active_only: bool = False,
        min_visibility: float = 0.0,
    ) -> List[GroundTruthAnnotation]:
        """Load ground truth annotations from gt/gt.txt with optional filters.

        Args:
            pedestrians_only: If True, keep only class_id == 1 (Pedestrian).
            active_only: If True, keep only conf == 1.0 (considered active in MOT benchmark).
            min_visibility: Minimum visibility ratio [0.0, 1.0].

        Returns:
            List of GroundTruthAnnotation objects.
        """
        if not self.has_ground_truth():
            raise FileNotFoundError(f"Ground truth not found at {self.gt_path}")

        annotations: List[GroundTruthAnnotation] = []
        with open(self.gt_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for line_no, row in enumerate(reader, start=1):
                if not row or not any(row):
                    continue
                if len(row) < 9:
                    raise ValueError(
                        f"Line {line_no} in {self.gt_path} has {len(row)} columns, expected >= 9"
                    )

                frame = int(row[0])
                track_id = int(row[1])
                bb_left = float(row[2])
                bb_top = float(row[3])
                bb_width = float(row[4])
                bb_height = float(row[5])
                conf = float(row[6])
                class_id = int(float(row[7]))
                visibility = float(row[8])

                if pedestrians_only and class_id != MOT17Class.PEDESTRIAN:
                    continue
                if active_only and conf != 1.0:
                    continue
                if visibility < min_visibility:
                    continue

                annotations.append(
                    GroundTruthAnnotation(
                        frame=frame,
                        track_id=track_id,
                        bb_left=bb_left,
                        bb_top=bb_top,
                        bb_width=bb_width,
                        bb_height=bb_height,
                        conf=conf,
                        class_id=class_id,
                        visibility=visibility,
                    )
                )

        return annotations

    def load_detections(self, min_conf: float = 0.0) -> List[Detection]:
        """Load public detections from det/det.txt with optional confidence filtering.

        Args:
            min_conf: Minimum confidence score threshold.

        Returns:
            List of Detection objects.
        """
        if not self.has_detections():
            raise FileNotFoundError(f"Detections not found at {self.det_path}")

        detections: List[Detection] = []
        with open(self.det_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for line_no, row in enumerate(reader, start=1):
                if not row or not any(row):
                    continue
                if len(row) < 7:
                    raise ValueError(
                        f"Line {line_no} in {self.det_path} has {len(row)} columns, expected >= 7"
                    )

                frame = int(row[0])
                track_id = int(row[1])
                bb_left = float(row[2])
                bb_top = float(row[3])
                bb_width = float(row[4])
                bb_height = float(row[5])
                conf = float(row[6])

                if conf < min_conf:
                    continue

                detections.append(
                    Detection(
                        frame=frame,
                        track_id=track_id,
                        bb_left=bb_left,
                        bb_top=bb_top,
                        bb_width=bb_width,
                        bb_height=bb_height,
                        conf=conf,
                    )
                )

        return detections

    def get_summary(self) -> Dict[str, Any]:
        """Produce an inspection summary dictionary for the sequence."""
        info = self.info
        summary: Dict[str, Any] = {
            "name": info.name,
            "seq_length": info.seq_length,
            "fps": info.frame_rate,
            "resolution": f"{info.im_width}x{info.im_height}",
            "im_ext": info.im_ext,
            "has_images": self.has_images(),
            "has_gt": self.has_ground_truth(),
            "has_det": self.has_detections(),
        }

        if summary["has_images"]:
            summary["images_on_disk"] = len(list(self.img_dir_path.glob(f"*{info.im_ext}")))

        if summary["has_gt"]:
            all_gt = self.load_ground_truth()
            active_peds = [g for g in all_gt if g.is_pedestrian and g.is_active]
            unique_tracks = len(set(g.track_id for g in active_peds))
            summary["total_gt_annotations"] = len(all_gt)
            summary["active_pedestrian_annotations"] = len(active_peds)
            summary["unique_pedestrian_tracks"] = unique_tracks

        if summary["has_det"]:
            all_det = self.load_detections()
            summary["total_detections"] = len(all_det)

        return summary
