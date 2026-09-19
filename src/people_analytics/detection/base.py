"""Base interface for person detectors."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union

from PIL import Image

from people_analytics.dataset.mot_sequence import MOT17Sequence
from people_analytics.detection.models import DetectionConfig, PersonDetection


class BasePersonDetector(ABC):
    """Abstract base class for all person detector backends."""

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()

    @abstractmethod
    def detect_frame(
        self,
        image: Union[Path, str, Image.Image, Any],
        frame_index: int,
    ) -> List[PersonDetection]:
        """Detect people in a single frame.

        Args:
            image: Image file path, PIL Image, or numpy array.
            frame_index: 1-based frame index to associate with detections.

        Returns:
            List of PersonDetection instances matching class 'person' with confidence >= threshold.
        """
        pass

    def detect_sequence(
        self,
        sequence: MOT17Sequence,
        start_frame: int = 1,
        end_frame: Optional[int] = None,
        stride: int = 1,
    ) -> List[PersonDetection]:
        """Run person detection over an MOT17 sequence.

        Args:
            sequence: MOT17Sequence instance providing frame image paths.
            start_frame: First frame index to process (1-based, inclusive).
            end_frame: Last frame index to process (1-based, inclusive). Defaults to seq_length.
            stride: Process every N-th frame. Defaults to 1 (every frame).

        Returns:
            List of all PersonDetection instances across the processed frames.
        """
        if start_frame < 1:
            raise ValueError(f"start_frame must be >= 1, got {start_frame}")

        max_frame = sequence.info.seq_length
        if end_frame is None:
            end_frame = max_frame
        else:
            end_frame = min(end_frame, max_frame)

        if start_frame > end_frame:
            raise ValueError(f"start_frame ({start_frame}) must be <= end_frame ({end_frame})")

        all_detections: List[PersonDetection] = []
        for frame_idx in range(start_frame, end_frame + 1, stride):
            frame_path = sequence.get_frame_path(frame_idx)
            if not frame_path.is_file():
                continue
            frame_dets = self.detect_frame(frame_path, frame_index=frame_idx)
            all_detections.extend(frame_dets)

        return all_detections
