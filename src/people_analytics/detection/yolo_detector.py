"""YOLOv8-based person detector backend using ultralytics."""

from pathlib import Path
from typing import Any, List, Optional, Union

from PIL import Image
from ultralytics import YOLO

from people_analytics.detection.base import BasePersonDetector
from people_analytics.detection.models import DetectionConfig, PersonDetection


class YOLOv8PersonDetector(BasePersonDetector):
    """Person detector powered by pretrained YOLOv8 models."""

    # COCO class index for 'person'
    PERSON_CLASS_ID = 0

    def __init__(self, config: Optional[DetectionConfig] = None, model: Optional[Any] = None):
        super().__init__(config)
        if model is not None:
            self._model = model
        else:
            self._model = YOLO(self.config.model_name)

    @property
    def model(self) -> Any:
        """Underlying YOLO model instance."""
        return self._model

    def detect_frame(
        self,
        image: Union[Path, str, Image.Image, Any],
        frame_index: int,
    ) -> List[PersonDetection]:
        """Run YOLO inference on a single frame and return person detections.

        Args:
            image: Image file path, PIL Image, or numpy array.
            frame_index: 1-based frame index to associate with detections.

        Returns:
            List of PersonDetection objects filtered for class 'person'.
        """
        # Run inference filtering directly for person class
        results = self._model(
            source=str(image) if isinstance(image, Path) else image,
            conf=self.config.confidence_threshold,
            classes=[self.PERSON_CLASS_ID],
            device=self.config.device,
            verbose=False,
        )

        detections: List[PersonDetection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        xyxy_coords = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), conf in zip(xyxy_coords, confidences):
            bb_left = float(x1)
            bb_top = float(y1)
            bb_width = float(x2 - x1)
            bb_height = float(y2 - y1)

            detections.append(
                PersonDetection(
                    frame_index=frame_index,
                    bb_left=bb_left,
                    bb_top=bb_top,
                    bb_width=bb_width,
                    bb_height=bb_height,
                    confidence=float(conf),
                    class_name="person",
                )
            )

        return detections
