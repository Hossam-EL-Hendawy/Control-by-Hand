#!/usr/bin/env python3
"""Live Vision AI: finger arithmetic and object detection from a webcam.

Run this program in a terminal.  Press Q in the camera window to stop it.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict, deque
from pathlib import Path
from statistics import median
from typing import NamedTuple

import cv2
import mediapipe as mp


OBJECT_CONFIDENCE = 0.45
OBJECT_DETECTION_EVERY_N_FRAMES = 8

class DetectedObject(NamedTuple):
    label: str
    confidence: float
    box: tuple[int, int, int, int]


class ObjectDetector:
    """Small local YOLO detector. It degrades safely when its model is unavailable."""

    def __init__(self, model_path: Path, enabled: bool) -> None:
        self.enabled = enabled
        self.error: str | None = None
        self.model = None
        if not enabled:
            return
        try:
            from ultralytics import YOLO
            # If the file is not present, Ultralytics downloads the small public
            # yolo11n model once, then subsequent starts are fully local.
            self.model = YOLO(str(model_path))
        except Exception as exc:
            self.error = str(exc)

    def detect(self, frame) -> list[DetectedObject]:
        if self.model is None:
            return []
        try:
            result = self.model(frame, verbose=False, conf=OBJECT_CONFIDENCE)[0]
            names = result.names
            found: list[DetectedObject] = []
            for box in result.boxes:
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                found.append(DetectedObject(str(names[class_id]), confidence, (x1, y1, x2, y2)))
            return found
        except Exception as exc:
            self.error = str(exc)
            return []


def distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def joint_angle(a, b, c) -> float:
    """Angle ABC in degrees; an extended finger is close to 180 degrees."""
    first = (a.x - b.x, a.y - b.y)
    second = (c.x - b.x, c.y - b.y)
    dot = first[0] * second[0] + first[1] * second[1]
    magnitude = math.hypot(*first) * math.hypot(*second)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / max(magnitude, 1e-6)))))


def count_fingers(hand, handedness: str) -> int:
    """Return an easy-to-read 0–5 finger count for one upright hand."""
    count = sum(
        hand.landmark[tip].y < hand.landmark[pip].y - 0.012
        and joint_angle(hand.landmark[mcp], hand.landmark[pip], hand.landmark[tip]) > 155
        for mcp, pip, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
    )
    thumb_tip = hand.landmark[4]
    thumb_joint = hand.landmark[3]
    thumb_points_out = (
        thumb_tip.x < thumb_joint.x if handedness == "Right"
        else thumb_tip.x > thumb_joint.x
    )
    thumb_is_open = (
        thumb_points_out
        and joint_angle(hand.landmark[2], thumb_joint, thumb_tip) > 150
        and distance(thumb_tip, hand.landmark[0]) > distance(thumb_joint, hand.landmark[0]) * 1.12
    )
    return count + int(thumb_is_open)


def draw_objects(image, objects: list[DetectedObject]) -> None:
    for item in objects:
        x1, y1, x2, y2 = item.box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 180, 0), 2)
        # OpenCV's built-in font is ASCII-only; use the reliable English COCO
        # label here (for example: cup, cell phone, chair, person).
        text = f"{item.label} {item.confidence:.0%}"
        cv2.putText(image, text, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 220, 120), 2, cv2.LINE_AA)


def eye_aspect_ratio(face, upper: int, lower: int, outer: int, inner: int) -> float:
    vertical = distance(face.landmark[upper], face.landmark[lower])
    horizontal = distance(face.landmark[outer], face.landmark[inner])
    return vertical / max(horizontal, 1e-6)


def eyes_closed(face) -> tuple[bool, bool]:
    # FaceMesh landmarks: left eye 159/145/33/133, right eye 386/374/362/263.
    left = eye_aspect_ratio(face, 159, 145, 33, 133) < 0.19
    right = eye_aspect_ratio(face, 386, 374, 362, 263) < 0.19
    return left, right


def draw_text(image, text: str, row: int, color=(255, 255, 255)) -> None:
    cv2.putText(image, text, (18, 34 + row * 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.66, color, 2, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Vision AI: hands and objects")
    parser.add_argument("--no-objects", action="store_true",
                        help="Start without the YOLO object detector.")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("yolo11n.pt"),
                        help="Path to a local YOLO model file (default: yolo11n.pt).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Camera not found. Check the camera permission and try again.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hands_api = mp.solutions.hands
    face_api = mp.solutions.face_mesh
    drawing = mp.solutions.drawing_utils
    hand_style = drawing.DrawingSpec(color=(0, 220, 120), thickness=2, circle_radius=3)
    detector = ObjectDetector(args.model, enabled=not args.no_objects)
    objects: list[DetectedObject] = []
    frame_number = 0
    # The median of recent frames prevents a one-frame landmark error from
    # making a raised index finger briefly display as two fingers.
    finger_history: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=5))

    with hands_api.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.70,
        min_tracking_confidence=0.65,
    ) as hands, face_api.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.60,
    ) as face_mesh:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_result = hands.process(rgb)
            face_result = face_mesh.process(rgb)
            frame_number += 1
            if detector.enabled and frame_number % OBJECT_DETECTION_EVERY_N_FRAMES == 0:
                objects = detector.detect(frame)

            face = face_result.multi_face_landmarks[0] if face_result.multi_face_landmarks else None
            left_eye_closed = right_eye_closed = False
            if face:
                left_eye_closed, right_eye_closed = eyes_closed(face)

            hand_pairs = []
            if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
                hand_pairs = list(zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness))
            finger_counts: list[int] = []
            for tracked_hand, classification in hand_pairs:
                handedness = classification.classification[0].label
                finger_history[handedness].append(count_fingers(tracked_hand, handedness))
                number = int(median(finger_history[handedness]))
                finger_counts.append(number)
                drawing.draw_landmarks(frame, tracked_hand, hands_api.HAND_CONNECTIONS, hand_style, hand_style)
                tip = tracked_hand.landmark[8]
                cv2.putText(frame, f"{handedness}: {number}",
                            (int(tip.x * frame.shape[1]), int(tip.y * frame.shape[0]) - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (70, 255, 170), 2, cv2.LINE_AA)

            draw_objects(frame, objects)
            draw_text(frame, "VISION AI: MOUSE CONTROL DISABLED", 0, (70, 255, 170))
            draw_text(frame, "Show fingers for counting | Show objects for labels | Q = quit", 1)
            eye_status = "not detected" if not face else ("closed" if left_eye_closed or right_eye_closed else "open")
            draw_text(frame, f"Eyes: {eye_status}", 2, (200, 230, 255))
            if len(finger_counts) == 2:
                draw_text(frame, f"HAND MATH: {finger_counts[0]} + {finger_counts[1]} = {sum(finger_counts)}", 3, (50, 255, 230))
            elif len(finger_counts) == 1:
                draw_text(frame, f"FINGERS: {finger_counts[0]}", 3, (50, 255, 230))
            if detector.error:
                draw_text(frame, "OBJECT AI unavailable: run pip install -r requirements.txt", 4, (90, 180, 255))
            elif detector.enabled:
                labels = ", ".join(item.label for item in objects[:3]) or "scanning..."
                draw_text(frame, f"OBJECT AI: {labels}", 4, (255, 220, 120))
            cv2.imshow("Control by Hand", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
