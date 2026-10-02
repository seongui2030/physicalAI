# make_teacher_templates_current.py
# ============================================================
# 교사 시범 영상 -> Ground Truth(JSON) 생성기 V2
#
# 목적
#   1) 원본 영상의 "모든 프레임"을 JSON에 보존
#   2) MediaPipe Pose + (축구) YOLO 공 검출
#   3) 검출 실패 프레임도 삭제하지 않고 pose_detected=False 기록
#   4) 실제 검출률 / 전체 프레임 커버리지 계산
#   5) DTW 학생평가에서 사용할 시계열 특징을 동일한 규칙으로 저장
#
# 현재 프로젝트 구조
# D:\physicalAI
# ├─ app.py
# ├─ pose_landmarker.task
# ├─ yolov8n.pt
# └─ videos
#     └─ teacher
#         ├─ rope
#         ├─ running
#         └─ soccer
#
# 실행
#   python make_teacher_templates_current.py
#   python make_teacher_templates_current.py --sport soccer
#   python make_teacher_templates_current.py --sport soccer --skill 1
#
# 결과
# D:\physicalAI\videos\teacher_templates
#
# 중요
#   - mp.solutions.pose는 사용하지 않습니다.
#   - MediaPipe Tasks API를 사용합니다.
#   - sequence 길이는 "실제로 읽은 전체 프레임 수"와 같아집니다.
#   - Pose 검출 실패 프레임도 sequence에서 삭제하지 않습니다.
# ============================================================

import argparse
import json
import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO


# ------------------------------------------------------------
# 1. 프로젝트 경로
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

VIDEO_ROOT = PROJECT_ROOT / "videos" / "teacher"
OUTPUT_ROOT = PROJECT_ROOT / "videos" / "teacher_templates"

POSE_MODEL_PATH = PROJECT_ROOT / "pose_landmarker.task"
YOLO_MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"


# ------------------------------------------------------------
# 2. 종목 이름
# ------------------------------------------------------------
SPORT_DIRS = {
    "rope": "줄넘기",
    "soccer": "축구",
    "running": "달리기",
}


# ------------------------------------------------------------
# 3. 현재 실제 파일 이름에 맞춘 동작 이름
# ------------------------------------------------------------
ROPE_SKILLS = {
    1: "번갈아 뛰기",
    2: "번갈아 2박자 뛰기",
    3: "옆흔들어 뛰기",
    4: "X자 뛰기",
    5: "가위바위보 뛰기",
    6: "앞흔들어 뛰기",
    7: "뒤흔들어 뛰기",
    8: "8자 돌리기",
    9: "되돌리기",
    10: "이단뛰기",
}

SOCCER_SKILLS = {
    1: "Inside Touch",
    2: "Ball Shift",
    3: "Sole Roll",
    4: "Roll Stop",
    5: "Outside Inside",
    6: "One Foot In-Out",
    7: "180 Turn",
    8: "V Cut",
    9: "Cruyff Turn",
    10: "Shot Fake",
}

RUNNING_SKILLS = {
    1: "Bounding",
    2: "C-Skip",
    3: "Straight-leg Extension",
    4: "Ankling",
    5: "1-2-Switch",
    6: "One-Two-Switch",
}

SKILLS = {
    "rope": ROPE_SKILLS,
    "soccer": SOCCER_SKILLS,
    "running": RUNNING_SKILLS,
}


# ------------------------------------------------------------
# 4. MediaPipe Pose landmark 번호
# ------------------------------------------------------------
NOSE = 0

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28


# ------------------------------------------------------------
# 5. 분석 설정
# ------------------------------------------------------------
POSE_DETECTION_THRESHOLD = 0.50
POSE_PRESENCE_THRESHOLD = 0.50
POSE_TRACKING_THRESHOLD = 0.50

YOLO_CONFIDENCE = 0.25

MIN_POSE_RATE = 0.80
MIN_COVERAGE_RATE = 0.999


# ------------------------------------------------------------
# 6. 파일명에서 동작 번호 찾기
# ------------------------------------------------------------
def get_skill_number(path):
    match = re.search(r"_(\d{2})_", path.stem)

    if match:
        return int(match.group(1))

    match = re.search(r"(\d{2})", path.stem)

    if match:
        return int(match.group(1))

    return None


# ------------------------------------------------------------
# 7. 한 점 가져오기
# ------------------------------------------------------------
def get_point(landmarks, index):
    landmark = landmarks[index]

    return np.array(
        [
            float(landmark.x),
            float(landmark.y),
        ],
        dtype=float,
    )


# ------------------------------------------------------------
# 8. 두 점 사이 거리
# ------------------------------------------------------------
def distance(a, b):
    return float(np.linalg.norm(a - b))


# ------------------------------------------------------------
# 9. 세 점으로 관절각 계산
# ------------------------------------------------------------
def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return None

    value = np.dot(ba, bc) / denominator
    value = np.clip(value, -1.0, 1.0)

    return float(np.degrees(np.arccos(value)))


# ------------------------------------------------------------
# 10. 선분의 기울기 각도
# ------------------------------------------------------------
def calculate_line_angle(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None

    return float(np.degrees(np.arctan2(dy, dx)))


# ------------------------------------------------------------
# 11. MediaPipe visibility 평균
# ------------------------------------------------------------
def calculate_visibility(landmarks):
    values = []

    for landmark in landmarks:
        visibility = getattr(landmark, "visibility", 1.0)

        try:
            values.append(float(visibility))
        except (TypeError, ValueError):
            values.append(1.0)

    if not values:
        return 0.0

    return float(np.mean(values))


# ------------------------------------------------------------
# 12. 프레임 하나에서 자세 특징 추출
#
# 주의:
# gaze_proxy는 실제 눈동자 시선 검출이 아닙니다.
# 코(nose)와 어깨 중심의 상대적 위치를 이용한 단순 proxy입니다.
# ------------------------------------------------------------
def extract_pose_features(landmarks, ball_point=None):
    left_shoulder = get_point(landmarks, LEFT_SHOULDER)
    right_shoulder = get_point(landmarks, RIGHT_SHOULDER)

    left_hip = get_point(landmarks, LEFT_HIP)
    right_hip = get_point(landmarks, RIGHT_HIP)

    left_knee = get_point(landmarks, LEFT_KNEE)
    right_knee = get_point(landmarks, RIGHT_KNEE)

    left_ankle = get_point(landmarks, LEFT_ANKLE)
    right_ankle = get_point(landmarks, RIGHT_ANKLE)

    nose = get_point(landmarks, NOSE)

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0

    shoulder_width = distance(
        left_shoulder,
        right_shoulder,
    )

    hip_width = distance(
        left_hip,
        right_hip,
    )

    body_scale = max(
        distance(shoulder_center, hip_center),
        1e-6,
    )

    # 몸 중심 기준 상대 좌표
    left_ankle_rel = (
        left_ankle - hip_center
    ) / body_scale

    right_ankle_rel = (
        right_ankle - hip_center
    ) / body_scale

    nose_rel = (
        nose - shoulder_center
    ) / body_scale

    features = {
        # 원래 사용하던 기본 특징
        "left_knee_angle": calculate_angle(
            left_hip,
            left_knee,
            left_ankle,
        ),
        "right_knee_angle": calculate_angle(
            right_hip,
            right_knee,
            right_ankle,
        ),

        "left_ankle_x": float(left_ankle[0]),
        "left_ankle_y": float(left_ankle[1]),

        "right_ankle_x": float(right_ankle[0]),
        "right_ankle_y": float(right_ankle[1]),

        "hip_x": float(hip_center[0]),
        "hip_y": float(hip_center[1]),

        "shoulder_x": float(shoulder_center[0]),
        "shoulder_y": float(shoulder_center[1]),

        "nose_x": float(nose[0]),
        "nose_y": float(nose[1]),

        # 단순 시선 proxy
        "gaze_proxy": abs(
            float(nose[0] - shoulder_center[0])
        ),

        "visibility": calculate_visibility(landmarks),

        # 추가 특징: 자세/균형 분석에 유용
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "body_scale": body_scale,

        "shoulder_angle": calculate_line_angle(
            left_shoulder,
            right_shoulder,
        ),

        "hip_angle": calculate_line_angle(
            left_hip,
            right_hip,
        ),

        # DTW에서 카메라 위치 변화에 조금 더 강한 상대 좌표
        "left_ankle_rel_x": float(left_ankle_rel[0]),
        "left_ankle_rel_y": float(left_ankle_rel[1]),

        "right_ankle_rel_x": float(right_ankle_rel[0]),
        "right_ankle_rel_y": float(right_ankle_rel[1]),

        "nose_rel_x": float(nose_rel[0]),
        "nose_rel_y": float(nose_rel[1]),
    }

    if ball_point is not None:
        features["ball_x"] = float(ball_point[0])
        features["ball_y"] = float(ball_point[1])

        # 공도 골반 중심 기준 상대 좌표 저장
        ball_rel = (
            np.array(ball_point) - hip_center
        ) / body_scale

        features["ball_rel_x"] = float(ball_rel[0])
        features["ball_rel_y"] = float(ball_rel[1])

    return features


# ------------------------------------------------------------
# 13. MediaPipe PoseLandmarker 생성
# ------------------------------------------------------------
def create_pose_detector():
    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "\nMediaPipe 모델이 없습니다.\n"
            f"필요한 파일: {POSE_MODEL_PATH}\n"
            "현재 app.py에서 사용하는 "
            "pose_landmarker.task를 프로젝트 폴더에 넣어 주세요."
        )

    base_options = python.BaseOptions(
        model_asset_path=str(POSE_MODEL_PATH)
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=POSE_DETECTION_THRESHOLD,
        min_pose_presence_confidence=POSE_PRESENCE_THRESHOLD,
        min_tracking_confidence=POSE_TRACKING_THRESHOLD,
        output_segmentation_masks=False,
    )

    return vision.PoseLandmarker.create_from_options(options)


# ------------------------------------------------------------
# 14. YOLO 생성
# ------------------------------------------------------------
def create_yolo_model():
    if not YOLO_MODEL_PATH.exists():
        print(
            "[주의] yolov8n.pt가 없습니다. "
            "축구공 검출 없이 자세 분석만 진행합니다."
        )
        return None

    try:
        model = YOLO(str(YOLO_MODEL_PATH))

        print(
            "[정보] YOLO 모델: yolov8n.pt"
        )
        print(
            "[정보] 축구공 클래스: sports ball"
        )
        print(
            f"[정보] YOLO confidence: {YOLO_CONFIDENCE}"
        )

        return model

    except Exception as error:
        print(
            f"[주의] YOLO 모델 로딩 실패: {error}"
        )
        return None


# ------------------------------------------------------------
# 15. 축구공 중심점 찾기
#
# 기존 코드의 문제:
#   sports ball을 찾으면 첫 번째 box를 바로 반환
#
# 개선:
#   여러 sports ball이 검출되면 confidence가 가장 높은
#   box 하나를 선택합니다.
# ------------------------------------------------------------
def detect_ball(yolo_model, frame):
    if yolo_model is None:
        return None

    best_box = None
    best_confidence = 0.0

    try:
        results = yolo_model(
            frame,
            verbose=False,
            conf=YOLO_CONFIDENCE,
        )

        for result in results:
            names = result.names

            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(
                    box.cls[0].item()
                )

                class_name = names[class_id]

                if class_name != "sports ball":
                    continue

                confidence = float(
                    box.conf[0].item()
                )

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = box

        if best_box is None:
            return None

        x1, y1, x2, y2 = map(
            float,
            best_box.xyxy[0].tolist(),
        )

        height, width = frame.shape[:2]

        center_x = (
            (x1 + x2) / 2.0
        ) / width

        center_y = (
            (y1 + y2) / 2.0
        ) / height

        return (
            float(center_x),
            float(center_y),
        )

    except Exception:
        return None


# ------------------------------------------------------------
# 16. 통계
# ------------------------------------------------------------
def calculate_stats(sequence, key):
    data = []

    for item in sequence:
        value = item.get(key)

        if value is None:
            continue

        try:
            value = float(value)

            if np.isfinite(value):
                data.append(value)
        except (TypeError, ValueError):
            continue

    if not data:
        return None

    return {
        "count": len(data),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


# ------------------------------------------------------------
# 17. 이동량 통계
# ------------------------------------------------------------
def calculate_movement(sequence, x_key, y_key):
    movements = []

    for before, after in zip(
        sequence,
        sequence[1:],
    ):
        x1 = before.get(x_key)
        y1 = before.get(y_key)
        x2 = after.get(x_key)
        y2 = after.get(y_key)

        if None in (x1, y1, x2, y2):
            continue

        dx = float(x2) - float(x1)
        dy = float(y2) - float(y1)

        movements.append(
            float(np.hypot(dx, dy))
        )

    if not movements:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "total": 0.0,
            "max": 0.0,
        }

    return {
        "count": len(movements),
        "mean": float(np.mean(movements)),
        "std": float(np.std(movements)),
        "total": float(np.sum(movements)),
        "max": float(np.max(movements)),
    }


# ------------------------------------------------------------
# 18. 프레임 간 속도/가속도 통계
#
# 단위는 영상 좌표 기준입니다.
# later evaluator에서 학생과 동일한 방식으로 계산합니다.
# ------------------------------------------------------------
def calculate_motion_stats(
    sequence,
    x_key,
    y_key,
    fps,
):
    speeds = []
    accelerations = []

    previous_speed = None

    for before, after in zip(
        sequence,
        sequence[1:],
    ):
        x1 = before.get(x_key)
        y1 = before.get(y_key)
        x2 = after.get(x_key)
        y2 = after.get(y_key)

        if None in (x1, y1, x2, y2):
            previous_speed = None
            continue

        distance_value = float(
            np.hypot(
                float(x2) - float(x1),
                float(y2) - float(y1),
            )
        )

        speed = distance_value * fps
        speeds.append(speed)

        if previous_speed is not None:
            acceleration = (
                speed - previous_speed
            ) * fps
            accelerations.append(
                float(acceleration)
            )

        previous_speed = speed

    def stats(values):
        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "max_abs": 0.0,
            }

        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "max_abs": float(
                np.max(np.abs(values))
            ),
        }

    return {
        "speed": stats(speeds),
        "acceleration": stats(accelerations),
    }


# ------------------------------------------------------------
# 19. 전체 영상의 모든 프레임을 분석
# ------------------------------------------------------------
def create_template(
    video_path,
    sport,
    pose_detector,
    yolo_model,
):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(
            f"[실패] 영상을 열 수 없습니다: "
            f"{video_path}"
        )
        return None

    reported_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    fps = (
        reported_fps
        if reported_fps > 0
        else 30.0
    )

    reported_frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    reported_duration = (
        reported_frame_count / fps
        if reported_frame_count > 0
        else 0.0
    )

    # --------------------------------------------------------
    # 중요:
    # sequence에는 Pose 검출 실패 프레임도 넣습니다.
    # 따라서 sequence 길이 = 실제 읽은 프레임 수
    # --------------------------------------------------------
    sequence = []

    pose_detected_frames = 0
    ball_detected_frames = 0

    actual_frame_count = 0

    print(
        f"\n분석: {video_path.name}"
    )
    print(
        f"  종목: {SPORT_DIRS[sport]}"
    )
    print(
        f"  FPS: {fps:.2f}"
    )
    print(
        f"  OpenCV frame_count: "
        f"{reported_frame_count}"
    )
    print(
        f"  예상 길이: "
        f"{reported_duration:.2f}초"
    )

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_index = actual_frame_count

        # --------------------------------------------
        # OpenCV BGR -> RGB
        # --------------------------------------------
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        detection_result = (
            pose_detector.detect(mp_image)
        )

        # --------------------------------------------
        # 축구공
        # --------------------------------------------
        ball_point = None

        if sport == "soccer":
            ball_point = detect_ball(
                yolo_model,
                frame,
            )

            if ball_point is not None:
                ball_detected_frames += 1

        # --------------------------------------------
        # 기본 프레임 정보
        # --------------------------------------------
        frame_data = {
            "frame": frame_index,
            "time_sec": round(
                frame_index / fps,
                4,
            ),
            "pose_detected": False,
            "ball_detected": (
                ball_point is not None
            )
            if sport == "soccer"
            else None,
        }

        # --------------------------------------------
        # Pose 검출 성공
        # --------------------------------------------
        if detection_result.pose_landmarks:
            landmarks = (
                detection_result.pose_landmarks[0]
            )

            features = extract_pose_features(
                landmarks,
                ball_point=ball_point,
            )

            frame_data.update(features)
            frame_data["pose_detected"] = True

            pose_detected_frames += 1

        sequence.append(frame_data)

        actual_frame_count += 1

    cap.release()

    # --------------------------------------------------------
    # 실제 영상 기준 품질 계산
    # --------------------------------------------------------
    sequence_count = len(sequence)

    pose_rate = (
        pose_detected_frames
        / sequence_count
        if sequence_count > 0
        else 0.0
    )

    ball_rate = (
        ball_detected_frames
        / sequence_count
        if sequence_count > 0
        else 0.0
    )

    # 실제로 분석된 frame과 sequence 길이가 같아야 합니다.
    sequence_coverage = (
        sequence_count
        / reported_frame_count
        if reported_frame_count > 0
        else 0.0
    )

    frame_count_difference = (
        sequence_count
        - reported_frame_count
    )

    # Pose가 있는 프레임만 뽑은 시계열.
    # DTW의 1차 입력으로 사용할 수 있습니다.
    valid_pose_sequence = [
        item
        for item in sequence
        if item["pose_detected"]
    ]

    valid_ball_sequence = [
        item
        for item in sequence
        if item.get("ball_detected") is True
    ]

    # --------------------------------------------------------
    # 통계는 Pose 검출 프레임만 대상으로 계산
    # --------------------------------------------------------
    summary = {
        "left_knee_angle": calculate_stats(
            valid_pose_sequence,
            "left_knee_angle",
        ),
        "right_knee_angle": calculate_stats(
            valid_pose_sequence,
            "right_knee_angle",
        ),

        "hip_x": calculate_stats(
            valid_pose_sequence,
            "hip_x",
        ),
        "hip_y": calculate_stats(
            valid_pose_sequence,
            "hip_y",
        ),

        "left_ankle_x": calculate_stats(
            valid_pose_sequence,
            "left_ankle_x",
        ),
        "right_ankle_x": calculate_stats(
            valid_pose_sequence,
            "right_ankle_x",
        ),

        "gaze_proxy": calculate_stats(
            valid_pose_sequence,
            "gaze_proxy",
        ),

        "visibility": calculate_stats(
            valid_pose_sequence,
            "visibility",
        ),

        "hip_movement": calculate_movement(
            valid_pose_sequence,
            "hip_x",
            "hip_y",
        ),

        "left_ankle_movement": calculate_movement(
            valid_pose_sequence,
            "left_ankle_x",
            "left_ankle_y",
        ),

        "right_ankle_movement": calculate_movement(
            valid_pose_sequence,
            "right_ankle_x",
            "right_ankle_y",
        ),

        "hip_motion": calculate_motion_stats(
            valid_pose_sequence,
            "hip_x",
            "hip_y",
            fps,
        ),

        "left_ankle_motion": calculate_motion_stats(
            valid_pose_sequence,
            "left_ankle_x",
            "left_ankle_y",
            fps,
        ),

        "right_ankle_motion": calculate_motion_stats(
            valid_pose_sequence,
            "right_ankle_x",
            "right_ankle_y",
            fps,
        ),
    }

    if sport == "soccer":
        summary["ball_x"] = calculate_stats(
            valid_ball_sequence,
            "ball_x",
        )

        summary["ball_y"] = calculate_stats(
            valid_ball_sequence,
            "ball_y",
        )

        summary["ball_relative_x"] = (
            calculate_stats(
                valid_ball_sequence,
                "ball_rel_x",
            )
        )

        summary["ball_relative_y"] = (
            calculate_stats(
                valid_ball_sequence,
                "ball_rel_y",
            )
        )

        summary["ball_motion"] = (
            calculate_motion_stats(
                valid_ball_sequence,
                "ball_x",
                "ball_y",
                fps,
            )
        )

    # --------------------------------------------------------
    # 검토 사유
    # --------------------------------------------------------
    review_reasons = []

    if sequence_count == 0:
        review_reasons.append(
            "프레임을 하나도 읽지 못했습니다."
        )

    if (
        reported_frame_count > 0
        and sequence_coverage < MIN_COVERAGE_RATE
    ):
        review_reasons.append(
            "실제 읽은 프레임 수와 영상의 "
            "frame_count가 일치하지 않습니다."
        )

    if pose_rate < MIN_POSE_RATE:
        review_reasons.append(
            "Pose 검출률이 80% 미만입니다."
        )

    if (
        sport == "soccer"
        and yolo_model is None
    ):
        review_reasons.append(
            "YOLO 모델이 없어 공 검출을 수행하지 못했습니다."
        )

    if (
        sport == "soccer"
        and yolo_model is not None
        and ball_rate < 0.20
    ):
        review_reasons.append(
            "축구공 검출률이 20% 미만입니다."
        )

    skill_no = get_skill_number(video_path)

    skill_name = SKILLS[sport].get(
        skill_no,
        video_path.stem,
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------
    result = {
        "template_version": "2.0",

        "sport_code": sport,
        "sport": SPORT_DIRS[sport],

        "skill_number": skill_no,
        "skill": skill_name,

        "source_video": video_path.name,

        "video": {
            "fps": float(fps),

            # OpenCV가 읽은 실제 프레임 수
            "frame_count": reported_frame_count,
            "actual_frame_count": actual_frame_count,

            "sequence_frame_count": sequence_count,

            "duration_sec": float(
                reported_duration
            ),

            "actual_duration_sec": (
                float(
                    sequence_count / fps
                )
                if fps > 0
                else 0.0
            ),
        },

        "coverage": {
            "sequence_coverage_rate": round(
                sequence_coverage,
                4,
            ),

            "frame_count_difference": (
                frame_count_difference
            ),
        },

        "quality": {
            # 전체 영상 프레임 중 Pose가 검출된 비율
            "pose_detection_rate": round(
                pose_rate,
                4,
            ),

            # 전체 영상 프레임 중 공이 검출된 비율
            "ball_detection_rate": (
                round(ball_rate, 4)
                if sport == "soccer"
                else None
            ),

            "mean_visibility": (
                summary["visibility"]["mean"]
                if summary["visibility"]
                else 0.0
            ),

            "pose_valid_frame_count": (
                pose_detected_frames
            ),

            "ball_valid_frame_count": (
                ball_detected_frames
                if sport == "soccer"
                else None
            ),

            "review_required": bool(
                review_reasons
            ),

            "review_reasons": review_reasons,
        },

        "analysis_config": {
            "pose_model": POSE_MODEL_PATH.name,
            "pose_detection_confidence": (
                POSE_DETECTION_THRESHOLD
            ),
            "pose_presence_confidence": (
                POSE_PRESENCE_THRESHOLD
            ),
            "pose_tracking_confidence": (
                POSE_TRACKING_THRESHOLD
            ),

            "yolo_model": (
                YOLO_MODEL_PATH.name
                if sport == "soccer"
                else None
            ),

            "yolo_class": (
                "sports ball"
                if sport == "soccer"
                else None
            ),

            "yolo_confidence": (
                YOLO_CONFIDENCE
                if sport == "soccer"
                else None
            ),
        },

        "feature_schema": {
            "pose": [
                "left_knee_angle",
                "right_knee_angle",
                "left_ankle_x",
                "left_ankle_y",
                "right_ankle_x",
                "right_ankle_y",
                "hip_x",
                "hip_y",
                "shoulder_x",
                "shoulder_y",
                "nose_x",
                "nose_y",
                "gaze_proxy",
                "visibility",
                "shoulder_width",
                "hip_width",
                "body_scale",
                "shoulder_angle",
                "hip_angle",
                "left_ankle_rel_x",
                "left_ankle_rel_y",
                "right_ankle_rel_x",
                "right_ankle_rel_y",
                "nose_rel_x",
                "nose_rel_y",
            ],
            "soccer_ball": [
                "ball_x",
                "ball_y",
                "ball_rel_x",
                "ball_rel_y",
            ],
        },

        "summary": summary,

        # 모든 프레임을 보존합니다.
        # pose_detected=False인 프레임도 삭제하지 않습니다.
        "sequence": sequence,

        # DTW에 바로 사용하기 위한 Pose 유효 프레임
        "dtw_sequence": valid_pose_sequence,
    }

    return result


# ------------------------------------------------------------
# 20. JSON 저장
# ------------------------------------------------------------
def save_template(
    result,
    sport,
    video_path,
):
    output_dir = (
        OUTPUT_ROOT / sport
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    skill_no = result["skill_number"]

    output_name = (
        f"{sport}_{skill_no:02d}_"
        f"{video_path.stem}.json"
    )

    output_path = (
        output_dir / output_name
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    return output_path


# ------------------------------------------------------------
# 21. 대상 영상 찾기
# ------------------------------------------------------------
def find_videos(sport):
    sport_dir = (
        VIDEO_ROOT / sport
    )

    if not sport_dir.exists():
        print(
            f"[오류] 폴더가 없습니다: "
            f"{sport_dir}"
        )
        return []

    videos = []

    for extension in (
        "*.mp4",
        "*.MP4",
        "*.mov",
        "*.MOV",
        "*.avi",
        "*.AVI",
    ):
        videos.extend(
            sport_dir.glob(extension)
        )

    return sorted(
        videos,
        key=lambda path: (
            get_skill_number(path)
            if get_skill_number(path)
            is not None
            else 999,
            path.name,
        ),
    )


# ------------------------------------------------------------
# 22. 실행
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "교사 시범 영상에서 "
            "Ground Truth JSON V2를 생성합니다."
        )
    )

    parser.add_argument(
        "--sport",
        choices=[
            "rope",
            "soccer",
            "running",
        ],
        help="rope / soccer / running",
    )

    parser.add_argument(
        "--skill",
        type=int,
        help="예: --skill 1 또는 --skill 01",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(
        "교사 시범 영상 Ground Truth 생성기 V2"
    )
    print("=" * 70)

    print(
        f"입력 폴더 : {VIDEO_ROOT}"
    )
    print(
        f"출력 폴더 : {OUTPUT_ROOT}"
    )

    # --------------------------------------------------------
    # 모델 준비
    # --------------------------------------------------------
    pose_detector = create_pose_detector()

    yolo_model = None

    if args.sport in (
        None,
        "soccer",
    ):
        yolo_model = create_yolo_model()

    success_count = 0
    review_count = 0
    fail_count = 0

    sports = (
        [args.sport]
        if args.sport
        else [
            "rope",
            "soccer",
            "running",
        ]
    )

    try:
        for sport in sports:
            print(
                "\n"
                + "-" * 70
            )

            print(
                f"[{SPORT_DIRS[sport]}] "
                f"{sport}"
            )

            print(
                "-" * 70
            )

            videos = find_videos(
                sport
            )

            if args.skill is not None:
                videos = [
                    video
                    for video in videos
                    if get_skill_number(video)
                    == args.skill
                ]

            if not videos:
                print(
                    "[주의] "
                    "분석할 영상이 없습니다."
                )
                continue

            for video_path in videos:
                try:
                    result = create_template(
                        video_path,
                        sport,
                        pose_detector,
                        yolo_model,
                    )

                    if result is None:
                        fail_count += 1
                        continue

                    output_path = (
                        save_template(
                            result,
                            sport,
                            video_path,
                        )
                    )

                    quality = (
                        result["quality"]
                    )

                    if quality[
                        "review_required"
                    ]:
                        review_count += 1
                        status = (
                            "검토 필요"
                        )
                    else:
                        success_count += 1
                        status = "정상"

                    print(
                        f"  [{status}] "
                        f"{output_path}"
                    )

                    print(
                        "    "
                        f"sequence="
                        f"{result['video']['sequence_frame_count']} / "
                        f"actual="
                        f"{result['video']['actual_frame_count']} | "
                        f"pose="
                        f"{quality['pose_detection_rate']:.1%}"
                    )

                    if sport == "soccer":
                        print(
                            "    "
                            f"ball="
                            f"{quality['ball_detection_rate']:.1%}"
                        )

                    if quality[
                        "review_reasons"
                    ]:
                        for reason in quality[
                            "review_reasons"
                        ]:
                            print(
                                f"    - {reason}"
                            )

                except Exception as error:
                    fail_count += 1

                    print(
                        f"  [실패] "
                        f"{video_path.name}"
                    )

                    print(
                        f"         {error}"
                    )

    finally:
        pose_detector.close()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Ground Truth 생성 결과"
    )

    print(
        "=" * 70
    )

    print(
        f"정상 생성 : {success_count}"
    )

    print(
        f"검토 필요 : {review_count}"
    )

    print(
        f"실패      : {fail_count}"
    )

    print(
        "\nJSON 저장 위치:"
    )

    print(
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()
