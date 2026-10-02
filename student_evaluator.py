"""
student_evaluator.py

학생 영상을 Ground Truth와 비교하여 평가하는 프로그램

평가 구조
1. 학생 영상 분석
2. MediaPipe Pose 특징 추출
3. 축구인 경우 YOLO 공 위치 추출
4. Ground Truth JSON 불러오기
5. 특징 정규화
6. DTW 비교
7. 자세 점수 / 움직임·리듬 점수 / 시선 점수 계산
8. 총점 및 등급 계산
9. gTTS 음성 피드백 생성

사용 예시
------------------------------------------------
전체 경로 자동 탐색:

python student_evaluator.py \
    --sport soccer \
    --skill 01 \
    --video videos/student/inside_touch.mp4

특정 Ground Truth 지정:

python student_evaluator.py \
    --sport soccer \
    --skill 01 \
    --video videos/student/inside_touch.mp4 \
    --teacher videos/teacher_templates/soccer/soccer_01_Inside_Touch.json
------------------------------------------------
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------
# MediaPipe Tasks API
# ---------------------------------------------------------
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------
# YOLO
# ---------------------------------------------------------
from ultralytics import YOLO


# =========================================================
# 1. 프로젝트 경로
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

POSE_MODEL_PATH = PROJECT_ROOT / "pose_landmarker.task"
YOLO_MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"

TEACHER_TEMPLATE_DIR = PROJECT_ROOT / "videos" / "teacher_templates"

RESULT_DIR = PROJECT_ROOT / "results"


# =========================================================
# 2. 평가 기준
# =========================================================

POSTURE_WEIGHT = 50
RHYTHM_WEIGHT = 30
GAZE_WEIGHT = 20

GRADE_A = 90
GRADE_B = 80
GRADE_C = 70


# =========================================================
# 3. MediaPipe / YOLO 초기화
# =========================================================

def create_pose_landmarker():
    """
    MediaPipe PoseLandmarker 생성
    """

    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pose 모델을 찾을 수 없습니다:\n{POSE_MODEL_PATH}"
        )

    base_options = python.BaseOptions(
        model_asset_path=str(POSE_MODEL_PATH)
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.PoseLandmarker.create_from_options(options)


def create_yolo_model():
    """
    축구공 검출용 YOLO 모델 생성
    """

    if not YOLO_MODEL_PATH.exists():
        print("⚠️ yolov8n.pt를 찾을 수 없습니다.")
        return None

    try:
        return YOLO(str(YOLO_MODEL_PATH))
    except Exception as e:
        print(f"⚠️ YOLO 초기화 실패: {e}")
        return None


# =========================================================
# 4. 기본 수학 함수
# =========================================================

def calculate_angle(a, b, c):
    """
    세 점 a-b-c의 각도 계산
    """

    if a is None or b is None or c is None:
        return None

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)

    angle = math.degrees(math.acos(cosine))

    return float(angle)


def safe_float(value):
    """
    숫자를 안전하게 float으로 변환
    """

    if value is None:
        return None

    try:
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def mean_or_none(values):
    """
    None을 제외한 평균
    """

    valid = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]

    if not valid:
        return None

    return float(np.mean(valid))


# =========================================================
# 5. Pose 특징 추출
# =========================================================

def extract_pose_features(image, pose_result):
    """
    MediaPipe Pose 결과에서 평가에 필요한 특징 추출

    Ground Truth와 학생 영상에서
    동일한 특징을 추출해야 한다.
    """

    if not pose_result.pose_landmarks:
        return None

    lm = pose_result.pose_landmarks[0]

    # MediaPipe landmark 번호
    NOSE = 0

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    LEFT_HIP = 23
    RIGHT_HIP = 24

    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    nose = [lm[NOSE].x, lm[NOSE].y]

    left_shoulder = [
        lm[LEFT_SHOULDER].x,
        lm[LEFT_SHOULDER].y
    ]

    right_shoulder = [
        lm[RIGHT_SHOULDER].x,
        lm[RIGHT_SHOULDER].y
    ]

    left_hip = [
        lm[LEFT_HIP].x,
        lm[LEFT_HIP].y
    ]

    right_hip = [
        lm[RIGHT_HIP].x,
        lm[RIGHT_HIP].y
    ]

    left_knee = [
        lm[LEFT_KNEE].x,
        lm[LEFT_KNEE].y
    ]

    right_knee = [
        lm[RIGHT_KNEE].x,
        lm[RIGHT_KNEE].y
    ]

    left_ankle = [
        lm[LEFT_ANKLE].x,
        lm[LEFT_ANKLE].y
    ]

    right_ankle = [
        lm[RIGHT_ANKLE].x,
        lm[RIGHT_ANKLE].y
    ]

    # -----------------------------------------------------
    # 중심점
    # -----------------------------------------------------

    shoulder_x = (
        left_shoulder[0] + right_shoulder[0]
    ) / 2

    shoulder_y = (
        left_shoulder[1] + right_shoulder[1]
    ) / 2

    hip_x = (
        left_hip[0] + right_hip[0]
    ) / 2

    hip_y = (
        left_hip[1] + right_hip[1]
    ) / 2

    # -----------------------------------------------------
    # 관절 각도
    # -----------------------------------------------------

    left_knee_angle = calculate_angle(
        left_hip,
        left_knee,
        left_ankle
    )

    right_knee_angle = calculate_angle(
        right_hip,
        right_knee,
        right_ankle
    )

    # -----------------------------------------------------
    # gaze proxy
    #
    # 실제 눈동자 추적이 아니라
    # 코 위치와 어깨 중심의 차이를 사용
    # -----------------------------------------------------

    gaze_proxy = abs(nose[0] - shoulder_x)

    # -----------------------------------------------------
    # visibility
    # -----------------------------------------------------

    selected_indices = [
        NOSE,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_ANKLE,
        RIGHT_ANKLE
    ]

    visibility_values = []

    for index in selected_indices:
        visibility = getattr(
            lm[index],
            "visibility",
            None
        )

        if visibility is not None:
            visibility_values.append(float(visibility))

    mean_visibility = mean_or_none(
        visibility_values
    )

    return {
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,

        "left_ankle_x": left_ankle[0],
        "left_ankle_y": left_ankle[1],

        "right_ankle_x": right_ankle[0],
        "right_ankle_y": right_ankle[1],

        "hip_x": hip_x,
        "hip_y": hip_y,

        "shoulder_x": shoulder_x,
        "shoulder_y": shoulder_y,

        "nose_x": nose[0],
        "nose_y": nose[1],

        "gaze_proxy": gaze_proxy,

        "visibility": mean_visibility,
    }


# =========================================================
# 6. YOLO 축구공 검출
# =========================================================

def detect_ball(image, yolo_model):
    """
    COCO의 sports ball 클래스를 이용해 공 중심 좌표 검출

    반환:
        [x, y]
        또는 None
    """

    if yolo_model is None:
        return None

    try:
        results = yolo_model(
            image,
            conf=0.25,
            verbose=False
        )

        if not results:
            return None

        result = results[0]

        if result.boxes is None:
            return None

        height, width = image.shape[:2]

        for box in result.boxes:

            class_id = int(box.cls[0])

            class_name = yolo_model.names[class_id]

            if class_name != "sports ball":
                continue

            x1, y1, x2, y2 = map(
                float,
                box.xyxy[0]
            )

            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height

            return [
                float(center_x),
                float(center_y)
            ]

    except Exception:
        return None

    return None


# =========================================================
# 7. 학생 영상 분석
# =========================================================

def analyze_student_video(
    video_path,
    sport,
    pose_landmarker,
    yolo_model=None
):
    """
    학생 영상을 프레임 단위로 분석
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"학생 영상을 찾을 수 없습니다:\n{video_path}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"영상을 열 수 없습니다:\n{video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    estimated_frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    sequence = []

    frame_index = 0
    pose_detected_frames = 0
    ball_detected_frames = 0

    print()
    print("🎥 학생 영상 분석 시작")
    print(f"   파일: {video_path.name}")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        pose_result = pose_landmarker.detect(
            mp_image
        )

        features = extract_pose_features(
            frame,
            pose_result
        )

        if features is not None:
            pose_detected_frames += 1

        ball = None

        if sport == "soccer":
            ball = detect_ball(
                frame,
                yolo_model
            )

            if ball is not None:
                ball_detected_frames += 1

        # -------------------------------------------------
        # Ground Truth와 동일하게 한 프레임씩 기록
        # -------------------------------------------------

        frame_data = {
            "frame": frame_index,
            "time_sec": frame_index / fps,
            "pose_detected": features is not None,
        }

        if features is not None:
            frame_data.update(features)
        else:
            # Pose 실패 프레임도 제거하지 않는다.
            frame_data.update({
                "left_knee_angle": None,
                "right_knee_angle": None,

                "left_ankle_x": None,
                "left_ankle_y": None,

                "right_ankle_x": None,
                "right_ankle_y": None,

                "hip_x": None,
                "hip_y": None,

                "shoulder_x": None,
                "shoulder_y": None,

                "nose_x": None,
                "nose_y": None,

                "gaze_proxy": None,
                "visibility": None,
            })

        if sport == "soccer":

            frame_data["ball_detected"] = (
                ball is not None
            )

            if ball is not None:
                frame_data["ball_x"] = ball[0]
                frame_data["ball_y"] = ball[1]
            else:
                frame_data["ball_x"] = None
                frame_data["ball_y"] = None

        sequence.append(frame_data)

        frame_index += 1

        if frame_index % 100 == 0:
            print(
                f"   처리 프레임: "
                f"{frame_index}/{estimated_frame_count}"
            )

    cap.release()

    actual_frame_count = len(sequence)

    pose_rate = (
        pose_detected_frames / actual_frame_count
        if actual_frame_count > 0
        else 0
    )

    ball_rate = None

    if sport == "soccer":
        ball_rate = (
            ball_detected_frames / actual_frame_count
            if actual_frame_count > 0
            else 0
        )

    print()
    print("✅ 학생 영상 분석 완료")
    print(f"   실제 프레임: {actual_frame_count}")
    print(f"   Pose 검출률: {pose_rate:.1%}")

    if sport == "soccer":
        print(f"   공 검출률: {ball_rate:.1%}")

    return {
        "video": {
            "fps": fps,
            "frame_count": actual_frame_count,
            "estimated_frame_count": estimated_frame_count,
            "duration_sec": (
                actual_frame_count / fps
                if fps > 0
                else 0
            ),
        },

        "quality": {
            "pose_detection_rate": pose_rate,
            "ball_detection_rate": ball_rate,
        },

        "sequence": sequence,
    }


# =========================================================
# 8. Ground Truth 불러오기
# =========================================================

def find_teacher_template(
    sport,
    skill_number,
    teacher_path=None
):
    """
    Ground Truth JSON 검색
    """

    if teacher_path is not None:

        path = Path(teacher_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Ground Truth 파일을 찾을 수 없습니다:\n{path}"
            )

        return path

    if not TEACHER_TEMPLATE_DIR.exists():
        raise FileNotFoundError(
            f"Ground Truth 폴더가 없습니다:\n"
            f"{TEACHER_TEMPLATE_DIR}"
        )

    skill_text = f"{int(skill_number):02d}"

    candidates = []

    for path in TEACHER_TEMPLATE_DIR.rglob("*.json"):

        name = path.stem.lower()

        sport_match = sport.lower() in name

        skill_match = (
            f"_{skill_text}_" in f"_{name}_"
            or name.startswith(
                f"{sport.lower()}_{skill_text}"
            )
        )

        if sport_match and skill_match:
            candidates.append(path)

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Ground Truth JSON을 찾지 못했습니다.\n"
            f"sport={sport}, skill={skill_text}"
        )

    if len(candidates) > 1:

        print("⚠️ Ground Truth 후보가 여러 개입니다.")

        for candidate in candidates:
            print(f"   - {candidate}")

        raise RuntimeError(
            "Ground Truth 파일을 하나로 특정할 수 없습니다. "
            "--teacher 옵션으로 직접 지정하세요."
        )

    return candidates[0]


def load_teacher_template(path):
    """
    Ground Truth JSON 로드
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if "sequence" not in data:
        raise ValueError(
            "Ground Truth JSON에 sequence가 없습니다."
        )

    if not isinstance(data["sequence"], list):
        raise ValueError(
            "Ground Truth sequence 형식이 잘못되었습니다."
        )

    return data


# =========================================================
# 9. 시계열 추출
# =========================================================

POSTURE_FEATURES = [
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
]

GAZE_FEATURES = [
    "gaze_proxy"
]

BALL_FEATURES = [
    "ball_x",
    "ball_y"
]


def extract_feature_matrix(
    sequence,
    feature_names
):
    """
    JSON sequence에서 특정 feature의
    시계열 데이터를 가져온다.

    반환:
        numpy array
        shape = (time, features)
    """

    rows = []

    for frame in sequence:

        row = []

        for feature in feature_names:

            value = frame.get(feature)

            value = safe_float(value)

            row.append(
                np.nan if value is None else value
            )

        rows.append(row)

    if not rows:
        return np.empty((0, len(feature_names)))

    return np.array(
        rows,
        dtype=float
    )


# =========================================================
# 10. 결측값 보간
# =========================================================

def interpolate_missing(data):
    """
    NaN을 선형 보간

    앞/뒤 값이 없는 경우
    존재하는 값으로 채운다.
    """

    data = np.asarray(
        data,
        dtype=float
    ).copy()

    if data.size == 0:
        return data

    for column in range(data.shape[1]):

        values = data[:, column]

        valid = np.isfinite(values)

        if not np.any(valid):
            values[:] = 0.0
            continue

        valid_indices = np.where(valid)[0]

        values[:valid_indices[0]] = (
            values[valid_indices[0]]
        )

        values[valid_indices[-1] + 1:] = (
            values[valid_indices[-1]]
        )

        missing_indices = np.where(~valid)[0]

        if len(missing_indices) > 0:

            values[missing_indices] = np.interp(
                missing_indices,
                valid_indices,
                values[valid_indices]
            )

        data[:, column] = values

    return data


# =========================================================
# 11. 특징 정규화
# =========================================================

def normalize_pair(
    teacher,
    student
):
    """
    Teacher 기준으로 두 시계열을 정규화한다.

    이렇게 하면 영상마다 발생하는
    단순한 좌표 크기 차이를 어느 정도 줄일 수 있다.
    """

    teacher = interpolate_missing(teacher)
    student = interpolate_missing(student)

    if teacher.shape[0] == 0:
        return teacher, student

    mean = np.mean(
        teacher,
        axis=0
    )

    std = np.std(
        teacher,
        axis=0
    )

    std[std < 1e-6] = 1.0

    teacher_normalized = (
        teacher - mean
    ) / std

    student_normalized = (
        student - mean
    ) / std

    return (
        teacher_normalized,
        student_normalized
    )


# =========================================================
# 12. DTW
# =========================================================

def frame_distance(a, b):
    """
    두 프레임의 feature 거리
    """

    return float(
        np.mean(
            np.abs(a - b)
        )
    )


def calculate_dtw_distance(
    teacher,
    student
):
    """
    기본 DTW 구현

    scipy 없이 사용할 수 있도록 직접 구현
    """

    if (
        teacher.shape[0] == 0
        or student.shape[0] == 0
    ):
        return float("inf")

    n = teacher.shape[0]
    m = student.shape[0]

    # 메모리 절약을 위해 두 행만 사용
    previous = np.full(
        m + 1,
        np.inf
    )

    current = np.full(
        m + 1,
        np.inf
    )

    previous[0] = 0.0

    for i in range(1, n + 1):

        current[:] = np.inf

        for j in range(1, m + 1):

            cost = frame_distance(
                teacher[i - 1],
                student[j - 1]
            )

            current[j] = cost + min(
                previous[j],
                current[j - 1],
                previous[j - 1]
            )

        previous, current = (
            current,
            previous
        )

    total_distance = previous[m]

    # 영상 길이에 영향을 덜 받도록 정규화
    path_length = n + m

    return float(
        total_distance / path_length
    )


# =========================================================
# 13. DTW 거리 → 유사도
# =========================================================

def distance_to_similarity(
    distance
):
    """
    DTW distance를 0~100 유사도로 변환

    exp 방식으로 부드럽게 감소시킨다.
    """

    if not math.isfinite(distance):
        return 0.0

    similarity = (
        100.0
        * math.exp(-distance)
    )

    return float(
        np.clip(
            similarity,
            0,
            100
        )
    )


# =========================================================
# 14. 자세 점수
# =========================================================

def calculate_posture_score(
    teacher_sequence,
    student_sequence
):
    """
    자세 관련 특징을 DTW로 비교

    최대 50점
    """

    teacher_matrix = extract_feature_matrix(
        teacher_sequence,
        POSTURE_FEATURES
    )

    student_matrix = extract_feature_matrix(
        student_sequence,
        POSTURE_FEATURES
    )

    teacher_matrix, student_matrix = (
        normalize_pair(
            teacher_matrix,
            student_matrix
        )
    )

    dtw_distance = calculate_dtw_distance(
        teacher_matrix,
        student_matrix
    )

    similarity = distance_to_similarity(
        dtw_distance
    )

    score = (
        similarity
        * POSTURE_WEIGHT
        / 100
    )

    return {
        "score": score,
        "similarity": similarity,
        "dtw_distance": dtw_distance,
    }


# =========================================================
# 15. 움직임 / 리듬 점수
# =========================================================

def calculate_movement_signal(
    sequence
):
    """
    발목 움직임의 프레임 간 변화량 계산
    """

    points = []

    for frame in sequence:

        lx = safe_float(
            frame.get("left_ankle_x")
        )
        ly = safe_float(
            frame.get("left_ankle_y")
        )

        rx = safe_float(
            frame.get("right_ankle_x")
        )
        ry = safe_float(
            frame.get("right_ankle_y")
        )

        if None in (lx, ly, rx, ry):
            points.append(None)
            continue

        points.append(
            [
                lx,
                ly,
                rx,
                ry
            ]
        )

    movement = []

    previous = None

    for point in points:

        if point is None:
            movement.append(None)
            continue

        if previous is None:
            movement.append(0.0)
        else:
            distance = np.linalg.norm(
                np.array(point)
                - np.array(previous)
            )

            movement.append(
                float(distance)
            )

        previous = point

    return np.array(
        movement,
        dtype=float
    ).reshape(-1, 1)


def calculate_ball_movement_signal(
    sequence
):
    """
    축구공의 프레임 간 이동량 계산
    """

    points = []

    for frame in sequence:

        x = safe_float(
            frame.get("ball_x")
        )

        y = safe_float(
            frame.get("ball_y")
        )

        if x is None or y is None:
            points.append(None)
        else:
            points.append([x, y])

    movement = []

    previous = None

    for point in points:

        if point is None:
            movement.append(None)
            continue

        if previous is None:
            movement.append(0.0)
        else:

            distance = np.linalg.norm(
                np.array(point)
                - np.array(previous)
            )

            movement.append(
                float(distance)
            )

        previous = point

    return np.array(
        movement,
        dtype=float
    ).reshape(-1, 1)


def calculate_rhythm_score(
    teacher_sequence,
    student_sequence,
    sport
):
    """
    움직임 패턴을 DTW로 비교

    현재 버전에서는 '터치 횟수'를 직접 세기보다는
    움직임/리듬 패턴을 평가한다.

    최대 30점
    """

    if sport == "soccer":

        teacher_signal = (
            calculate_ball_movement_signal(
                teacher_sequence
            )
        )

        student_signal = (
            calculate_ball_movement_signal(
                student_sequence
            )
        )

        # 공 검출 데이터가 거의 없으면
        # 발목 움직임으로 fallback
        teacher_valid = np.isfinite(
            teacher_signal
        ).sum()

        student_valid = np.isfinite(
            student_signal
        ).sum()

        if (
            teacher_valid < 5
            or student_valid < 5
        ):
            teacher_signal = (
                calculate_movement_signal(
                    teacher_sequence
                )
            )

            student_signal = (
                calculate_movement_signal(
                    student_sequence
                )
            )

    else:

        teacher_signal = (
            calculate_movement_signal(
                teacher_sequence
            )
        )

        student_signal = (
            calculate_movement_signal(
                student_sequence
            )
        )

    teacher_signal, student_signal = (
        normalize_pair(
            teacher_signal,
            student_signal
        )
    )

    dtw_distance = calculate_dtw_distance(
        teacher_signal,
        student_signal
    )

    similarity = distance_to_similarity(
        dtw_distance
    )

    score = (
        similarity
        * RHYTHM_WEIGHT
        / 100
    )

    return {
        "score": score,
        "similarity": similarity,
        "dtw_distance": dtw_distance,
    }


# =========================================================
# 16. 시선 점수
# =========================================================

def calculate_gaze_score(
    teacher_sequence,
    student_sequence
):
    """
    gaze_proxy 패턴 비교

    주의:
    실제 눈동자 시선 추적이 아니다.
    """

    teacher_matrix = extract_feature_matrix(
        teacher_sequence,
        GAZE_FEATURES
    )

    student_matrix = extract_feature_matrix(
        student_sequence,
        GAZE_FEATURES
    )

    teacher_matrix, student_matrix = (
        normalize_pair(
            teacher_matrix,
            student_matrix
        )
    )

    dtw_distance = calculate_dtw_distance(
        teacher_matrix,
        student_matrix
    )

    similarity = distance_to_similarity(
        dtw_distance
    )

    score = (
        similarity
        * GAZE_WEIGHT
        / 100
    )

    return {
        "score": score,
        "similarity": similarity,
        "dtw_distance": dtw_distance,
    }


# =========================================================
# 17. 총점
# =========================================================

def calculate_total_score(
    posture_result,
    rhythm_result,
    gaze_result
):
    """
    총점 계산
    """

    total = (
        posture_result["score"]
        + rhythm_result["score"]
        + gaze_result["score"]
    )

    return float(
        np.clip(
            total,
            0,
            100
        )
    )


# =========================================================
# 18. 등급
# =========================================================

def calculate_grade(score):

    if score >= GRADE_A:
        return "A"

    if score >= GRADE_B:
        return "B"

    if score >= GRADE_C:
        return "C"

    return "D"


# =========================================================
# 19. 피드백
# =========================================================

def make_feedback(
    score,
    grade,
    posture_result,
    rhythm_result,
    gaze_result
):
    """
    학생에게 보여줄 간단한 피드백
    """

    feedback = []

    if grade == "A":
        feedback.append(
            "전체 동작이 선생님 기준 동작과 매우 유사합니다."
        )

    elif grade == "B":
        feedback.append(
            "전체적인 동작은 잘 수행했습니다."
        )

    elif grade == "C":
        feedback.append(
            "기본 동작을 다시 천천히 연습해 보세요."
        )

    else:
        feedback.append(
            "동작을 작은 단계로 나누어 다시 연습해 보세요."
        )

    # 자세
    if posture_result["similarity"] < 70:
        feedback.append(
            "자세와 관절 움직임을 조금 더 정확하게 만들어 보세요."
        )

    # 리듬
    if rhythm_result["similarity"] < 70:
        feedback.append(
            "동작의 속도와 리듬을 일정하게 유지해 보세요."
        )

    # 시선
    if gaze_result["similarity"] < 70:
        feedback.append(
            "동작 중 머리 방향과 시선을 안정적으로 유지해 보세요."
        )

    return feedback


# =========================================================
# 20. 결과 저장
# =========================================================

def save_result(
    result,
    student_video
):
    """
    평가 결과 JSON 저장
    """

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    video_name = Path(
        student_video
    ).stem

    output_path = (
        RESULT_DIR
        / f"{video_name}_evaluation.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    return output_path


# =========================================================
# 21. gTTS
# =========================================================

def make_voice_feedback(
    feedback,
    output_path
):
    """
    gTTS를 이용한 음성 피드백

    gTTS가 설치되지 않았으면
    프로그램 전체가 중단되지 않도록 처리한다.
    """

    try:

        from gtts import gTTS

        text = " ".join(feedback)

        tts = gTTS(
            text=text,
            lang="ko"
        )

        tts.save(
            str(output_path)
        )

        return True

    except Exception as e:

        print(
            f"⚠️ 음성 피드백 생성 실패: {e}"
        )

        return False


# =========================================================
# 22. 메인 평가
# =========================================================

def evaluate_student(
    sport,
    skill_number,
    student_video,
    teacher_path=None,
    make_voice=False
):
    """
    전체 평가 실행
    """

    print("=" * 60)
    print("🏃 Physical AI Student Evaluator")
    print("=" * 60)

    # -----------------------------------------------------
    # Ground Truth
    # -----------------------------------------------------

    teacher_json = find_teacher_template(
        sport,
        skill_number,
        teacher_path
    )

    print()
    print(f"📚 Ground Truth: {teacher_json}")

    teacher_data = load_teacher_template(
        teacher_json
    )

    teacher_sequence = (
        teacher_data["sequence"]
    )

    # -----------------------------------------------------
    # 모델
    # -----------------------------------------------------

    print()
    print("🤖 AI 모델 초기화")

    pose_landmarker = (
        create_pose_landmarker()
    )

    yolo_model = None

    if sport == "soccer":
        yolo_model = create_yolo_model()

    # -----------------------------------------------------
    # 학생 영상
    # -----------------------------------------------------

    student_data = analyze_student_video(
        student_video,
        sport,
        pose_landmarker,
        yolo_model
    )

    student_sequence = (
        student_data["sequence"]
    )

    # -----------------------------------------------------
    # 평가
    # -----------------------------------------------------

    print()
    print("📊 DTW 평가 시작")

    posture_result = (
        calculate_posture_score(
            teacher_sequence,
            student_sequence
        )
    )

    rhythm_result = (
        calculate_rhythm_score(
            teacher_sequence,
            student_sequence,
            sport
        )
    )

    gaze_result = (
        calculate_gaze_score(
            teacher_sequence,
            student_sequence
        )
    )

    total_score = calculate_total_score(
        posture_result,
        rhythm_result,
        gaze_result
    )

    grade = calculate_grade(
        total_score
    )

    feedback = make_feedback(
        total_score,
        grade,
        posture_result,
        rhythm_result,
        gaze_result
    )

    # -----------------------------------------------------
    # 결과
    # -----------------------------------------------------

    result = {
        "evaluation_version": "1.0",

        "sport": sport,
        "skill_number": int(skill_number),

        "student_video": str(
            Path(student_video)
        ),

        "teacher_template": str(
            teacher_json
        ),

        "scores": {
            "posture": round(
                posture_result["score"],
                2
            ),

            "rhythm": round(
                rhythm_result["score"],
                2
            ),

            "gaze": round(
                gaze_result["score"],
                2
            ),

            "total": round(
                total_score,
                2
            )
        },

        "similarity": {
            "posture": round(
                posture_result["similarity"],
                2
            ),

            "rhythm": round(
                rhythm_result["similarity"],
                2
            ),

            "gaze": round(
                gaze_result["similarity"],
                2
            )
        },

        "dtw": {
            "posture": posture_result[
                "dtw_distance"
            ],

            "rhythm": rhythm_result[
                "dtw_distance"
            ],

            "gaze": gaze_result[
                "dtw_distance"
            ]
        },

        "student_quality": (
            student_data["quality"]
        ),

        "grade": grade,

        "feedback": feedback,
    }

    # -----------------------------------------------------
    # 결과 출력
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("📋 평가 결과")
    print("=" * 60)

    print(
        f"자세      : "
        f"{result['scores']['posture']:.1f} / 50"
    )

    print(
        f"리듬      : "
        f"{result['scores']['rhythm']:.1f} / 30"
    )

    print(
        f"시선      : "
        f"{result['scores']['gaze']:.1f} / 20"
    )

    print("-" * 60)

    print(
        f"총점      : "
        f"{result['scores']['total']:.1f} / 100"
    )

    print(
        f"등급      : {result['grade']}"
    )

    print()

    print("💬 피드백")

    for message in feedback:
        print(f"  • {message}")

    # -----------------------------------------------------
    # 결과 JSON
    # -----------------------------------------------------

    result_path = save_result(
        result,
        student_video
    )

    print()
    print(
        f"💾 결과 저장: {result_path}"
    )

    # -----------------------------------------------------
    # 음성
    # -----------------------------------------------------

    if make_voice:

        voice_path = (
            result_path.parent
            / f"{result_path.stem}.mp3"
        )

        if make_voice_feedback(
            feedback,
            voice_path
        ):

            print(
                f"🔊 음성 저장: {voice_path}"
            )

    return result


# =========================================================
# 23. CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Physical AI 학생 영상 평가 프로그램"
    )

    parser.add_argument(
        "--sport",
        required=True,
        choices=[
            "rope",
            "soccer",
            "running"
        ],
        help="종목"
    )

    parser.add_argument(
        "--skill",
        required=True,
        type=int,
        help="기술 번호"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="학생 영상 경로"
    )

    parser.add_argument(
        "--teacher",
        default=None,
        help="Ground Truth JSON 직접 지정"
    )

    parser.add_argument(
        "--voice",
        action="store_true",
        help="gTTS 음성 피드백 생성"
    )

    args = parser.parse_args()

    evaluate_student(
        sport=args.sport,
        skill_number=args.skill,
        student_video=args.video,
        teacher_path=args.teacher,
        make_voice=args.voice
    )


if __name__ == "__main__":
    main()