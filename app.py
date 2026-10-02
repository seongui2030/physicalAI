# ==============================================================================
# 교재명: 체육시간에 인공지능 쓰기: YOLO와 MediaPipe로 구축하는 피트니스 AI
# ==============================================================================

# ------------------------------------------------------------------------------
# [01. 프로젝트 소개 & 02. 필수 라이브러리 설치 및 불러오기]
# ------------------------------------------------------------------------------
#%pip install -q "numpy<2" "opencv-python<5" mediapipe ultralytics gradio gtts moviepy
# (※ Colab 첫 실행 시 위 설치 명령어를 주석 해제하여 먼저 실행하세요!)

import cv2                   # 영상 및 이미지 처리 라이브러리
import numpy as np           # 수학 및 배열 계산 라이브러리
import os                    # 파일 경로 및 시스템 제어 라이브러리
import urllib.request        # 인터넷 파일 다운로드 라이브러리
import mediapipe as mp       # 관절(Pose) 추적 인공지능
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO  # 객체 탐지(사람 찾기) 인공지능
from gtts import gTTS        # 글자를 음성(MP3)으로 바꿔주는 라이브러리
import gradio as gr          # 파이썬 웹 화면(UI) 제작 라이브러리


# MoviePy: 영상을 웹 브라우저용 표준 MP3/MP4로 안전하게 변환
try:
    from moviepy.editor import VideoFileClip, AudioFileClip
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False

print("✅ [01~02] 필수 라이브러리 불러오기 완료!")

# ------------------------------------------------------------------------------
# [05 & 06. AI 모델(YOLOv8 & MediaPipe Pose) 다운로드 및 로드]
# ------------------------------------------------------------------------------
# 1) MediaPipe Pose 모델 파일이 없으면 구글 서버에서 자동 다운로드
POSE_MODEL_PATH = 'pose_landmarker.task'
if not os.path.exists(POSE_MODEL_PATH):
    print("📥 MediaPipe Pose 인공지능 모델을 다운로드합니다...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, POSE_MODEL_PATH)
    print("✅ MediaPipe Pose 모델 다운로드 완료!")

# 2) YOLOv8 사람 탐지 모델 로드
yolo_model = YOLO('yolov8n.pt')
print("✅ [05~06] AI 모델 로드 완료!")

# ------------------------------------------------------------------------------
# [07. 관절 각도 계산 함수 (삼각함수 arctan2 활용)]
# ------------------------------------------------------------------------------
def calculate_angle(a, b, c):
    """
    세 점(a, b, c) 좌표를 받아 중심점 b에서의 각도(0~180도)를 계산합니다.
    - 예: a(어깨), b(팔꿈치), c(손목) -> 팔꿈치 관절 각도
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    # 두 선분의 기울기 각도 차이 구하기 (라디안)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi) # 라디안을 일반 도(degree)로 변환
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# ------------------------------------------------------------------------------
# [12 & 13. 음성 피드백 생성 및 1.5배속 변환 함수]
# ------------------------------------------------------------------------------
def create_tts_audio(text_message, output_path="feedback_audio.mp3", speed=1.5):
    """
    피드백 글자를 gTTS로 음성 파일로 만든 후, MoviePy로 1.5배속으로 빠르게 저장합니다.
    """
    temp_mp3 = "temp_gtts_sound.mp3"

    # 글자를 한국어 음성(MP3)으로 변환
    tts = gTTS(text=text_message, lang='ko')
    tts.save(temp_mp3)

    # MoviePy를 활용한 1.5배속 속도 조절
    if HAS_MOVIEPY:
        try:
            audio_clip = AudioFileClip(temp_mp3)
            fast_audio = audio_clip.speedx(factor=speed)
            fast_audio.write_audiofile(output_path, verbose=False, logger=None)
            audio_clip.close()
            fast_audio.close()
            if os.path.exists(temp_mp3): os.remove(temp_mp3)
            return output_path
        except:
            return temp_mp3
    return temp_mp3

# ------------------------------------------------------------------------------
# [14. 웹 재생을 위한 H.264 MP4 영상 코덱 변환 함수]
# ------------------------------------------------------------------------------
def convert_to_h264_web_video(input_path, output_path):
    """
    OpenCV로 만든 영상이 웹 브라우저에서 검게 뜨는 현상을 막기 위해 H.264 코덱으로 재인코딩합니다.
    """
    if HAS_MOVIEPY:
        try:
            clip = VideoFileClip(input_path)
            clip.write_videofile(output_path, codec="libx264", audio=False, verbose=False, logger=None)
            clip.close()
            return output_path
        except Exception as e:
            print(f"MoviePy 변환 중 예외: {e}")
            return input_path
    return input_path

# ------------------------------------------------------------------------------
# [04, 08~11. AI 핵심 분석 엔진: 영상 자르기 + 스켈레톤 추출 + 진단]
# ------------------------------------------------------------------------------
def analyze_sports_posture(sports_type, student_id, student_name, video_input, start_time=0.0, end_time=5.0):
    if video_input is None:
        return None, "영상을 업로드해주세요.", None

    # 학생 정보 및 종목 이름 정리
    student_id = str(student_id).strip() if student_id else "0000"
    student_name = str(student_name).strip() if student_name else "학생"
    caption_info = f"{student_id}_{student_name}_{sports_type}"

    raw_cropped_path = f"{caption_info}_cropped.mp4"
    raw_output_path = f"{caption_info}_raw_analyzed.mp4"
    final_web_video_path = f"{caption_info}_analyzed.mp4"
    audio_path = f"{caption_info}_feedback.mp3"

    # [04] 지정한 시작/종료 시간 범위만 영상 자르기 (Crop)
    cap = cv2.VideoCapture(video_input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_crop = cv2.VideoWriter(raw_cropped_path, fourcc, fps, (width, height))

    curr_frame = start_frame
    while cap.isOpened() and curr_frame < end_frame:
        ret, frame = cap.read()
        if not ret: break
        out_crop.write(frame)
        curr_frame += 1
    cap.release()
    out_crop.release()

    # [06] MediaPipe Pose 감지기 설정
    base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    pose_detector = vision.PoseLandmarker.create_from_options(options)

    # 영상 프레임 단위 분석 진행
    cap = cv2.VideoCapture(raw_cropped_path)
    out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    elbow_angles, knee_angles, trunk_angles = [], [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # [08] YOLOv8 사람 바운딩 박스 그려주기
        yolo_results = yolo_model(frame, verbose=False)[0]
        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            if yolo_model.names[cls_id] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 2)

        # [09] MediaPipe Pose 33개 관절점 및 뼈대(스켈레톤) 그리기
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = pose_detector.detect(mp_img)

        if detection_result.pose_landmarks:
            for pose_landmarks in detection_result.pose_landmarks:
                def get_pt(idx):
                    lm = pose_landmarks[idx]
                    return [int(lm.x * width), int(lm.y * height)]

                # 주요 관절 좌표 가져오기 (어깨, 팔꿈치, 손목, 골반, 무릎, 발목)
                l_shoulder, r_shoulder = get_pt(11), get_pt(12)
                l_elbow, l_wrist = get_pt(13), get_pt(15)
                l_hip, l_knee, l_ankle = get_pt(23), get_pt(25), get_pt(27)

                # 관절 각도 계산하기
                elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)
                knee = calculate_angle(l_hip, l_knee, l_ankle)
                mid_shoulder = [(l_shoulder[0] + r_shoulder[0]) // 2, (l_shoulder[1] + r_shoulder[1]) // 2]
                vertical_pt = [mid_shoulder[0], l_hip[1]]
                trunk = calculate_angle(vertical_pt, l_hip, mid_shoulder)

                elbow_angles.append(elbow)
                knee_angles.append(knee)
                trunk_angles.append(trunk)

                # 관절점 및 선 시각화
                cv2.line(frame, tuple(l_shoulder), tuple(l_elbow), (255, 0, 0), 3)
                cv2.line(frame, tuple(l_elbow), tuple(l_wrist), (255, 0, 0), 3)
                cv2.line(frame, tuple(l_hip), tuple(l_knee), (0, 255, 0), 3)
                cv2.line(frame, tuple(l_knee), tuple(l_ankle), (0, 255, 0), 3)
                for pt in [l_shoulder, l_elbow, l_wrist, l_hip, l_knee, l_ankle]:
                    cv2.circle(frame, tuple(pt), 6, (0, 0, 255), -1)

                # [10] 화면 상단 실시간 자막 오버레이
                cv2.rectangle(frame, (10, 10), (480, 90), (0, 0, 0), -1)
                cv2.putText(frame, f"Info: {caption_info}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Elbow: {int(elbow)}deg | Knee: {int(knee)}deg | Trunk: {int(trunk)}deg",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    # 웹 재생용 인코딩 변환
    final_output = convert_to_h264_web_video(raw_output_path, final_web_video_path)

    # 임시 파일 정리
    for temp_f in [raw_cropped_path, raw_output_path]:
        if os.path.exists(temp_f) and temp_f != final_output:
            try: os.remove(temp_f)
            except: pass

    # [11] 조건문(if-else)을 이용한 운동 자세 자동 진단
    avg_elbow = np.mean(elbow_angles) if elbow_angles else 90
    avg_knee = np.min(knee_angles) if knee_angles else 170
    avg_trunk = np.mean(trunk_angles) if trunk_angles else 0

    feedback_msgs = [f"[{sports_type} 자세 분석 결과]"]

    if 80 <= avg_elbow <= 120:
        feedback_msgs.append("1. 팔꿈치 각도: 양호합니다.")
    else:
        feedback_msgs.append("1. 팔꿈치 각도: 주의가 필요합니다. 팔을 과도하게 흔들지 마세요.")

    if avg_trunk < 15:
        feedback_msgs.append("2. 상체 기울임: 양호합니다.")
    else:
        feedback_msgs.append("2. 상체 기울임: 경고! 상체가 기울어졌습니다. 몸을 바로 세우세요.")

    if avg_knee < 140:
        feedback_msgs.append("3. 무릎 착지 각도: 양호합니다. 충격을 잘 완화하고 있습니다.")
    else:
        feedback_msgs.append("3. 무릎 착지 각도: 경고! 착지 시 무릎을 구부려 충격을 완화하세요.")

    final_feedback_text = "\n".join(feedback_msgs)
    tts_text = f"{student_name} 학생의 {sports_type} 자세 분석 결과입니다. " + " ".join(feedback_msgs[1:])

    # 1.5배속 음성 생성
    create_tts_audio(tts_text, audio_path, speed=1.5)

    return final_output, final_feedback_text, audio_path

# ------------------------------------------------------------------------------
# [03, 15~17. Gradio UI 대시보드 화면 설계 및 실행]
# ------------------------------------------------------------------------------
with gr.Blocks(title="피트니스 AI") as demo:
    gr.Markdown("# 🏃‍♂️ 체육시간에 인공지능 쓰기: 피트니스 AI")
    gr.Markdown("분석할 **운동 종목**을 선택하고, 학생 정보와 영상 구간을 입력한 뒤 **분석 실행** 버튼을 눌러주세요.")

    # [03] 화면 입력 요소
    selected_sport = gr.Radio(
        choices=["줄넘기", "배구", "축구", "달리기", "멀리뛰기"],
        value="줄넘기",
        label="🏀 운동 종목 선택"
    )

    with gr.Row():
        student_id_input = gr.Textbox(label="학번", placeholder="예: 10101")
        student_name_input = gr.Textbox(label="이름", placeholder="예: 홍길동")

    with gr.Row():
        video_input = gr.Video(label="운동 영상 업로드")
        with gr.Column():
            start_time_input = gr.Number(label="분석 시작 시간 (초)", value=0.0)
            end_time_input = gr.Number(label="분석 종료 시간 (초)", value=5.0)

    btn_analyze = gr.Button("🔍 AI 자세 분석 실행", variant="primary")

    # [16] 대시보드 결과 출력 요소
    gr.Markdown("### 📊 분석 및 피드백 대시보드")
    with gr.Row():
        output_video = gr.Video(label="분석 결과 영상 (Skeleton Overlay)")
        with gr.Column():
            output_feedback = gr.Textbox(label="피드백 진단 결과", lines=6)
            output_audio = gr.Audio(label="음성 피드백 (gTTS 1.5배속)", autoplay=True)

    # [15] 버튼 클릭 이벤트와 AI 분석 함수 연결
    btn_analyze.click(
        fn=analyze_sports_posture,
        inputs=[selected_sport, student_id_input, student_name_input, video_input, start_time_input, end_time_input],
        outputs=[output_video, output_feedback, output_audio]
    )

# [17] Colab 외부 접속 가능한 공유 링크(share=True) 생성 실행
if __name__ == "__main__":
    demo.launch(share=True)