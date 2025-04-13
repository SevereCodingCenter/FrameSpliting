import cv2
import os

# 비디오 파일 경로 (raw string 또는 슬래시 사용)
video_path = r'' 
output_folder = 'output_frames-stain2'

# 출력 폴더가 없다면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

# 비디오가 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()  # 프레임 읽기

    if not ret:  # 비디오의 끝에 도달한 경우
        break

    # 프레임을 이미지 파일로 저장 (예: 'frame_0001.jpg')
    frame_filename = os.path.join(output_folder, f'frameplus_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

# 비디오 객체 해제
cap.release()

print(f"프레임이 {frame_count}개 저장되었습니다.")
