import cv2
import numpy as np
import os

def homomorphic_filter(img):
    img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y = img_YUV[:, :, 0]

    rows, cols = y.shape
    imgLog = np.log1p(np.array(y, dtype='float') / 255)

    M = 2 * rows + 1
    N = 2 * cols + 1

    sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
    Xc = np.ceil(N / 2)
    Yc = np.ceil(M / 2)
    gaussianNumerator = (X - Xc) ** 2 + (Y - Yc) ** 2

    LPF = np.exp(-gaussianNumerator / (2 * sigma * sigma))
    HPF = 1 - LPF

    LPF_shift = np.fft.ifftshift(LPF.copy())
    HPF_shift = np.fft.ifftshift(HPF.copy())

    img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
    img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N)))
    img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N)))

    gamma1 = 0.3
    gamma2 = 1.5
    img_adjusting = gamma1 * img_LF[0:rows, 0:cols] + gamma2 * img_HF[0:rows, 0:cols]

    img_exp = np.expm1(img_adjusting)
    img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp))
    img_out = np.array(255 * img_exp, dtype='uint8')

    img_YUV[:, :, 0] = img_out
    result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)

    return result

# 이미지 폴더 경로 설정
input_folder = r'C:\Users\82105\Desktop\opencv-image\output_frames-stain2'  
output_folder = 'C:/Users/82105/output_homomo'
os.makedirs(output_folder, exist_ok=True)

# 폴더 내 모든 이미지 파일 처리
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f'filtered2_{filename}')

        img = cv2.imread(input_path)
        if img is None:
            print(f"이미지 로딩 실패: {filename}")
            continue

        result = homomorphic_filter(img)
        cv2.imwrite(output_path, result)
        print(f"저장 완료: {output_path}")
