# AeroEyes: Finding and Rescuing with AI-Powered Drones

Dự án này là giải pháp cho bài toán **AeroEyes** thuộc **Zalo AI Challenge 2025 - Track 1**, tập trung vào việc định vị không gian - thời gian (Spatio-Temporal Localization) các đối tượng cần tìm kiếm cứu nạn từ video quay bằng drone.

Thông tin cuộc thi xem tại: [Zalo AI Challenge - AeroEyes Portal](https://challenge.zalo.ai/portal/aero-eyes)

## I. Danh sách thành viên:

**Team:** [HCMUS - FIT] T5H

| STT | Họ và tên | MSSV | Vai trò |
|:---:|-----------|------|---------|
| 1 | Hà Đức Huy | 22120133 | Nhóm trưởng |
| 2 | Trần Gia Hào | 22120099 | Thành viên |
| 3 | Nguyễn Văn Hiến | 22120101 | Thành viên |
| 4 | Nguyễn Minh Hưng | 22120123 | Thành viên |
| 5 | Nguyễn Tấn Hưng | 22120126 | Thành viên |

## II. Cấu trúc Repository

Dựa trên cấu trúc thư mục hiện tại của dự án:

```
Zalo_AIC25_Track1/
├── ckpts/                          # Chứa checkpoint mô hình đã huấn luyện
│   ├── yolo11n/                    # Checkpoint cho bản Nano
│   │   └── best.pt
│   └── yolo11s/                    # Checkpoint cho bản Small (Best model)
│       └── best.pt
├── source/                         # Mã nguồn chính (Jupyter Notebooks)
│   ├── data_preprocessing.ipynb    # Notebook tiền xử lý: Trích xuất frame, chuẩn hóa label
│   ├── train.ipynb                 # Notebook huấn luyện mô hình YOLO
│   └── inference.ipynb             # Notebook chạy inference, tracking và tạo file submission
├── submit_private_test/            # Chứa các file kết quả submission.json cho tập private_test
├── Report.pdf                      # Báo cáo chi tiết
├── requirements.txt                # Danh sách thư viện cần thiết
└── README.md                       # <- File này
```

## III. Cài đặt môi trường (Environment Setup)

Dự án yêu cầu **Python 3.12** và hỗ trợ GPU (CUDA) để huấn luyện/suy luận hiệu quả. Hoặc có thể sử dụng Google Colab với GPU L4.

**1. Tạo và kích hoạt môi trường conda:**

Nếu sử dụng môi trường conda, chạy các lệnh sau trong terminal (Nếu sử dung Google Colab, bước này có thể bỏ qua):

```bash
conda create -n aeroeyes python=3.12
conda activate aeroeyes

```

**2. Cài đặt thư viện:**

Chạy lệnh sau để cài đặt tất cả các thư viện cần thiết từ file `requirements.txt`:

```bash
pip install -r requirements.txt

```

Với Google Colab, do một số thư viện có thể đã được cài sẵn, vì thế chỉ cần cài đặt thêm các thư viện chưa có như `ultralytics`, `rembg`, `onnxruntime`, v.v.

## IV. Dữ liệu (Dataset)

**1. Tải dữ liệu:**
Các tâp dữ liệu sử dụng được cung cấp bởi Zalo AI Challenge:
  - **Train/Valid Data:** [Drive Link](https://drive.google.com/file/d/1ZIOQWhBuAd26jqkDXAnftCZDkJHNs8vd/view?usp=sharing)
  - **Public Test Data:** [Drive Link](https://drive.google.com/file/d/1WtHDIIXl9PIfegOJBXRVZypWSVq56W8q/view?usp=sharing)
  - **Private Test Data:** [Drive Link](https://drive.google.com/file/d/1GcmJIulmENEK3qsApuQoQ1UcHI3q73YT/view?usp=sharing)

**2. Cấu trúc thư mục dữ liệu:**
Cấu trúc thư mục dữ liệu sau khi giải nén như sau:
```
dataset_root/
├── train/
│   ├── annotations/
│   │   └── annotations.json
│   └── samples/
│       ├── video_id_1/
│       │   └── drone_video.mp4
│       └── ...
└── ...
```

## V. Tiền xử lý (Preprocessing)

Giai đoạn này trích xuất khung hình từ video, chuyển đổi bounding box sang định dạng YOLO và chia tập train/val .

- **File thực hiện:** `source/data_preprocessing.ipynb`
- **Cách chạy:** Mở notebook và chạy tuần tự các cell.
- **Output:** Notebook sẽ tạo ra thư mục dữ liệu chuẩn YOLO (`yolo_dataset/`) chứa ảnh và nhãn `.txt` tương ứng .

Dữ liệu sau khi tiền xử lý có thể tải ở đây: [Train/Valid Data](https://drive.google.com/file/d/1M6DSPi5En1DZXn-iJsDwMjISd3bagboY/view?usp=sharing)

## VI. Huấn luyện (Training)

Nhóm sử dụng mô hình **YOLO11** (phiên bản `yolo11s.pt`) làm backbone chính.

- **File thực hiện:** `source/train.ipynb`
- **Cách chạy:** Mở notebook, trỏ đường dẫn tới file `dataset.yaml` được tạo ra ở bước Preprocessing và bắt đầu huấn luyện.
- **Cấu hình chính:**
    * Epochs: 100
    * Image Size: 640
    * Batch size: 64
    * Augmentation: MixUp, Rotation, Shear.
- Sau khi train xong, weight tốt nhất sẽ được lưu tại `best.pt`.

## VII. Suy luận & Đánh giá (Inference & Evaluation)

Quy trình suy luận bao gồm Detection (YOLO), Verification (DINOv3 + Color Filter) và Temporal Tracking .

**1. Chuẩn bị Checkpoint:**
Đảm bảo đã có file `best.pt`. Nếu không huấn luyện lại thì có thể sử dụng checkpoint đi kèm trong repo ở thư mục `ckpts/yolo11s/best.pt` (ưu tiên) hoặc `ckpts/yolo11n/best.pt`.

**2. Chạy Inference:**
- **File thực hiện:** `source/inference.ipynb`
- **Cách chạy:**
1. Cấu hình đường dẫn tới file video test và ảnh tham chiếu (reference images).
2. Load model YOLO từ `/best.pt`.
3. Chạy các cell để thực hiện streaming inference và post-processing (tracking, smoothing).
4. Kết quả cuối cùng sẽ được xuất ra file JSON (ví dụ: `submission.json`).


## VIII. Kết quả (Results)

- Hiệu suất mô hình được đánh giá trên tập dữ liệu public_test và private_test bằng chỉ số **ST-IoU**:

| Model Configuration           | Public Score | Private Score |
| ----------------------------- | ------------ | ------------- |
| YOLO11n                       | 0.46140      | -             |
| YOLO11s                       | 0.50470      | -             |
| YOLO11n + Tracking            | 0.53050      | 0.21530       |
| **YOLO11s + Tracking (Best)** | **0.55460**  | **0.30200**   |

- Nhóm đạt được **hạng 53** trên **Public Leaderboard** với mô hình YOLO11s kết hợp Tracking.

## IX. Tham khảo (References)

1. [Zalo AI Challenge - AeroEyes Portal](https://challenge.zalo.ai/portal/aero-eyes)
2. [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
3. [Meta DINOv3](https://ai.meta.com/dinov3/)
4. [Daniel Gatis, rembg](https://github.com/danielgatis/rembg)