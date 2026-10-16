# HỆ THỐNG NHẬN DIỆN KHUÔN MẶT & CHỐNG GIẢ MẠO TẦNG BIÊN (PI 4)
> **Dự án IoT Edge AI:** Nhận diện khuôn mặt thời gian thực & Chống giả mạo sinh trắc học (Liveness Detection) kết hợp Bảng điều khiển phần cứng **Ma trận 10 LED (2 hàng x 5 cột)** trên **Raspberry Pi 4 + USB Webcam**.

---

## ĐẶC ĐIỂM

- **Xử lý cục bộ:** Chạy trực tiếp trên Raspberry Pi 4, không kết nối server bên ngoài.
- **Lưu trữ dữ liệu:** Sử dụng SQLite (`faces.db`) để lưu trữ thông tin người dùng và vector đặc trưng 512 chiều trích xuất từ ArcFace MobileFaceNet.
- **Hai chế độ hoạt động:**
  - *Chế độ 1 (KHÔNG PAD):* Bỏ qua bước kiểm tra liveness để tối ưu tốc độ xử lý; các đèn LED Xanh sáng theo số lượng khuôn mặt phát hiện được.
  - *Chế độ 2 (CÓ PAD):* Tích hợp mô hình MiniFASNetV2 để phân biệt người thật và ảnh/màn hình giả mạo; giới hạn xử lý tối đa 5 khuôn mặt tương ứng 5 cột LED.
  - *Chuyển đổi chế độ:* Phím `p` trên bàn phím, cờ dòng lệnh CLI (`--pad` / `--no-pad`), hoặc cấu hình trong `.env`.
- **Chỉ thị trạng thái bằng 10 LED (gpiozero):** Bố trí 2 hàng x 5 cột (5 Đỏ, 5 Xanh) hiển thị trạng thái từng khuôn mặt (Spoof: Đỏ, Real: Xanh, Unknown: Đỏ + Xanh).
- **Đăng ký khuôn mặt (`enroll.py`):** Hỗ trợ chụp mẫu qua webcam (chụp 5 frame tính vector trung bình) hoặc quét từ thư mục ảnh.

---

## 🛠️ YÊU CẦU PHẦN CỨNG & SƠ ĐỒ ĐẤU DÂY GPIO

### 1. Danh mục linh kiện:
1. **Board Raspberry Pi 4** (RAM 2GB, 4GB hoặc 8GB) đã cài Raspberry Pi OS 64-bit.
2. **1 USB Webcam tiêu chuẩn** (cắm cổng USB 2.0 hoặc 3.0).
3. **5 Đèn LED Đỏ + 5 Đèn LED Xanh lá**.
4. **10 Điện trở** ($220\Omega$ đến $330\Omega$) và Dây cắm Breadboard (Jumper Wires).

---

### 2. Sơ đồ Ma trận 10 LED (2 hàng x 5 cột):

```text
         Cột 1       Cột 2       Cột 3       Cột 4       Cột 5
       (Slot 1)    (Slot 2)    (Slot 3)    (Slot 4)    (Slot 5)
     ┌───────────┬───────────┬───────────┬───────────┬───────────┐
  🔴 │ LED Đỏ 1  │ LED Đỏ 2  │ LED Đỏ 3  │ LED Đỏ 4  │ LED Đỏ 5  │  <- Hàng 1 (Đỏ)
  🟢 │ LED Xanh 1│ LED Xanh 2│ LED Xanh 3│ LED Xanh 4│ LED Xanh 5│  <- Hàng 2 (Xanh)
     └───────────┴───────────┴───────────┴───────────┴───────────┘
```

### 3. Bảng Đấu Nối Chân GPIO (Chuẩn BCM):

| Tên Đèn | Cột | Chân GPIO (BCM) | Chân Vật Lý (Header 40-pin) | Ghi Chú |
| :--- | :---: | :---: | :---: | :--- |
| **RED_1** | Cột 1 | `GPIO 27` | Pin 13 | Đỏ Cột 1 |
| **GREEN_1**| Cột 1 | `GPIO 17` | Pin 11 | Xanh Cột 1 |
| **RED_2** | Cột 2 | `GPIO 22` | Pin 15 | Đỏ Cột 2 |
| **GREEN_2**| Cột 2 | `GPIO 18` | Pin 12 | Xanh Cột 2 |
| **RED_3** | Cột 3 | `GPIO 23` | Pin 16 | Đỏ Cột 3 |
| **GREEN_3**| Cột 3 | `GPIO 24` | Pin 18 | Xanh Cột 3 |
| **RED_4** | Cột 4 | `GPIO 25` | Pin 22 | Đỏ Cột 4 |
| **GREEN_4**| Cột 4 | `GPIO 12` | Pin 32 | Xanh Cột 4 |
| **RED_5** | Cột 5 | `GPIO 16` | Pin 36 | Đỏ Cột 5 |
| **GREEN_5**| Cột 5 | `GPIO 26` | Pin 37 | Xanh Cột 5 |
| **GND** | Chung | `GND` | Pin 6, 9, 14, 20, 25, 30, 34, 39 | Cực âm chung (nối qua trở) |

*(Lưu ý: Có thể thay đổi các chân GPIO này bất kỳ lúc nào trong file cấu hình `.env`).*

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT & CHẠY NHANH (QUICKSTART)

### Bước 1: Clone kho mã nguồn về Raspberry Pi
Mở Terminal trên Pi và chạy lệnh:
```bash
git clone https://github.com/<tai_khoan_cua_ban>/pi4-edge-face-recognition.git
cd pi4-edge-face-recognition
```

### Bước 2: Cài đặt môi trường tự động (1-Click Setup)
Chạy script cài đặt tự động:
```bash
chmod +x setup.sh
./setup.sh
```
Script sẽ tự động tạo môi trường ảo Python `venv`, cài các thư viện `onnxruntime`, `opencv`, `gpiozero`, và tạo sẵn file cấu hình `.env`.

### Bước 3: Kích hoạt môi trường ảo & Kiểm tra đèn LED
```bash
source venv/bin/activate

# Chạy bài test kiểm tra chớp nháy tuần tự 10 đèn LED:
python -m peripherals.led_matrix
```

### Bước 4: Đăng ký khuôn mặt mẫu
Đứng trước webcam và đăng ký tên của bạn:
```bash
python enroll.py --name "Nguyen Van A"
```
*Nhìn thẳng vào camera và nhấn phím **`s`** để hệ thống tự động chụp 5 góc mặt và lưu vào database SQLite `faces.db`.*

### Bước 5: Khởi chạy hệ thống nhận diện!
```bash
python run.py
```
* Cửa sổ camera sẽ hiển thị lên màn hình.
* Đứng trước camera:
  * Nếu là bạn: Khung xanh lá + hiển thị Tên và % tương đồng. Cột LED tương ứng **sáng đèn XANH**.
  * Nếu người lạ: Khung cam + hiển thị `Unknown`. Cột LED tương ứng **sáng cả ĐỎ và XANH**.
  * Nếu giơ ảnh in / màn hình điện thoại: Khung đỏ + cảnh báo `SPOOF`. Cột LED tương ứng **sáng đèn ĐỎ**!
* **Nhấn phím `p`:** Chuyển ngay sang chế độ **KHÔNG PAD** (FPS tăng lên 25 FPS, các đèn xanh sáng đếm số người).
* **Nhấn phím `q`:** Thoát chương trình.

---

## ⌨️ CÁC TÙY CHỌN DÒNG LỆNH HỮU ÍCH

### 1. Quản lý người dùng trong Database (`enroll.py`):
```bash
# Đăng ký trực tiếp qua Webcam (chương trình sẽ hỏi Tên -> mở camera chụp):
python enroll.py

# Đăng ký nhanh qua Webcam với tên truyền sẵn:
python enroll.py --name "Nguyen Van A"

# Nạp hàng loạt từ thư mục ảnh faces/<Ten_Nguoi>/anh.jpg:
python enroll.py --folder faces/

# Xem danh sách tất cả những ai đã đăng ký:
python enroll.py --list

# Xóa một người dùng khỏi database:
python enroll.py --delete "Nguyen Van A"
```

### 2. Khởi chạy với các cờ tùy chọn (`run.py`):
```bash
# Chạy ép buộc chế độ CÓ PAD ngay từ đầu:
python run.py --pad

# Chạy ép buộc chế độ KHÔNG PAD:
python run.py --no-pad

# Chạy qua SSH không cần màn hình desktop (Headless):
python run.py --headless

# Đổi cổng camera (nếu cắm nhiều webcam):
python run.py --camera 1
```

---

## 📂 CẤU TRÚC THƯ MỤC DỰ ÁN

```text
pi4-edge-face-recognition/
├── database/
│   ├── __init__.py
│   └── local_db.py              # SQLite: lưu users, embeddings 512D; Cosine matching
├── models/                      # Đóng gói 3 weights ONNX nhẹ (~17.8 MB)
│   ├── detector_scrfd.onnx      # InsightFace SCRFD (~2.5 MB)
│   ├── antispoof_minifasnet.onnx# MiniFASNet R7 (~1.7 MB - chống giả mạo)
│   └── recognition_arcface.onnx # ArcFace MobileFaceNet 512D (~13.6 MB)
├── detection/                   # Wrapper bộ dò InsightFace SCRFD
├── tracking/                    # Lucas-Kanade Optical Flow Cadence Tracker
├── antispoof/                   # MiniFASNet Predictor & Temporal Voting
├── alignment/                   # 5-point landmark affine aligner
├── recognition/                 # ArcFace Feature Embedder
├── peripherals/
│   ├── __init__.py
│   └── led_matrix.py            # Quản lý 10 LED qua gpiozero (5 đỏ, 5 xanh)
├── faces/                       # Thư mục lưu ảnh mẫu nạp theo từng người
├── enroll.py                    # Script đăng ký khuôn mặt (Webcam phím 's' hoặc Folder)
├── run.py                       # Entrypoint chính của hệ thống
├── setup.sh                     # Script 1-click cài đặt môi trường
├── requirements.txt             # Thư viện Python phụ thuộc
├── .env.example                 # File mẫu cấu hình hệ thống & chân GPIO
├── .gitignore                   # Loại trừ file tạm, database và venv
└── README.md                    # Tài liệu hướng dẫn A-Z
```

---

## ❓ CÁC CÂU HỎI THƯỜNG GẶP (TROUBLESHOOTING)

1. **Lỗi `Không thể kết nối với Webcam tại /dev/video0`:**
   * Kiểm tra xem cáp USB Webcam đã cắm chặt vào Pi chưa bằng lệnh: `ls /dev/video*`.
   * Nếu webcam nằm ở `/dev/video1` hoặc `/dev/video2`, hãy mở file `.env` và sửa `CAMERA_INDEX=1` (hoặc chạy `python run.py --camera 1`).

2. **Lỗi `gpiozero.exc.BadPinFactory: Unable to load any default pin factory`:**
   * Hãy đảm bảo bạn đã cài đặt gói `lgpio` bằng lệnh:
     ```bash
     pip install lgpio
     # hoặc trên Pi OS: sudo apt install -y python3-gpiozero python3-lgpio
     ```

3. **Cơ chế hoạt động & Bộ dò:**
   * Hệ thống chuẩn hóa sử dụng **InsightFace SCRFD** (`detector_scrfd.onnx` ~2.5 MB) để trích xuất 5 điểm mốc (landmarks) chuẩn xác cho ArcFace và chống rung hình khi theo dõi khuôn mặt.
   * Nhấn phím `p` trên bàn phím để chuyển đổi giữa 2 chế độ: **CÓ PAD** và **KHÔNG PAD**.

---
*Chúc các bạn hoàn thành bài tập thực hành IoT xuất sắc!*
