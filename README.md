# HỆ THỐNG KIỂM SOÁT CỬA TỰ ĐỘNG BẰNG AI & CẢM BIẾN TẦNG BIÊN (PI 4)
> **Dự án IoT Edge AI:** Nhận diện khuôn mặt thời gian thực & Chống giả mạo sinh trắc học kết hợp **Cảm biến Siêu âm HC-SR04 + Động cơ Servo (Mở 90 độ) + 2 Đèn LED (Đỏ / Xanh)** trên **Raspberry Pi 4**.

---

## 📋 ĐẶC TÍNH & NGUYÊN LÝ HOẠT ĐỘNG

1. **Cảm biến Siêu âm kích hoạt thông minh (Tiết kiệm CPU):**
   - Chỉ khi cảm biến siêu âm phát hiện có người / vật cản phía trước ($d \le 40\text{ cm}$) thì hệ thống mới kích hoạt camera dò tìm và nhận diện khuôn mặt.
   - Khi không có người: Hệ thống ở chế độ chờ (Standby), tắt đèn, đóng servo.
2. **Xử lý theo các điều kiện sinh trắc học:**
   - **Nhận diện ĐÚNG (Người thật + Có trong CSDL):** Quay động cơ **Servo mở 90 độ** trong 3 giây (tự động đóng lại về 0 độ), tắt đèn Đỏ và Xanh.
   - **Phát hiện GIẢ MẠO (Spoofing - Ảnh in, điện thoại, video):** Bật **Đèn ĐỎ**, giữ cửa đóng (Servo 0 độ).
   - **Người thật nhưng CHƯA CÓ TRONG CSDL (Người lạ / Unknown):** Bật **Đèn XANH**, giữ cửa đóng (Servo 0 độ).
3. **Phím tắt hỗ trợ:**
   - `[p]`: Bật/Tắt chế độ chống giả mạo (PAD).
   - `[o]`: Bật/Tắt mô phỏng vật cản siêu âm (tiện lợi khi test thử nghiệm).
   - `[q]`: Thoát chương trình.

---

## 🛠️ YÊU CẦU PHẦN CỨNG & SƠ ĐỒ ĐẤU DÂY GPIO

### 1. Danh mục linh kiện cần chuẩn bị:
1. **Board Raspberry Pi 4** (RAM 2GB, 4GB hoặc 8GB) cài Raspberry Pi OS 64-bit.
2. **1 USB Webcam**.
3. **1 Cảm biến siêu âm HC-SR04** (hoặc HC-SR04P).
4. **1 Động cơ Servo SG90 hoặc MG90S**.
5. **1 Đèn LED Đỏ + 1 Đèn LED Xanh**.
6. **2 Điện trở $220\Omega - 330\Omega$** (cho LED) + **Điện trở $1k\Omega$ và $2k\Omega$** (cầu phân áp cho chân ECHO của HC-SR04 nếu dùng loại 5V).
7. **Breadboard & Dây cắm Jumper Wires**.

---

### 2. Bảng Đấu Nối Chân GPIO (Chuẩn BCM):

| Thiết bị | Chân thiết bị | Chân GPIO Pi (BCM) | Chân Vật Lý (Pin) | Ghi chú đấu nối |
| :--- | :---: | :---: | :---: | :--- |
| **Cảm biến Siêu âm** | `VCC` | `5V` | Pin 2 hoặc 4 | Cấp nguồn 5V |
| | `GND` | `GND` | Pin 6 hoặc 9 | Nối Mass |
| | `TRIG` | `GPIO 23` | **Pin 16** | Tín hiệu phát siêu âm |
| | `ECHO` | `GPIO 24` | **Pin 18** | Tín hiệu thu (qua trở 1k-2k về GPIO) |
| **Động cơ Servo** | Dây Đỏ (VCC) | `5V` | Pin 2 hoặc 4 | Cấp nguồn 5V |
| | Dây Nâu/Đen (GND) | `GND` | Pin 14 hoặc 20 | Nối Mass |
| | Dây Cam/Vàng (PWM) | `GPIO 18` | **Pin 12** | Điều khiển góc quay 0° / 90° |
| **Đèn LED Đỏ** | Anode (+) | `GPIO 27` | **Pin 13** | Nối tiếp trở $220\Omega$ (Báo giả mạo) |
| | Cathode (-) | `GND` | GND chung | Nối Mass |
| **Đèn LED Xanh** | Anode (+) | `GPIO 17` | **Pin 11** | Nối tiếp trở $220\Omega$ (Báo người lạ) |
| | Cathode (-) | `GND` | GND chung | Nối Mass |

---

## 🚀 HƯỚNG DẪN CÀI ĐẶT & CHẠY

### Bước 1: Kích hoạt môi trường ảo
```bash
cd pi4-edge-face-recognition
source venv/bin/activate
```

### Bước 2: Kiểm tra riêng phần cứng ngoại vi
Chạy bài test kiểm tra cảm biến siêu âm, 2 đèn LED và động cơ servo quay 90 độ:
```bash
python -m peripherals.access_controller
```

### Bước 3: Đăng ký khuôn mặt người dùng vào Database
```bash
python enroll.py --name "Nguyen Van A"
```
*(Nhìn thẳng vào camera và nhấn phím **`s`** để lưu đặc trưng vào `faces.db`)*

### Bước 4: Khởi chạy hệ thống kiểm soát cửa thông minh
```bash
python run.py
```

* Nếu test trên máy tính chưa gắn cảm biến: Nhấn phím **`o`** để bật/tắt giả lập có vật cản phía trước.
