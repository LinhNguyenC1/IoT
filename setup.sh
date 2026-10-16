#!/usr/bin/env bash
# ====================================================================
# SCRIPT CÀI ĐẶT TỰ ĐỘNG DỰ ÁN PI 4 EDGE FACE RECOGNITION (1-CLICK SETUP)
# ====================================================================

set -e

echo "=================================================================="
echo " KHỞI TẠO MÔI TRƯỜNG DỰ ÁN PI 4 EDGE FACE RECOGNITION"
echo "=================================================================="

# 1. Cài đặt các thư viện hệ thống cần thiết cho OpenCV và GPIO
echo "\n[1/4] Cập nhật hệ thống và cài đặt gói bổ trợ..."
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0

# 2. Tạo môi trường ảo Python venv
echo "\n[2/4] Tạo môi trường ảo Python (venv)..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  -> Đã tạo thư mục venv thành công."
else
    echo "  -> Môi trường venv đã tồn tại, tiếp tục..."
fi

# 3. Kích hoạt venv và cài đặt thư viện Python
echo "\n[3/4] Cài đặt các thư viện Python từ requirements.txt..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Tạo file cấu hình .env từ .env.example nếu chưa có
echo "\n[4/4] Kiểm tra file cấu hình .env..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  -> Đã tạo file .env từ .env.example."
else
    echo "  -> File .env đã tồn tại sẵn."
fi

echo "\n=================================================================="
echo " [THÀNH CÔNG] HỆ THỐNG ĐÃ SẴN SÀNG HOẠT ĐỘNG!"
echo "=================================================================="
echo "Các bước tiếp theo dành cho bạn:"
echo " 1. Kích hoạt môi trường:  source venv/bin/activate"
echo " 2. Test phần cứng 10 LED: python -m peripherals.led_matrix"
echo " 3. Đăng ký khuôn mặt:     python enroll.py --name \"Ho Va Ten\""
echo " 4. Chạy nhận diện:        python run.py"
echo "==================================================================\n"
