"""
Module Điều Khiển Ma Trận 10 LED (2 hàng x 5 cột) Trên Raspberry Pi 4
Sử dụng thư viện chuẩn gpiozero.

Sơ đồ phân bổ 5 Cột:
- Hàng 1 (Đỏ) : [RED_1, RED_2, RED_3, RED_4, RED_5]
- Hàng 2 (Xanh): [GREEN_1, GREEN_2, GREEN_3, GREEN_4, GREEN_5]
"""

import os
import sys
import time
from typing import Dict, List, Optional

# Cố gắng import gpiozero (hoạt động trực tiếp trên Raspberry Pi)
try:
    from gpiozero import LED
    HAS_GPIO = True
except (ImportError, Exception):
    LED = None
    HAS_GPIO = False


class LEDMatrix:
    """
    Quản lý trực tiếp 10 chân GPIO điều khiển ma trận 2 hàng x 5 cột bằng gpiozero.
    """

    def __init__(
        self,
        red_pins: Optional[List[int]] = None,
        green_pins: Optional[List[int]] = None,
    ):
        if red_pins is None:
            pins_str = os.getenv("LED_RED_PINS", "27,22,23,25,16")
            red_pins = [int(p.strip()) for p in pins_str.split(",") if p.strip()]

        if green_pins is None:
            pins_str = os.getenv("LED_GREEN_PINS", "17,18,24,12,26")
            green_pins = [int(p.strip()) for p in pins_str.split(",") if p.strip()]

        self.red_pins = red_pins[:5]
        self.green_pins = green_pins[:5]

        self.red_leds = []
        self.green_leds = []
        self.is_active = False

        if HAS_GPIO:
            try:
                for pin in self.red_pins:
                    self.red_leds.append(LED(pin))
                for pin in self.green_pins:
                    self.green_leds.append(LED(pin))
                self.is_active = True
                self.reset_all()
            except Exception as e:
                print(f"[LED Matrix] Cảnh báo: Không thể khởi tạo chân GPIO qua gpiozero: {e}")
                self.is_active = False
        else:
            print("[LED Matrix] gpiozero không khả dụng (môi trường không phải Pi OS hoặc thiếu quyền GPIO).")

    def reset_all(self):
        """Tắt toàn bộ 10 đèn LED."""
        if not self.is_active:
            return
        for led in self.red_leds:
            led.off()
        for led in self.green_leds:
            led.off()

    def set_no_pad_mode(self, face_count: int):
        """
        Chế độ KHÔNG PAD:
        - Tắt toàn bộ 5 đèn Đỏ.
        - Bật sáng K đèn Xanh tương ứng với số khuôn mặt phát hiện được (tối đa 5).
        """
        if not self.is_active:
            return

        # 1. Tắt tất cả LED Đỏ
        for led in self.red_leds:
            led.off()

        # 2. Bật số lượng LED Xanh tương ứng
        clamped_count = min(max(face_count, 0), 5)
        for i, led in enumerate(self.green_leds):
            if i < clamped_count:
                led.on()
            else:
                led.off()

    def set_pad_mode(self, slot_statuses: Dict[int, str]):
        """
        Chế độ CÓ PAD (5 cột):
        slot_statuses: Dict mapping index cột (0..4) -> trạng thái:
        - 'SPOOF'     : LED Đỏ sáng, LED Xanh tắt
        - 'REAL_KNOWN': LED Xanh sáng, LED Đỏ tắt
        - 'UNKNOWN'   : Cả 2 LED Đỏ và Xanh cùng sáng
        - 'EMPTY'     : Cả 2 LED tắt
        """
        if not self.is_active:
            return

        for col in range(5):
            status = slot_statuses.get(col, "EMPTY")

            red_led = self.red_leds[col]
            green_led = self.green_leds[col]

            if status == "SPOOF":
                red_led.on()
                green_led.off()
            elif status == "REAL_KNOWN":
                red_led.off()
                green_led.on()
            elif status == "UNKNOWN":
                red_led.on()
                green_led.on()
            else:  # 'EMPTY' hoặc trạng thái khác
                red_led.off()
                green_led.off()

    def close(self):
        """Dập tắt và giải phóng tài nguyên GPIO."""
        if self.is_active:
            self.reset_all()
            for led in self.red_leds + self.green_leds:
                try:
                    led.close()
                except Exception:
                    pass
            self.is_active = False


if __name__ == "__main__":
    print("=" * 60)
    print(" BÀI TEST PHẦN CỨNG MA TRẬN 10 LED (GPIOZERO)")
    print("=" * 60)
    matrix = LEDMatrix()
    if not matrix.is_active:
        print("Lỗi: Không thể kết nối với phần cứng GPIO. Vui lòng kiểm tra lại trên Raspberry Pi!")
        sys.exit(1)

    print(f"Chân LED Đỏ : {matrix.red_pins}")
    print(f"Chân LED Xanh: {matrix.green_pins}")
    print("\n[Bước 1] Test chớp nháy tuần tự 5 Cột (Đỏ -> Xanh -> Cả hai)...")

    try:
        for col in range(5):
            print(f"  -> Cột {col + 1}: Bật ĐỎ (SPOOF test)")
            matrix.set_pad_mode({col: "SPOOF"})
            time.sleep(0.4)

            print(f"  -> Cột {col + 1}: Bật XANH (REAL_KNOWN test)")
            matrix.set_pad_mode({col: "REAL_KNOWN"})
            time.sleep(0.4)

            print(f"  -> Cột {col + 1}: Bật CẢ HAI (UNKNOWN test)")
            matrix.set_pad_mode({col: "UNKNOWN"})
            time.sleep(0.4)

            matrix.reset_all()
            time.sleep(0.2)

        print("\n[Bước 2] Test Chế độ KHÔNG PAD: Tăng dần số khuôn mặt (1 -> 5)...")
        for count in range(1, 6):
            print(f"  -> Đếm {count} khuôn mặt: Bật {count} LED Xanh")
            matrix.set_no_pad_mode(count)
            time.sleep(0.5)

        print("\n[OK] Test hoàn tất! Toàn bộ 10 đèn LED hoạt động chuẩn xác.")
    finally:
        matrix.close()
