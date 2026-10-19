"""
Module Điều Khiển Phần Cứng: Cảm Biến Siêu Âm + Động Cơ Servo + 2 Đèn LED (Đỏ / Xanh)
Dành cho Hệ thống Kiểm soát Vào/Ra Tự động (Edge AI Face Recognition & Access Control)
Sử dụng thư viện gpiozero trên Raspberry Pi 4.

Các thành phần:
1. Cảm biến siêu âm HC-SR04: Đo khoảng cách phát hiện vật cản phía trước.
2. Đèn LED Đỏ: Báo động khi phát hiện GIẢ MẠO (Spoofing: ảnh in, điện thoại,...).
3. Đèn LED Xanh: Báo hiệu NGƯỜI THẬT nhưng KHÔNG CÓ trong CSDL (Người lạ / Unknown).
4. Động cơ Servo (SG90/MG90S): Mở 90 độ khi NHẬN DIỆN ĐÚNG người thật đã đăng ký trong CSDL.
"""

import os
import sys
import time
from typing import Optional

try:
    from gpiozero import LED, DistanceSensor, AngularServo, Servo
    from gpiozero.pins.native import NativeFactory
    HAS_GPIO = True
except (ImportError, Exception):
    LED = None
    DistanceSensor = None
    AngularServo = None
    Servo = None
    HAS_GPIO = False


class AccessController:
    """
    Quản lý toàn bộ ngoại vi điều khiển vào/ra:
    - Cảm biến siêu âm HC-SR04 (Trig, Echo)
    - 2 LED tín hiệu (Đỏ, Xanh)
    - Động cơ Servo (0 độ: Đóng, 90 độ: Mở)
    """

    def __init__(
        self,
        trig_pin: Optional[int] = None,
        echo_pin: Optional[int] = None,
        servo_pin: Optional[int] = None,
        red_pin: Optional[int] = None,
        green_pin: Optional[int] = None,
        obstacle_distance_cm: Optional[float] = None,
        door_hold_seconds: Optional[float] = None,
    ):
        # 1. Đọc chân GPIO từ tham số hoặc biến môi trường .env
        self.trig_pin = int(trig_pin or os.getenv("ULTRASONIC_TRIG_PIN", "23"))
        self.echo_pin = int(echo_pin or os.getenv("ULTRASONIC_ECHO_PIN", "24"))
        self.servo_pin = int(servo_pin or os.getenv("SERVO_PIN", "18"))
        self.red_pin = int(red_pin or os.getenv("LED_RED_PIN", "27"))
        self.green_pin = int(green_pin or os.getenv("LED_GREEN_PIN", "17"))

        self.obstacle_distance_cm = float(obstacle_distance_cm or os.getenv("OBSTACLE_DISTANCE_CM", "40.0"))
        self.door_hold_seconds = float(door_hold_seconds or os.getenv("DOOR_HOLD_SECONDS", "3.0"))

        self.is_active = False
        self.ultrasonic = None
        self.red_led = None
        self.green_led = None
        self.servo = None

        # Trạng thái hiện tại
        self.is_door_open = False
        self.door_opened_time = 0.0
        self.current_state = "IDLE"  # "IDLE", "SPOOF", "UNKNOWN", "AUTHORIZED"
        self.simulated_obstacle = False  # Dùng khi test mô phỏng trên máy không có GPIO

        # 2. Khởi tạo thiết bị phần cứng nếu có gpiozero
        if HAS_GPIO:
            try:
                # LED Đỏ & Xanh
                self.red_led = LED(self.red_pin)
                self.green_led = LED(self.green_pin)

                # Cảm biến siêu âm HC-SR04 (ngưỡng tối đa 2 mét)
                try:
                    self.ultrasonic = DistanceSensor(
                        echo=self.echo_pin,
                        trigger=self.trig_pin,
                        max_distance=2.0,
                        threshold_distance=self.obstacle_distance_cm / 100.0,
                    )
                except Exception as ex_dist:
                    print(f"[AccessController] Cảnh báo cảm biến siêu âm: {ex_dist}")
                    self.ultrasonic = None

                # Động cơ Servo (SG90 thường hoạt động xung 0.5ms - 2.5ms)
                try:
                    self.servo = AngularServo(
                        self.servo_pin,
                        min_angle=0,
                        max_angle=180,
                        min_pulse_width=0.0005,
                        max_pulse_width=0.0025,
                    )
                    self.servo.angle = 0
                except Exception:
                    # Thử fallback sang Servo thường
                    try:
                        self.servo = Servo(
                            self.servo_pin,
                            min_pulse_width=0.0005,
                            max_pulse_width=0.0025,
                        )
                        self.servo.min()  # Vị trí đóng ban đầu
                    except Exception as ex_servo:
                        print(f"[AccessController] Cảnh báo Servo: {ex_servo}")
                        self.servo = None

                self.is_active = True
                self.reset_all()
                print(f"[AccessController] Đã kết nối phần cứng:")
                print(f"  - Cảm biến siêu âm : TRIG=GPIO{self.trig_pin}, ECHO=GPIO{self.echo_pin} (Ngưỡng: <= {self.obstacle_distance_cm} cm)")
                print(f"  - Đèn LED Đỏ      : GPIO{self.red_pin}")
                print(f"  - Đèn LED Xanh    : GPIO{self.green_pin}")
                print(f"  - Động cơ Servo   : GPIO{self.servo_pin} (Mở 90 độ, giữ {self.door_hold_seconds}s)")
            except Exception as e:
                print(f"[AccessController] Không thể khởi tạo GPIO: {e}")
                self.is_active = False
        else:
            print("[AccessController] gpiozero không khả dụng (Đang chạy ở Chế độ Mô Phỏng Simulation).")

    def get_distance_cm(self) -> float:
        """
        Đọc khoảng cách từ cảm biến siêu âm (đơn vị: cm).
        Nếu không có phần cứng thực tế, trả về giá trị mô phỏng.
        """
        if self.is_active and self.ultrasonic is not None:
            try:
                # distance trả về đơn vị mét (0.0 đến 2.0)
                dist_m = self.ultrasonic.distance
                return round(dist_m * 100.0, 1)
            except Exception:
                return 999.0
        else:
            # Mô phỏng: Nếu bật cờ simulated_obstacle thì trả về 20cm, ngược lại 150cm
            return 20.0 if self.simulated_obstacle else 150.0

    def has_obstacle(self) -> bool:
        """Kiểm tra có vật cản phía trước trong phạm vi ngưỡng không."""
        dist = self.get_distance_cm()
        return dist <= self.obstacle_distance_cm

    def set_spoof(self):
        """
        Khuôn mặt GIẢ MẠO (Ảnh/Màn hình):
        - Bật đèn ĐỎ.
        - Tắt đèn Xanh.
        - Đóng Servo (0 độ).
        """
        self.current_state = "SPOOF"
        if self.is_active:
            if self.red_led:
                self.red_led.on()
            if self.green_led:
                self.green_led.off()
            self._set_servo_angle(0)
        self.is_door_open = False

    def set_unknown(self):
        """
        Người thật nhưng KHÔNG CÓ trong cơ sở dữ liệu:
        - Bật đèn XANH.
        - Tắt đèn Đỏ.
        - Đóng Servo (0 độ).
        """
        self.current_state = "UNKNOWN"
        if self.is_active:
            if self.green_led:
                self.green_led.on()
            if self.red_led:
                self.red_led.off()
            self._set_servo_angle(0)
        self.is_door_open = False

    def set_authorized_open(self):
        """
        Người thật ĐÃ ĐĂNG KÝ trong cơ sở dữ liệu (Nhận diện ĐÚNG):
        - Mở Servo 90 độ.
        - Tắt đèn Đỏ.
        - Tắt đèn Xanh (hoặc tắt cả 2 đèn).
        - Ghi nhận thời gian mở cửa để tự động đóng sau hold_seconds.
        """
        self.current_state = "AUTHORIZED"
        if self.is_active:
            if self.red_led:
                self.red_led.off()
            if self.green_led:
                self.green_led.off()
            self._set_servo_angle(90)

        self.is_door_open = True
        self.door_opened_time = time.time()

    def reset_leds(self):
        """Tắt cả 2 đèn LED."""
        if self.is_active:
            if self.red_led:
                self.red_led.off()
            if self.green_led:
                self.green_led.off()

    def reset_all(self):
        """Tắt 2 LED và đóng Servo về 0 độ."""
        self.current_state = "IDLE"
        self.reset_leds()
        self._set_servo_angle(0)
        self.is_door_open = False

    def update(self):
        """
        Cập nhật chu kỳ: Tự động đóng cửa (quay servo về 0 độ)
        sau khi đã mở đủ thời gian quy định (door_hold_seconds).
        """
        if self.is_door_open:
            elapsed = time.time() - self.door_opened_time
            if elapsed >= self.door_hold_seconds:
                self._set_servo_angle(0)
                self.is_door_open = False
                self.current_state = "IDLE"
                self.reset_leds()

    def _set_servo_angle(self, angle: int):
        """Hàm nội bộ điều khiển góc quay của servo (0 đến 90 độ)."""
        if not self.is_active or self.servo is None:
            return
        try:
            if hasattr(self.servo, "angle"):
                self.servo.angle = angle
            else:
                # Nếu là Servo thường (-1 đến 1):
                # 0 độ tương ứng min (-1), 90 độ tương ứng mid (0)
                if angle == 0:
                    self.servo.min()
                elif angle == 90:
                    self.servo.mid()
                elif angle >= 180:
                    self.servo.max()
        except Exception as e:
            pass

    def close(self):
        """Giải phóng toàn bộ tài nguyên GPIO khi thoát chương trình."""
        self.reset_all()
        if self.is_active:
            for dev in [self.red_led, self.green_led, self.servo, self.ultrasonic]:
                if dev is not None:
                    try:
                        dev.close()
                    except Exception:
                        pass
            self.is_active = False


if __name__ == "__main__":
    print("=" * 65)
    print(" BÀI TEST TỔNG HỢP NGOẠI VI: SIÊU ÂM + 2 LED + SERVO")
    print("=" * 65)
    ctrl = AccessController()

    try:
        print("\n[Bước 1] Kiểm tra cảm biến siêu âm trong 5 giây...")
        for _ in range(10):
            d = ctrl.get_distance_cm()
            obstacle = "CÓ VẬT CẢN" if ctrl.has_obstacle() else "KHÔNG CÓ VẬT CẢN"
            print(f"  -> Khoảng cách đo được: {d:5.1f} cm | {obstacle}")
            time.sleep(0.5)

        print("\n[Bước 2] Kiểm tra Đèn ĐỎ (Mô phỏng GIẢ MẠO - SPOOF)...")
        ctrl.set_spoof()
        print("  -> Đèn ĐỎ đang BẬT, Đèn Xanh TẮT, Servo 0 độ.")
        time.sleep(2)

        print("\n[Bước 3] Kiểm tra Đèn XANH (Mô phỏng NGƯỜI LẠ - UNKNOWN)...")
        ctrl.set_unknown()
        print("  -> Đèn XANH đang BẬT, Đèn Đỏ TẮT, Servo 0 độ.")
        time.sleep(2)

        print("\n[Bước 4] Kiểm tra MỞ CỬA (Mô phỏng NHẬN DIỆN ĐÚNG - QUAY SERVO 90 ĐỘ)...")
        ctrl.set_authorized_open()
        print(f"  -> Đã mở Servo 90 độ! Giữ trong {ctrl.door_hold_seconds} giây...")
        while ctrl.is_door_open:
            ctrl.update()
            time.sleep(0.1)
        print("  -> Đã tự động đóng Servo về 0 độ và dập tắt đèn.")

        print("\n[OK] Bài test phần cứng hoàn tất thành công!")
    finally:
        ctrl.close()
