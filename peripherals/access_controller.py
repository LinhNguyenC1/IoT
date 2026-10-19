"""
Module Điều Khiển Phần Cứng: Cảm Biến Siêu Âm + Động Cơ Servo + 2 Đèn LED (Đỏ / Xanh)
Dành cho Hệ thống Kiểm soát Vào/Ra Tự động (Edge AI Face Recognition & Access Control)
Hỗ trợ cả pigpio (xung mượt) và gpiozero (dự phòng tự động nếu chưa bật pigpiod).

Các thành phần:
1. Cảm biến siêu âm HC-SR04: Đo khoảng cách phát hiện vật cản phía trước.
2. Đèn LED Đỏ: Báo động khi phát hiện GIẢ MẠO (Spoofing: ảnh in, điện thoại,...).
3. Đèn LED Xanh: Báo hiệu NGƯỜI THẬT nhưng KHÔNG CÓ trong CSDL (Người lạ / Unknown).
4. Động cơ Servo (SG90/MG90S): Mở 90 độ khi NHẬN DIỆN ĐÚNG người thật đã đăng ký trong CSDL.
"""

import os
import time
from typing import Optional

try:
    from gpiozero import LED, DistanceSensor, AngularServo
    HAS_GPIO = True
except (ImportError, Exception):
    LED = None
    DistanceSensor = None
    AngularServo = None
    HAS_GPIO = False

try:
    import pigpio
except ImportError:
    pigpio = None


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
        # 1. Đọc chân GPIO từ tham số hoặc biến môi trường .env (Mặc định chuẩn BCM)
        self.trig_pin = int(trig_pin if trig_pin is not None else os.getenv("ULTRASONIC_TRIG_PIN", "23"))
        self.echo_pin = int(echo_pin if echo_pin is not None else os.getenv("ULTRASONIC_ECHO_PIN", "24"))
        self.servo_pin = int(servo_pin if servo_pin is not None else os.getenv("SERVO_PIN", "18"))
        self.red_pin = int(red_pin if red_pin is not None else os.getenv("LED_RED_PIN", "27"))
        self.green_pin = int(green_pin if green_pin is not None else os.getenv("LED_GREEN_PIN", "17"))

        self.obstacle_distance_cm = float(
            obstacle_distance_cm if obstacle_distance_cm is not None else os.getenv("OBSTACLE_DISTANCE_CM", "40.0")
        )
        self.door_hold_seconds = float(
            door_hold_seconds if door_hold_seconds is not None else os.getenv("DOOR_HOLD_SECONDS", "3.0")
        )

        self.is_active = False
        self.ultrasonic = None
        self.red_led = None
        self.green_led = None
        self.servo_gpio = None       # Dùng nếu có pigpio
        self.servo_fallback = None   # Dùng gpiozero nếu chưa chạy pigpiod

        # Trạng thái hệ thống
        self.is_door_open = False
        self.door_opened_time = 0.0
        self.current_state = "IDLE"  # "IDLE", "SPOOF", "UNKNOWN", "AUTHORIZED"
        self.authorized_latched = False
        self.servo_angle = None
        self.servo_error_reported = False
        self.simulated_obstacle = False  # Dùng khi test mô phỏng trên máy tính
        self.distance_error_reported = False
        self.last_distance_cm = None
        self.last_distance_log_time = 0.0

        # 2. Khởi tạo thiết bị phần cứng
        if HAS_GPIO:
            try:
                # Khởi tạo 2 LED (Đỏ & Xanh)
                self.red_led = LED(self.red_pin)
                self.green_led = LED(self.green_pin)

                # Khởi tạo Cảm biến siêu âm HC-SR04
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

                # Khởi tạo Servo: Ưu tiên pigpio, tự động fallback sang gpiozero
                servo_mode_desc = "Chưa kết nối"
                if pigpio is not None:
                    try:
                        self.servo_gpio = pigpio.pi()
                        if self.servo_gpio.connected:
                            self.servo_gpio.set_mode(self.servo_pin, pigpio.OUTPUT)
                            servo_mode_desc = "pigpio (ổn định cao)"
                        else:
                            self.servo_gpio.stop()
                            self.servo_gpio = None
                    except Exception:
                        self.servo_gpio = None

                # Nếu pigpio chưa bật daemon, tự động dùng gpiozero.AngularServo
                if self.servo_gpio is None:
                    try:
                        self.servo_fallback = AngularServo(
                            self.servo_pin,
                            min_angle=0,
                            max_angle=180,
                            min_pulse_width=0.0005,
                            max_pulse_width=0.0025,
                        )
                        servo_mode_desc = "gpiozero.AngularServo (Dự phòng)"
                    except Exception as ex_fallback:
                        print(f"[AccessController] Không thể khởi tạo servo dự phòng: {ex_fallback}")
                        self.servo_fallback = None

                self.is_active = True
                self.reset_all()
                print(f"[AccessController] Đã kết nối phần cứng:")
                print(f"  - Siêu âm HC-SR04 : TRIG=GPIO{self.trig_pin}, ECHO=GPIO{self.echo_pin} (Ngưỡng <= {self.obstacle_distance_cm} cm)")
                print(f"  - Đèn LED Đỏ      : GPIO{self.red_pin}")
                print(f"  - Đèn LED Xanh    : GPIO{self.green_pin}")
                print(f"  - Động cơ Servo   : GPIO{self.servo_pin} [{servo_mode_desc}] (Mở 90°, giữ {self.door_hold_seconds}s)")
            except Exception as e:
                print(f"[AccessController] Lỗi khởi tạo GPIO: {e}")
                self.is_active = False
        else:
            print("[AccessController] Đang chạy ở Chế độ Mô Phỏng (Simulation).")

    def get_distance_cm(self) -> float:
        """Đọc khoảng cách từ cảm biến siêu âm (cm)."""
        if self.is_active and self.ultrasonic is not None:
            try:
                dist_m = self.ultrasonic.distance
                self.distance_error_reported = False
                return round(dist_m * 100.0, 1)
            except Exception as error:
                if not self.distance_error_reported:
                    print(f"[AccessController] Lỗi đọc HC-SR04: {error}")
                    self.distance_error_reported = True
                return 999.0
        else:
            # Mô phỏng: Nếu bật cờ simulated_obstacle thì trả về 20cm, ngược lại 150cm
            return 20.0 if self.simulated_obstacle else 150.0

    def has_obstacle(self) -> bool:
        """Kiểm tra có vật cản trong phạm vi ngưỡng không."""
        self.last_distance_cm = self.get_distance_cm()
        obstacle = self.last_distance_cm <= self.obstacle_distance_cm
        now = time.time()
        if now - self.last_distance_log_time >= 1.0:
            state = "CÓ VẬT CẢN" if obstacle else "KHÔNG CÓ VẬT CẢN"
            print(f"[HC-SR04] Khoảng cách: {self.last_distance_cm:.1f} cm | {state}")
            self.last_distance_log_time = now
        return obstacle

    def set_spoof(self):
        """Khuôn mặt GIẢ MẠO: Bật đèn ĐỎ, Tắt đèn Xanh, Đóng servo."""
        self.current_state = "SPOOF"
        self.authorized_latched = False
        if self.is_active:
            if self.red_led:
                self.red_led.on()
            if self.green_led:
                self.green_led.off()
            if not self.is_door_open:
                self._set_servo_angle(0)

    def set_unknown(self):
        """Người thật nhưng CHƯA CÓ TRONG CSDL: Bật đèn XANH, Tắt đèn Đỏ, Đóng servo."""
        self.current_state = "UNKNOWN"
        self.authorized_latched = False
        if self.is_active:
            if self.green_led:
                self.green_led.on()
            if self.red_led:
                self.red_led.off()
            if not self.is_door_open:
                self._set_servo_angle(0)

    def set_authorized_open(self) -> bool:
        """
        Nhận diện ĐÚNG (Người thật + Có trong CSDL):
        - Mở Servo 90 độ.
        - Tắt 2 đèn LED.
        - Giữ cửa mở trong door_hold_seconds giây.
        """
        # Nếu cửa đang mở sẵn trong chu kỳ hiện tại thì duy trì
        if self.is_door_open and self.authorized_latched:
            return True

        # Thực hiện quay servo 90 độ
        if not self._set_servo_angle(90):
            return False

        self.authorized_latched = True
        self.is_door_open = True
        self.door_opened_time = time.monotonic()
        self.current_state = "AUTHORIZED"

        if self.is_active:
            if self.red_led:
                self.red_led.off()
            if self.green_led:
                self.green_led.off()

        print(f"[AccessController] CỬA ĐÃ MỞ (Servo 90°)! Sẽ giữ mở trong {self.door_hold_seconds}s...")
        return True

    def reset_leds(self):
        """Tắt cả 2 đèn LED."""
        if self.is_active:
            if self.red_led:
                self.red_led.off()
            if self.green_led:
                self.green_led.off()

    def reset_all(self):
        """Đóng cửa về 0 độ và tắt đèn."""
        self.current_state = "IDLE"
        self.authorized_latched = False
        self.is_door_open = False
        self.reset_leds()
        self._set_servo_angle(0)

    def handle_recognition(self, status: Optional[str]) -> bool:
        """Xử lý trạng thái nhận diện dựa trên cảm biến siêu âm."""
        self.update()
        if not self.has_obstacle() and not self.is_door_open:
            self.set_idle()
            return False

        if status == "SPOOF":
            self.set_spoof()
        elif status == "UNKNOWN":
            self.set_unknown()
        elif status == "AUTHORIZED":
            return self.set_authorized_open()
        else:
            if not self.is_door_open:
                self.set_idle()
        return True

    def set_idle(self):
        """Trạng thái chờ: Tắt đèn nếu cửa không mở."""
        if not self.is_door_open:
            if self.current_state != "IDLE":
                self.reset_leds()
            self.current_state = "IDLE"
            self.authorized_latched = False

    def update(self):
        """
        Tự động đóng cửa (quay servo về 0 độ) sau khi hết thời gian giữ mở.
        ĐÃ SỬA: Reset authorized_latched = False để sẵn sàng mở cho lần tiếp theo.
        """
        if self.is_door_open:
            elapsed = time.monotonic() - self.door_opened_time
            if elapsed >= self.door_hold_seconds:
                print("[AccessController] Hết thời gian giữ mở. Đóng cửa về 0°...")
                self._set_servo_angle(0)
                self.is_door_open = False
                self.authorized_latched = False  # Bắt buộc reset để cho phép mở lại lần sau
                if self.current_state == "AUTHORIZED":
                    self.current_state = "IDLE"
                    self.reset_leds()

    def _set_servo_angle(self, angle: int) -> bool:
        """Điều khiển góc quay của servo (0 đến 90 độ)."""
        if not self.is_active:
            print(f"[Servo Mô Phỏng] Quay góc -> {angle}°")
            self.servo_angle = angle
            return True

        # Cách 1: Ưu tiên dùng pigpio
        if self.servo_gpio is not None:
            try:
                self.servo_gpio.set_mode(self.servo_pin, pigpio.OUTPUT)
                # Dải xung chuẩn SG90: 500 µs (0°) đến 2500 µs (180°) -> 90° = 1500 µs
                pulse_width = round(500 + (angle / 180.0) * 2000)
                result = self.servo_gpio.set_servo_pulsewidth(self.servo_pin, pulse_width)
                if result < 0:
                    raise RuntimeError(f"pigpio trả về mã lỗi {result}")
                self.servo_angle = angle
                self.servo_error_reported = False
                print(f"[Servo pigpio] GPIO{self.servo_pin} -> {angle}° ({pulse_width} µs)")
                return True
            except Exception as error:
                print(f"[AccessController] Lỗi pigpio: {error}. Thử fallback...")

        # Cách 2: Tự động dự phòng bằng gpiozero.AngularServo nếu pigpio không chạy
        if self.servo_fallback is not None:
            try:
                self.servo_fallback.angle = angle
                self.servo_angle = angle
                print(f"[Servo gpiozero] GPIO{self.servo_pin} -> {angle}°")
                return True
            except Exception as error:
                print(f"[AccessController] Lỗi servo gpiozero: {error}")
                return False

        if not self.servo_error_reported:
            print(f"[AccessController] Không có driver servo nào sẵn sàng trên GPIO{self.servo_pin}. Hãy chạy 'sudo pigpiod'.")
            self.servo_error_reported = True
        return False

    def close(self):
        """Giải phóng tài nguyên GPIO khi thoát."""
        self.reset_all()
        if self.is_active:
            for dev in [self.red_led, self.green_led, self.ultrasonic, self.servo_fallback]:
                if dev is not None:
                    try:
                        dev.close()
                    except Exception:
                        pass
            if self.servo_gpio is not None:
                try:
                    self.servo_gpio.set_servo_pulsewidth(self.servo_pin, 0)
                    self.servo_gpio.stop()
                except Exception:
                    pass
                self.servo_gpio = None
            self.is_active = False


if __name__ == "__main__":
    print("=" * 65)
    print(" BÀI TEST TỔNG HỢP: SIÊU ÂM + 2 LED + SERVO")
    print("=" * 65)
    ctrl = AccessController()

    try:
        print("\n[Bước 1] Kiểm tra cảm biến siêu âm trong 3 giây...")
        for _ in range(6):
            d = ctrl.get_distance_cm()
            obstacle = "CÓ VẬT CẢN" if d <= ctrl.obstacle_distance_cm else "KHÔNG CÓ VẬT CẢN"
            print(f"  -> Khoảng cách đo được: {d:5.1f} cm | {obstacle}")
            time.sleep(0.5)

        print("\n[Bước 2] Kiểm tra trạng thái GIẢ MẠO (SPOOF)...")
        ctrl.set_spoof()
        print("  -> LED ĐỎ bật, LED xanh tắt, Servo 0°.")
        time.sleep(1.5)

        print("\n[Bước 3] Kiểm tra trạng thái NGƯỜI LẠ (UNKNOWN)...")
        ctrl.set_unknown()
        print("  -> LED XANH bật, LED đỏ tắt, Servo 0°.")
        time.sleep(1.5)

        print("\n[Bước 4] Kiểm tra MỞ CỬA LẦN 1 (Quay Servo 90 độ)...")
        ctrl.set_authorized_open()
        while ctrl.is_door_open:
            ctrl.update()
            time.sleep(0.1)

        print("\n[Bước 5] Kiểm tra MỞ CỬA LẦN 2 (Kiểm tra xem bug kẹt latched đã hết chưa)...")
        ctrl.set_authorized_open()
        while ctrl.is_door_open:
            ctrl.update()
            time.sleep(0.1)

        print("\n[OK] Toàn bộ test hoàn tất xuất sắc!")
    finally:
        ctrl.close()
