"""
servo_controller.py
===================
Controls the Tower Pro servo motor connected to the Jetson Orin Nano GPIO.

The servo operates a slide panel / hatch that sorts chicks into:
  - Healthy hatch  (servo angle: HEALTHY_ANGLE)
  - Defective hatch (servo angle: DEFECT_ANGLE)

Wiring (Jetson Orin Nano Developer Kit)
----------------------------------------
  Servo signal → GPIO pin 32 (PWM-capable)
  Servo VCC    → External 5V supply (NOT Jetson 3.3V)
  Servo GND    → Common GND with Jetson

IMPORTANT: Do NOT power the servo from Jetson's 3.3V rail.
           Use an external 5V supply to avoid brownouts.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Servo positions (degrees)
NEUTRAL_ANGLE = 90
HEALTHY_ANGLE = 45   # Opens healthy-chick hatch
DEFECT_ANGLE  = 135  # Opens defective-chick hatch

# Timing
HATCH_OPEN_DURATION = 2.0   # seconds to hold hatch open
RETURN_DELAY        = 0.5   # seconds before returning to neutral

# GPIO pin (BCM numbering on Jetson)
SERVO_GPIO_PIN = 32
PWM_FREQUENCY  = 50  # Hz (standard for Tower Pro servos)


def _angle_to_duty_cycle(angle: float) -> float:
    """Convert servo angle (0–180°) to PWM duty cycle (2.5–12.5%)."""
    return 2.5 + (angle / 180.0) * 10.0


class ServoController:
    """
    Interface for the Tower Pro servo motor.

    On platforms without Jetson.GPIO the controller enters a simulation
    mode where actions are logged but no physical output occurs.
    This allows testing on a development machine.
    """

    def __init__(self, pin: int = SERVO_GPIO_PIN, simulate: bool = False):
        self.pin = pin
        self.simulate = simulate
        self._pwm = None
        self._gpio = None
        self._current_angle: float = NEUTRAL_ANGLE

        if not simulate:
            self._init_gpio()

    def _init_gpio(self):
        try:
            import Jetson.GPIO as GPIO
            self._gpio = GPIO
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
            self._pwm = GPIO.PWM(self.pin, PWM_FREQUENCY)
            self._pwm.start(_angle_to_duty_cycle(NEUTRAL_ANGLE))
            logger.info(f"Servo initialized on GPIO pin {self.pin}")
        except ImportError:
            logger.warning(
                "Jetson.GPIO not found. Servo running in SIMULATION mode. "
                "Install with: pip install Jetson.GPIO"
            )
            self.simulate = True
        except Exception as e:
            logger.error(f"GPIO init failed: {e}. Entering simulation mode.")
            self.simulate = True

    def _set_angle(self, angle: float):
        self._current_angle = angle
        if self.simulate:
            logger.info(f"[SIM] Servo → {angle:.1f}°")
            return
        if self._pwm:
            self._pwm.ChangeDutyCycle(_angle_to_duty_cycle(angle))
            time.sleep(0.3)   # Allow servo to reach position

    def open_healthy_hatch(self):
        """Rotate servo to route chick to healthy lane."""
        logger.info("Opening HEALTHY hatch")
        self._set_angle(HEALTHY_ANGLE)
        time.sleep(HATCH_OPEN_DURATION)
        self._set_angle(NEUTRAL_ANGLE)
        time.sleep(RETURN_DELAY)

    def open_defect_hatch(self):
        """Rotate servo to route chick to defective lane."""
        logger.info("Opening DEFECTIVE hatch")
        self._set_angle(DEFECT_ANGLE)
        time.sleep(HATCH_OPEN_DURATION)
        self._set_angle(NEUTRAL_ANGLE)
        time.sleep(RETURN_DELAY)

    def cleanup(self):
        if self.simulate:
            return
        if self._pwm:
            self._pwm.stop()
        if self._gpio:
            self._gpio.cleanup()
        logger.info("GPIO cleaned up.")
