import Xlib.display
import pyautogui

from loguru import logger


class MouseController:
    """
    This is a sample class that you can use to control the mouse pointer.

    It uses the pyautogui library. You can set the precision for mouse movement
    (how much the mouse moves) and the speed (how fast it moves) by changing
    precision_dict and speed_dict.
    """

    def __init__(self, precision, speed):
        precision_dict = {"high": 100, "low": 1000, "medium": 500}
        speed_dict = {"fast": 1, "slow": 10, "medium": 5}

        self.precision = precision_dict[precision]
        self.speed = speed_dict[speed]

    def move(self, x, y):
        """Move mouse pointer to position the x and y."""
        try:
            start_pos = pyautogui.position()
            pyautogui.moveRel(
                -x * self.precision, y * self.precision, duration=self.speed
            )
            end_pos = pyautogui.position()
            logger.info(f"Mouse -> start_pos: {start_pos}, end_pos: {end_pos}")
        except pyautogui.FailSafeException:
            logger.exception(f"Position: {x}, {y} are out of the screen")
            pyautogui.moveRel(
                x * self.precision, -1 * y * self.precision, duration=self.speed
            )
    def left_click(self):
        pass

    def right_click(self):
        pass
