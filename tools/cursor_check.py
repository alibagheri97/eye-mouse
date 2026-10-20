"""Explicit X11 cursor/stop check: move a few pixels, inject Escape to our grab, restore pointer."""

import json

from Xlib import X, display
from Xlib.ext import xtest

from eye_mouse.cursor import NativeCursor
from eye_mouse.storage import atomic_json


def main():
    connection = display.Display()
    root = connection.screen().root
    original = root.query_pointer()
    cursor = NativeCursor()
    report = {}
    try:
        cursor.arm()
        report["armed"] = cursor.armed
        x = min(max(10, original.root_x + 3), root.get_geometry().width - 4)
        y = min(max(10, original.root_y + 3), root.get_geometry().height - 4)
        cursor.move(x, y)
        cursor.display.sync()
        point = root.query_pointer()
        report["native_movement_verified"] = point.root_x == x and point.root_y == y
        xtest.fake_input(connection, X.KeyPress, cursor.escape_key)
        xtest.fake_input(connection, X.KeyRelease, cursor.escape_key)
        connection.sync()
        # Round-trip on the grab's connection ensures the server has delivered queued input.
        cursor.display.sync()
        report["global_escape_received"] = cursor.emergency_requested()
    finally:
        cursor.disarm()
        root.warp_pointer(original.root_x, original.root_y)
        connection.sync()
        connection.close()
    report["disarmed"] = not cursor.armed
    atomic_json("artifacts/cursor-runtime.json", report)
    print(json.dumps(report, indent=2))
    if not all(report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
