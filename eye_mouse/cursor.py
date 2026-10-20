"""X11 pointer and global Escape; no clicks or key injection."""

import os


class NativeCursor:
    def __init__(self):
        self.display = None
        self.armed = False

    def arm(self):
        if os.environ.get("XDG_SESSION_TYPE") == "wayland" or not os.environ.get("DISPLAY"):
            raise RuntimeError("Native cursor requires an X11 desktop. Preview works on Wayland.")
        from Xlib import XK, X, display

        self.display = display.Display()
        self.root = self.display.screen().root
        self.escape_key = self.display.keysym_to_keycode(XK.string_to_keysym("Escape"))
        errors = []
        # AnyModifier conflicts with desktop shortcuts such as Ctrl+Alt+Escape.
        # Reserve plain Escape, including the common CapsLock/NumLock states.
        self.modifiers = (0, X.LockMask, X.Mod2Mask, X.LockMask | X.Mod2Mask)
        for mask in self.modifiers:
            self.root.grab_key(
                self.escape_key,
                mask,
                False,
                X.GrabModeAsync,
                X.GrabModeAsync,
                onerror=lambda error, request: errors.append(error) or 1,
            )
        self.display.sync()
        if errors:
            self.display.close()
            self.display = None
            raise RuntimeError("Cannot register global Escape; mouse remains disarmed")
        self.armed = True

    def emergency_requested(self):
        if not self.display:
            return False
        from Xlib import X

        while self.display.pending_events():
            event = self.display.next_event()
            if event.type == X.KeyPress and event.detail == self.escape_key:
                return True
        # A physical pointer move to the desktop's upper-left also disarms.
        point = self.root.query_pointer()
        return point.root_x <= 1 and point.root_y <= 1

    def move(self, x, y):
        if self.armed and self.display:
            self.root.warp_pointer(int(round(x)), int(round(y)))
            self.display.flush()

    def disarm(self):
        self.armed = False
        if self.display:
            from Xlib import X

            try:
                self.root.ungrab_key(self.escape_key, X.AnyModifier)
                self.display.sync()
            finally:
                self.display.close()
                self.display = None
