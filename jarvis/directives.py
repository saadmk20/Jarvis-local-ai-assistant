"""In-band directives the model may emit, stripped before the reply is shown or spoken."""
import re


OPEN_MARK = "OPEN:"


class OpenDirective:
    """Strips `OPEN: <url>` out of the answer before it reaches the screen or the voice.

    The model is told to put that line on its own line, but text arrives in small pieces, so this
    holds back only the few characters that could still grow into the marker.
    """

    def __init__(self):
        self.hold = ""        # characters that may still become the marker
        self.at_start = True  # we are at the beginning of a line
        self.in_dir = False   # inside a directive line
        self.confirmed = False
        self.dir = ""
        self.urls = []

    def feed(self, piece):
        out = []
        for ch in piece:
            if self.in_dir:
                if not self.confirmed:            # "OPEN:" alone proves nothing: wait for the address
                    if ch in " \t":
                        continue
                    if ch in "hH\"'<":            # http://  "http…  <http…
                        self.confirmed = True
                        self.dir += ch
                        continue
                    out.append("OPEN: " + ch)     # it was prose after all: give the line back
                    self.dir, self.in_dir, self.at_start = "", False, ch == "\n"
                    continue
                if ch == "\n":
                    self._close()
                    out.append("\n")          # keep the sentence after the directive from gluing to the one before
                else:
                    self.dir += ch
                continue
            if self.at_start:
                self.hold += ch
                probe = self.hold.lstrip(" \t->*•").upper()
                if OPEN_MARK.startswith(probe):
                    if probe == OPEN_MARK:      # the marker is complete: swallow the rest of the line
                        self.in_dir, self.dir, self.hold, self.at_start = True, "", "", False
                        self.confirmed = False
                    continue
                out.append(self.hold)           # ordinary sentence after all: release what we held
                self.hold = ""
                self.at_start = ch == "\n"
                continue
            out.append(ch)
            self.at_start = ch == "\n"
        return "".join(out), self._take()

    def _close(self):
        m = re.search(r"https?://[^\s<>\"'`]+", self.dir, re.I)
        if m:
            self.urls.append(m.group(0).rstrip(").,;:!?"))
        self.dir = ""
        self.in_dir = False
        self.confirmed = False
        self.at_start = True

    def _take(self):
        got = self.urls
        self.urls = []
        return got

    def finish(self):
        """End of stream: close a dangling directive and release anything still held."""
        if self.in_dir and self.confirmed:
            self._close()
        elif self.in_dir:                     # a bare "OPEN:" with nothing after it: drop it
            self.dir, self.in_dir, self.confirmed = "", False, False
        rest = self.hold
        self.hold = ""
        return rest, self._take()
