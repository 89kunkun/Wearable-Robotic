import threading
import serial


class SerialReader(threading.Thread):
    """
    Generic non-blocking serial reader.

    - Runs in its own thread.
    - Blocks internally on serial.read().
    - Calls a user-provided line_callback(line: str) for each complete line.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        line_callback=None,
        name: str = "SerialReader",
        read_chunk: int = 128,
        timeout: float = 1.0,
    ):
        super().__init__(daemon=True, name=name)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.read_chunk = read_chunk
        self.line_callback = line_callback
        self._ser = None
        self._running = threading.Event()
        self._running.set()
        self._buffer = b""

    def open(self):
        self._ser = serial.Serial(
            self.port,
            self.baudrate,
            timeout=self.timeout,
        )

    def run(self):
        try:
            self.open()
            print(f"[{self.name}] Opened {self.port} @ {self.baudrate}")
        except Exception as e:
            print(f"[{self.name}] Failed to open {self.port}: {e}")
            return

        while self._running.is_set():
            try:
                data = self._ser.read(self.read_chunk)
                if not data:
                    continue

                self._buffer += data

                # Process all complete lines in the buffer
                while b"\n" in self._buffer:
                    raw_line, self._buffer = self._buffer.split(b"\n", 1)
                    line = raw_line.decode(errors="ignore").strip()
                    if self.line_callback is not None and line:
                        self.line_callback(line)

            except Exception as e:
                print(f"[{self.name}] Error: {e}")
                break

        # Clean up
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
            print(f"[{self.name}] Closed {self.port}")

    def stop(self):
        self._running.clear()