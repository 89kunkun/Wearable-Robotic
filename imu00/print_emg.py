from serial_reader import SerialReader


PORT = "COM7"       # <-- change to your Arduino port
BAUDRATE = 115200


def main():
    def emg_line_callback(line: str):
        # Just print the line exactly as Arduino sends it
        print("EMG:", line)

    reader = SerialReader(
        port=PORT,
        baudrate=BAUDRATE,
        line_callback=emg_line_callback,
        name="EMG",
    )

    reader.start()

    print("Reading EMG data... Press Ctrl+C to stop.")
    try:
        while True:
            pass   # keep program alive
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        reader.stop()


if __name__ == "__main__":
    main()