from serial_reader import SerialReader
from imu_viz import IMUVisualizer
from state import IMUState


def main():
    IMU_PORT = "/dev/cu.usbmodem101"   # change these as needed
    BAUD = 115200

    imu_state = IMUState()

    # callbacks for the serial readers
    def imu_line_callback(line: str) -> None:
        imu_state.update_from_line(line)


    imu_reader = SerialReader(
        port=IMU_PORT,
        baudrate=BAUD,
        line_callback=imu_line_callback,
        name="IMU",
    )
    imu_reader.start()

    visualizer = IMUVisualizer(
        get_quaternion=imu_state.get_quaternion,
        dt_ms=20,
    )

    try:
        visualizer.start()
    finally:
        imu_reader.stop()


if __name__ == "__main__":
    main()