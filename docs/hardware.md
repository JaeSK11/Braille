# Hardware

## Sensor: DFRobot SEN0628, "Gravity: 8x8 Matrix ToF 3D Distance Sensor"

| Item | Value |
|---|---|
| Ranging die | ST VL53L7CX: flash LiDAR, 940 nm VCSEL emitter, SPAD receiver array, direct time of flight with per-zone histograms |
| Onboard MCU | RP2040 running DFRobot firmware |
| Zones | 8x8 = 64 |
| Field of view | 60° x 60° (90° diagonal), about 7.5° per zone |
| Range | 20 mm to 3.5 m |
| Accuracy (datasheet) | ±11–12 mm under 20 cm; ±5% (white) / ±6% (grey) beyond 20 cm |
| Frame rate | 15–60 Hz spec; about 16 fps in USB text mode |
| Host interface | USB-C, enumerates as a CDC serial port (`/dev/ttyACM*`), 115200 baud, text output |
| Other interfaces | Gravity I2C and UART (not used here) |
| Firmware | V1.3, flashed by UF2 drag-and-drop. V1.3 reports invalid zones as exactly `4000` |

Why this sensor: it is a genuine single-photon time-of-flight LiDAR device,
gives a native 2-D grid, and plugs straight into a computer over USB with no
extra microcontroller or soldering.

Alternatives considered: a 2-D spinning LiDAR (one scan plane only, would need
a tilt mechanism), a bare VL53L8CX breakout (needs an STM32 host board), and a
structured-light depth camera (not LiDAR, with a large minimum range).

## Serial stream format

One line per row, values in millimetres, trailing comma:

```
y0:v,v,v,v,v,v,v,v,
...
y7:v,v,v,v,v,v,v,v,
```

Rows `y0..y7` run top to bottom and values within a row run left to right,
in **camera view**. Column 0 is the sensor's left, which is the right of a
person facing the lens. `4000` means no valid return. `src/orientation.json`
holds flip/transpose flags in case a different mount needs them.

## Measured behaviour

These properties were characterised on the bench and are what
`src/sensor_sim.py` reproduces:

- **Noise on a solid surface:** about 5 mm standard deviation per zone at
  25–50 cm, rising to about 8 mm under 20 cm.
- **Perpendicular distance:** a flat wall facing the sensor reads flat across
  all 64 zones rather than curved. Any remaining gradient is sensor tilt,
  typically a few centimetres across the field.
- **Mixed zones:** a zone that straddles two surfaces (e.g. a box edge in
  front of a wall) returns an intermediate depth, flickers between the two
  surfaces from frame to frame (std of 50 mm or more), and sometimes drops
  out as `4000`.
- **Zone footprint:** zone width ≈ 0.144 x distance. It is set by the
  distance to the surface in that zone, not to the background.

## Setup notes (Linux)

1. Use a USB-C **data** cable. A charge-only cable gives no serial port.
2. Add your user to the `dialout` group, then log out and back in.
3. Some front-panel USB ports failed to enumerate the device (`-71` errors),
   and a rear port worked. If `dmesg` shows setup-address errors, change ports.
4. ModemManager may briefly hold a new `ttyACM` port after plug-in.
5. Check the stream: `python3 -m serial.tools.miniterm /dev/ttyACM0 115200`.
6. Check orientation by moving a hand across the field and confirming which
   rows and columns change.

## Test targets

- **Reference target:** a flat matte wall, then a single box standing off the
  wall (a few centimetres to about 8 cm of stand-off).
- **Shape library targets:** cubes of about 3 cm on a 3.2 cm grid at a 30 cm
  wall distance with 8 cm stand-off. At that distance one zone is about 3.2 cm,
  so each cube fills one zone and any 5x5 library mask can be built.
- Use matte, light surfaces. The accuracy spec does not hold on glossy or
  black targets.

## References

- Product page: https://www.dfrobot.com/product-2999.html
- Wiki (firmware, USB output, API): https://wiki.dfrobot.com/sen0628/
- STMicroelectronics VL53L7CX datasheet: https://www.st.com/en/imaging-and-photonics-solutions/vl53l7cx.html
