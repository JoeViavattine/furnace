// Raw accelerometer streamer: LSM6DSOX FIFO -> binary packets over USB.
// All analysis happens on the host (logger.py / analyze.py).
//
// Packet format (little-endian), one packet per FIFO burst:
//   0-1   sync        0xA5 0x5A
//   2     n           samples in this packet (1..16)
//   3     flags       bit0 = FIFO overrun seen, bit1 = I2C error seen (since last packet)
//   4-7   t_us        uint32 micros() when this burst was read
//   8-11  first_idx   uint32 running index of the first sample in this packet
//   12..  samples     n x (int16 x, int16 y, int16 z), raw counts (+/-2 g: 0.061 mg/LSB)
//   last  checksum    uint8 sum of bytes 2..(end of samples), mod 256
// Do NOT open the Serial Monitor while this runs - the output is binary.

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_LSM6DSOX.h>

constexpr uint8_t ADDR = 0x6A;
constexpr uint8_t REG_FIFO_CTRL3 = 0x09;
constexpr uint8_t REG_FIFO_CTRL4 = 0x0A;
constexpr uint8_t REG_CTRL3_C    = 0x12;
constexpr uint8_t REG_FIFO_STAT1 = 0x3A;
constexpr uint8_t REG_FIFO_TAG   = 0x78;
constexpr uint8_t TAG_ACCEL      = 0x02;
constexpr int BURST = 16;

Adafruit_LSM6DSOX sox;
uint32_t sampleIndex = 0;
uint8_t pendingFlags = 0;

bool readReg(uint8_t reg, uint8_t *buf, size_t len) {
  Wire.beginTransmission(ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint16_t)ADDR, (size_t)len, true) != len) return false;
  for (size_t i = 0; i < len; i++) buf[i] = Wire.read();
  return true;
}

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);

#ifdef PIN_I2C_POWER
  pinMode(PIN_I2C_POWER, OUTPUT);
  digitalWrite(PIN_I2C_POWER, HIGH);
  delay(10);
#endif

  if (!sox.begin_I2C()) {
    while (true) {                       // human-readable error, repeated
      Serial.println("LSM6DSOX not found - check STEMMA QT cable");
      delay(1000);
    }
  }
  Wire.setClock(400000);
  sox.setAccelRange(LSM6DS_ACCEL_RANGE_2_G);
  sox.setAccelDataRate(LSM6DS_RATE_3_33K_HZ);
  sox.setGyroDataRate(LSM6DS_RATE_SHUTDOWN);
  writeReg(REG_CTRL3_C, 0x44);           // block data update + auto-increment
  writeReg(REG_FIFO_CTRL3, 0x09);        // batch accel at 3333 Hz nominal
  writeReg(REG_FIFO_CTRL4, 0x00);        // bypass: clear FIFO
  writeReg(REG_FIFO_CTRL4, 0x06);        // continuous mode
}

void loop() {
  uint8_t s[2];
  if (!readReg(REG_FIFO_STAT1, s, 2)) { pendingFlags |= 0x02; return; }
  if (s[1] & 0x48) pendingFlags |= 0x01;              // overrun (live or latched)
  int count = ((s[1] & 0x03) << 8) | s[0];
  if (count == 0) return;
  int nread = count < BURST ? count : BURST;

  uint8_t raw[BURST * 7];
  if (!readReg(REG_FIFO_TAG, raw, nread * 7)) { pendingFlags |= 0x02; return; }
  uint32_t t = micros();

  uint8_t pkt[12 + BURST * 6 + 1];
  int n = 0;
  for (int w = 0; w < nread; w++) {
    const uint8_t *p = &raw[w * 7];
    if ((p[0] >> 3) != TAG_ACCEL) continue;
    memcpy(&pkt[12 + n * 6], &p[1], 6);              // x,y,z little-endian int16
    n++;
  }
  if (n == 0) return;

  pkt[0] = 0xA5; pkt[1] = 0x5A;
  pkt[2] = (uint8_t)n;
  pkt[3] = pendingFlags;
  memcpy(&pkt[4], &t, 4);
  memcpy(&pkt[8], &sampleIndex, 4);
  uint8_t sum = 0;
  for (int i = 2; i < 12 + n * 6; i++) sum += pkt[i];
  pkt[12 + n * 6] = sum;

  Serial.write(pkt, 13 + n * 6);
  sampleIndex += n;
  pendingFlags = 0;
}
