// MLX90393 Magnetometer Joystick + DRV2605 Haptic (Buzz only) + MPR121 Finger Bend
// Python 시각화 코드와 호환 유지 (Serial 토글 가능)
//
// I2C 장치:
//   0x18 - MLX90393 (Magnetometer)
//   0x5A - DRV2605 (Haptic Driver)
//   0x5B - MPR121  (Capacitive Touch) ← ADDR 핀을 3.3V에 연결
//
// 버튼 핀 배정:
//   D13 - Magnetometer 오프셋 캘리브레이션
//   D12 - MPR121 Finger 캘리브레이션
//   D11 - Serial 출력 토글 (ON/OFF)

#include <Wire.h>
#include "Adafruit_MLX90393.h"
#include "Adafruit_DRV2605.h"
#include <math.h>

// ===== 센서 객체 =====
Adafruit_MLX90393 sensor = Adafruit_MLX90393();
Adafruit_DRV2605 drv;

// ===== 핀 정의 =====
#define BUTTON_MAG_CAL    13  // Magnetometer 캘리브레이션
#define BUTTON_FINGER_CAL 12  // Finger 캘리브레이션
#define BUTTON_SERIAL     11  // Serial 출력 토글

// =====================================================
// [A] MLX90393 + DRV2605 설정
// =====================================================

float x_offset = 0;
float y_offset = 0;
float z_offset = 0;

float k_sensitivity = 1.0;
float C_mm_per_rad = 1.0;

const float MAGNET_THRESHOLD = 90.0;
const float CENTER_DEADZONE  = 28.0;
const float MAX_DISPLACEMENT = 80.0;

// Haptic: 연속 Buzz만 사용 (5구간)
const uint8_t BUZZ_EFFECTS[] = {51, 50, 49, 48, 47};  // Buzz 20%~100%
const int     HAPTIC_ZONES = 5;
const uint8_t CENTER_SNAP_EFFECT = 1;  // Strong Click 100%

// 상태머신: 0=OFF, 1=WAITING, 2=ON
uint8_t uiState = 0;
unsigned long detectionStartTime = 0;
const unsigned long ACTIVATION_TIME_MS = 500;

int  prevZone = -1;
bool wasInCenter = false;
bool wasOutside = false;

unsigned long lastBuzzTime = 0;
const unsigned long BUZZ_INTERVAL_MS = 120;

bool serialEnabled = true;

// 버튼 디바운스
unsigned long lastFingerCalPress = 0;
unsigned long lastSerialPress = 0;
const unsigned long DEBOUNCE_MS = 250;

// =====================================================
// [B] MPR121 Finger Bend 설정
// =====================================================

#define MPR121_ADDR          0x5B  // ADDR 핀 → 3.3V
#define MPR121_SOFT_RESET    0x80
#define MPR121_AFE_CONFIG    0x5C
#define MPR121_FILTER_CONFIG 0x5D
#define MPR121_ECR           0x5E
#define MPR121_AUTOCONFIG0   0x7B
#define MPR121_DATA_BASE_ADDR 0x04

#define NUM_FINGERS 3
#define BENT_OFFSET 1.0

float fingerThresholds[NUM_FINGERS] = {0, 0, 0};
bool  fingerCalibrated = false;
bool  fingerBent[NUM_FINGERS] = {false, false, false};  // Middle, Ring, Pinky

// =====================================================
// [C] 손가락 굽힘 조합 (비트플래그)
// =====================================================
#define COMBO_NONE              0x00  // 000
#define COMBO_MIDDLE            0x01  // 001
#define COMBO_RING              0x02  // 010
#define COMBO_MIDDLE_RING       0x03  // 011
#define COMBO_RING_PINKY        0x06  // 110
#define COMBO_MIDDLE_RING_PINKY 0x07  // 111

uint8_t fingerCombo = COMBO_NONE;

// TODO: 손가락 조합별 트리거 매핑
//   switch(fingerCombo) {
//     case COMBO_MIDDLE:            break;
//     case COMBO_RING:              break;
//     case COMBO_MIDDLE_RING:       break;
//     case COMBO_RING_PINKY:        break;
//     case COMBO_MIDDLE_RING_PINKY: break;
//   }

// ===== 함수 선언 =====
void playEffect(uint8_t effect);
int  getZone(float displacement);
void handleHapticFeedback(float displacement);
void mpr121_writeRegister(uint8_t reg, uint8_t value);
uint16_t mpr121_read16(uint8_t reg);
void updateFingerBend();
void calibrateFingers();

// =====================================================
void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  Serial.println("MLX90393 + DRV2605 + MPR121 Glove Controller");

  // --- MLX90393 (0x18) ---
  if (!sensor.begin_I2C(0x18)) {
    Serial.println("MLX90393 not found!");
    while (1) { delay(10); }
  }
  Serial.println("MLX90393 OK (0x18)");

  sensor.setGain(MLX90393_GAIN_1X);
  sensor.setResolution(MLX90393_X, MLX90393_RES_17);
  sensor.setResolution(MLX90393_Y, MLX90393_RES_17);
  sensor.setResolution(MLX90393_Z, MLX90393_RES_16);
  sensor.setOversampling(MLX90393_OSR_3);
  sensor.setFilter(MLX90393_FILTER_5);

  // --- DRV2605 (0x5A) ---
  if (!drv.begin()) {
    Serial.println("DRV2605 not found!");
    while (1) { delay(10); }
  }
  Serial.println("DRV2605 OK (0x5A)");
  drv.selectLibrary(1);
  drv.setMode(DRV2605_MODE_INTTRIG);

  // --- MPR121 (0x5B) ---
  mpr121_writeRegister(MPR121_SOFT_RESET, 0x63);
  delay(10);
  mpr121_writeRegister(MPR121_ECR, 0x00);
  mpr121_writeRegister(MPR121_AUTOCONFIG0, 0x00);
  mpr121_writeRegister(MPR121_AFE_CONFIG, 0x95);
  mpr121_writeRegister(MPR121_FILTER_CONFIG, 0xD0);
  mpr121_writeRegister(MPR121_ECR, 0x80 | NUM_FINGERS);
  Serial.println("MPR121 OK (0x5B)");

  // --- 핀 설정 ---
  pinMode(BUTTON_MAG_CAL,    INPUT_PULLUP);
  pinMode(BUTTON_FINGER_CAL, INPUT_PULLUP);
  pinMode(BUTTON_SERIAL,     INPUT_PULLUP);

  Serial.println("Ready. D13=MagCal, D12=FingerCal, D11=Serial");
}

// =====================================================
void loop() {
  unsigned long now = millis();

  // ===== [1] 버튼 처리 =====

  // D13: Magnetometer 캘리브레이션
  if (digitalRead(BUTTON_MAG_CAL) == LOW) {
    sensors_event_t calEvent;
    sensor.getEvent(&calEvent);
    x_offset = calEvent.magnetic.x;
    y_offset = calEvent.magnetic.y;
    z_offset = calEvent.magnetic.z;
    prevZone = -1;
    wasInCenter = false;
    wasOutside = false;
    if (serialEnabled) Serial.println(">> Mag Calibrated!");
    delay(200);
  }

  // D12: Finger 캘리브레이션
  if (digitalRead(BUTTON_FINGER_CAL) == LOW && (now - lastFingerCalPress > DEBOUNCE_MS)) {
    lastFingerCalPress = now;
    calibrateFingers();
  }

  // D11: Serial 출력 토글
  if (digitalRead(BUTTON_SERIAL) == LOW && (now - lastSerialPress > DEBOUNCE_MS)) {
    lastSerialPress = now;
    serialEnabled = !serialEnabled;
    Serial.print(">> Serial Output: ");
    Serial.println(serialEnabled ? "ON" : "OFF");
  }

  // ===== [2] Magnetometer 읽기 =====
  sensors_event_t event;
  sensor.getEvent(&event);

  float value_x = event.magnetic.x - x_offset;
  float value_y = event.magnetic.y - y_offset;
  float value_z = event.magnetic.z - z_offset;

  // ===== [3] 상태머신 (OFF→WAITING→ON) =====
  bool isMagnetDetected = (abs(value_z) >= MAGNET_THRESHOLD);

  if (isMagnetDetected) {
    if (uiState == 0) {
      uiState = 1;
      detectionStartTime = now;
    } else if (uiState == 1) {
      if (now - detectionStartTime >= ACTIVATION_TIME_MS) {
        uiState = 2;
      }
    }
  } else {
    uiState = 0;
    detectionStartTime = 0;
    prevZone = -1;
    wasInCenter = false;
    wasOutside = false;
  }

  // ===== [4] Heading & Displacement =====
  float heading = atan2(value_y, value_x) * 180.0 / PI;

  float B_planar = sqrt(sq(value_x) + sq(value_y));
  float B_z_abs = abs(event.magnetic.z);
  if (B_z_abs < 0.1) B_z_abs = 0.1;

  float displacement_angle = atan(B_planar / (k_sensitivity * B_z_abs));
  float displacement = C_mm_per_rad * displacement_angle * (180.0 / PI);

  // ===== [5] Haptic 피드백 (ON 상태에서만) =====
  if (uiState == 2) {
    handleHapticFeedback(displacement);
  }

  // ===== [6] Finger Bend 읽기 =====
  updateFingerBend();

  // ===== [7] Serial 출력 (Python 호환 형식) =====
  if (serialEnabled) {
    Serial.print("X_corr:"); Serial.print(value_x); Serial.print(",");
    Serial.print("Y_corr:"); Serial.print(value_y); Serial.print(",");
    Serial.print("Z_corr:"); Serial.print(value_z); Serial.print(",");
    Serial.print("Heading:"); Serial.print(heading); Serial.print(",");
    Serial.print("Displacement:"); Serial.print(displacement); Serial.print(",");
    Serial.print("State:"); Serial.print(uiState); Serial.print(",");
    Serial.print("Zone:"); Serial.print(getZone(displacement)); Serial.print(",");
    Serial.print("HapticMode:"); Serial.print(0); Serial.print(",");
    Serial.print("Bending:");
    Serial.print(fingerBent[0] ? "O" : "X");
    Serial.print(fingerBent[1] ? "O" : "X");
    Serial.print(fingerBent[2] ? "O" : "X");
    Serial.print(",");
    Serial.print("Fingers:"); Serial.println(fingerCombo);
  }
}

// =====================================================
// Haptic 피드백 — 연속 Buzz만 사용
// =====================================================
void handleHapticFeedback(float displacement) {
  unsigned long now = millis();
  bool inCenter = (displacement <= CENTER_DEADZONE);
  int  zone = getZone(displacement);

  // 중앙 복귀 snap-back
  if (inCenter && wasOutside) {
    playEffect(CENTER_SNAP_EFFECT);
    wasInCenter = true;
    wasOutside = false;
    prevZone = -1;
    lastBuzzTime = now;
    return;
  }

  if (inCenter) {
    wasInCenter = true;
    wasOutside = false;
    prevZone = -1;
    return;
  }

  // 중앙 밖: 연속 Buzz
  wasOutside = true;
  wasInCenter = false;

  if (now - lastBuzzTime >= BUZZ_INTERVAL_MS) {
    lastBuzzTime = now;
    playEffect(BUZZ_EFFECTS[zone]);
  }
}

int getZone(float displacement) {
  if (displacement <= CENTER_DEADZONE) return 0;
  float range = MAX_DISPLACEMENT - CENTER_DEADZONE;
  float normalized = (displacement - CENTER_DEADZONE) / range;
  int zone = (int)(normalized * HAPTIC_ZONES);
  if (zone >= HAPTIC_ZONES) zone = HAPTIC_ZONES - 1;
  if (zone < 0) zone = 0;
  return zone;
}

void playEffect(uint8_t effect) {
  drv.setWaveform(0, effect);
  drv.setWaveform(1, 0);
  drv.go();
}

// =====================================================
// MPR121 Finger Bend
// =====================================================
void updateFingerBend() {
  fingerCombo = COMBO_NONE;

  for (int i = 0; i < NUM_FINGERS; i++) {
    uint8_t regAddr = MPR121_DATA_BASE_ADDR + (i * 2);
    uint16_t rawData = mpr121_read16(regAddr);
    float capacitance = (rawData > 0) ? 104260.0 / (float)rawData : 0;

    fingerBent[i] = fingerCalibrated && (capacitance >= fingerThresholds[i] + BENT_OFFSET);

    if (fingerBent[i]) {
      fingerCombo |= (1 << i);
    }
  }
}

void calibrateFingers() {
  for (int i = 0; i < NUM_FINGERS; i++) {
    uint8_t regAddr = MPR121_DATA_BASE_ADDR + (i * 2);
    uint16_t rawData = mpr121_read16(regAddr);
    fingerThresholds[i] = (rawData > 0) ? 104260.0 / (float)rawData : 0;
  }
  fingerCalibrated = true;

  if (serialEnabled) {
    Serial.print(">> Finger Cal: ");
    const char* names[] = {"Mid", "Ring", "Pinky"};
    for (int i = 0; i < NUM_FINGERS; i++) {
      Serial.print(names[i]); Serial.print("=");
      Serial.print(fingerThresholds[i], 1);
      if (i < NUM_FINGERS - 1) Serial.print(", ");
    }
    Serial.println();
  }
}

// MPR121 I2C
void mpr121_writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPR121_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

uint16_t mpr121_read16(uint8_t reg) {
  Wire.beginTransmission(MPR121_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPR121_ADDR, 2);
  if (Wire.available() == 2) {
    uint8_t lsb = Wire.read();
    uint8_t msb = Wire.read();
    return ((msb << 8) | lsb) & 0x03FF;
  }
  return 0;
}