// MLX90393 Magnetometer Joystick + DRV2605 Haptic (Buzz + Hovering Double Click) + MPR121 Finger Bend
// Python 시각화 코드와 호환 유지 (Serial 토글 가능)

#include <Wire.h>
#include "Adafruit_MLX90393.h"
#include "Adafruit_DRV2605.h"
#include <math.h>

// ===== 센서 객체 =====
Adafruit_MLX90393 sensor = Adafruit_MLX90393();
Adafruit_DRV2605 drv;

// ===== 핀 정의 =====
#define BUTTON_MAG_CAL    13
#define BUTTON_FINGER_CAL 12
#define BUTTON_SERIAL     11

// =====================================================
// [A] MLX90393 + DRV2605 설정
// =====================================================

float x_offset = 0;
float y_offset = 0;
float z_offset = 0;

float k_sensitivity = 1.0;
float C_mm_per_rad = 1.0;

// 새로 정의된 Norm 기반 Threshold 값들
const float THRESHOLD_OFF_MAX   = 600.0;
const float THRESHOLD_HOVER_MIN = 600.0;
const float THRESHOLD_HOVER_MAX = 1300.0;
const float THRESHOLD_ON_MIN    = 1500.0;

const float CENTER_DEADZONE  = 28.0;
const float MAX_DISPLACEMENT = 80.0;

// Haptic (ON 상태용 연속 Buzz)
const uint8_t BUZZ_EFFECTS[] = {51, 50, 49, 48, 47};
const int     HAPTIC_ZONES = 5;
const uint8_t CENTER_SNAP_EFFECT = 1;

// Haptic (HOVER 상태용 확실히 구분되는 진동: triple Click 100%)
const uint8_t HOVER_EFFECT = 12;
unsigned long lastHoverBuzzTime = 0;
const unsigned long HOVER_BUZZ_INTERVAL_MS = 800; // 0.8초마다 톡톡

// 상태머신: 0=OFF, 1=WAITING, 2=ON, 3=HOVERING
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

#define MPR121_ADDR          0x5B
#define MPR121_SOFT_RESET    0x80
#define MPR121_AFE_CONFIG    0x5C
#define MPR121_FILTER_CONFIG 0x5D
#define MPR121_ECR           0x5E
#define MPR121_AUTOCONFIG0   0x7B
#define MPR121_DATA_BASE_ADDR 0x04

#define NUM_FINGERS 3

const float BENT_OFFSETS[NUM_FINGERS] = {1.0, 0.8, 0.7}; 
float fingerThresholds[NUM_FINGERS] = {0, 0, 0};

bool  fingerCalibrated = false;
bool  fingerBent[NUM_FINGERS] = {false, false, false};

// =====================================================
// [C] 손가락 굽힘 조합
// =====================================================
#define COMBO_NONE              0x00
#define COMBO_MIDDLE            0x01
#define COMBO_RING              0x02
#define COMBO_MIDDLE_RING       0x03
#define COMBO_RING_PINKY        0x06
#define COMBO_MIDDLE_RING_PINKY 0x07

uint8_t fingerCombo = COMBO_NONE;

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

  if (!sensor.begin_I2C(0x18)) {
    Serial.println("MLX90393 not found!");
    while (1) { delay(10); }
  }
  sensor.setGain(MLX90393_GAIN_1X);
  sensor.setResolution(MLX90393_X, MLX90393_RES_17);
  sensor.setResolution(MLX90393_Y, MLX90393_RES_17);
  sensor.setResolution(MLX90393_Z, MLX90393_RES_16);
  sensor.setOversampling(MLX90393_OSR_3);
  sensor.setFilter(MLX90393_FILTER_5);

  if (!drv.begin()) {
    Serial.println("DRV2605 not found!");
    while (1) { delay(10); }
  }
  drv.selectLibrary(1);
  drv.setMode(DRV2605_MODE_INTTRIG);

  mpr121_writeRegister(MPR121_SOFT_RESET, 0x63);
  delay(10);
  mpr121_writeRegister(MPR121_ECR, 0x00);
  mpr121_writeRegister(MPR121_AUTOCONFIG0, 0x00);
  mpr121_writeRegister(MPR121_AFE_CONFIG, 0x95);
  mpr121_writeRegister(MPR121_FILTER_CONFIG, 0xD0);
  mpr121_writeRegister(MPR121_ECR, 0x80 | NUM_FINGERS);

  pinMode(BUTTON_MAG_CAL,    INPUT_PULLUP);
  pinMode(BUTTON_FINGER_CAL, INPUT_PULLUP);
  pinMode(BUTTON_SERIAL,     INPUT_PULLUP);
}

// =====================================================
void loop() {
  unsigned long now = millis();

  // ===== [1] 버튼 처리 =====
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

  if (digitalRead(BUTTON_FINGER_CAL) == LOW && (now - lastFingerCalPress > DEBOUNCE_MS)) {
    lastFingerCalPress = now;
    calibrateFingers();
  }

  if (digitalRead(BUTTON_SERIAL) == LOW && (now - lastSerialPress > DEBOUNCE_MS)) {
    lastSerialPress = now;
    serialEnabled = !serialEnabled;
  }

  // ===== [2] Magnetometer 읽기 =====
  sensors_event_t event;
  sensor.getEvent(&event);

  float value_x = event.magnetic.x - x_offset;
  float value_y = event.magnetic.y - y_offset;
  float value_z = event.magnetic.z - z_offset;

  // ===== [3] 상태머신 (Norm 기반 Hysteresis 적용) =====
  float magnitude = sqrt(sq(value_x) + sq(value_y) + sq(value_z));

  uint8_t targetState = uiState; // 기본값은 현재 상태 유지 (1300~1500 데드존 처리용)
  if (magnitude >= THRESHOLD_ON_MIN) {
    targetState = 2; // ON
  } else if (magnitude >= THRESHOLD_HOVER_MIN && magnitude < THRESHOLD_HOVER_MAX) {
    targetState = 3; // HOVERING
  } else if (magnitude < THRESHOLD_OFF_MAX) {
    targetState = 0; // OFF
  }

  // 상태 전환 로직
  if (targetState == 2) {
    if (uiState == 0 || uiState == 3) {
      uiState = 1; // 500ms 대기 진입
      detectionStartTime = now;
    } else if (uiState == 1) {
      if (now - detectionStartTime >= ACTIVATION_TIME_MS) {
        uiState = 2; // 완전 활성화
      }
    }
  } else {
    // ON이 아니면 즉시 상태 변경 (0 또는 3)
    uiState = targetState;
    detectionStartTime = 0;
    if (uiState != 2) {
      prevZone = -1;
      wasInCenter = false;
      wasOutside = false;
    }
  }

  // ===== [4] Heading & Displacement =====
  float heading = atan2(value_y, value_x) * 180.0 / PI;
  float B_planar = sqrt(sq(value_x) + sq(value_y));
  float B_z_abs = abs(event.magnetic.z);
  if (B_z_abs < 0.1) B_z_abs = 0.1;

  float displacement_angle = atan(B_planar / (k_sensitivity * B_z_abs));
  float displacement = C_mm_per_rad * displacement_angle * (180.0 / PI);

  // ===== [5] Haptic 피드백 =====
  if (uiState == 2) {
    // 활성화 상태: 조이스틱 변위에 따른 연속 햅틱
    handleHapticFeedback(displacement);
  } else if (uiState == 3) {
    // Hovering 상태: 0.8초마다 더블 클릭 진동 (기존과 확실히 구분)
    if (now - lastHoverBuzzTime >= HOVER_BUZZ_INTERVAL_MS) {
      lastHoverBuzzTime = now;
      playEffect(HOVER_EFFECT); 
    }
  }

  // ===== [6] Finger Bend 읽기 =====
  updateFingerBend();

  // ===== [7] Serial 출력 =====
  if (serialEnabled) {
    uint8_t outState = uiState;  // 0=OFF 1=WAITING 2=ON 3=HOVERING

    Serial.print("X_corr:"); Serial.print(value_x); Serial.print(",");
    Serial.print("Y_corr:"); Serial.print(value_y); Serial.print(",");
    Serial.print("Z_corr:"); Serial.print(value_z); Serial.print(",");
    Serial.print("Heading:"); Serial.print(heading); Serial.print(",");
    Serial.print("Displacement:"); Serial.print(displacement); Serial.print(",");
    Serial.print("State:"); Serial.print(outState); Serial.print(",");
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
void handleHapticFeedback(float displacement) {
  unsigned long now = millis();
  bool inCenter = (displacement <= CENTER_DEADZONE);
  int  zone = getZone(displacement);

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
void updateFingerBend() {
  fingerCombo = COMBO_NONE;
  for (int i = 0; i < NUM_FINGERS; i++) {
    uint8_t regAddr = MPR121_DATA_BASE_ADDR + (i * 2);
    uint16_t rawData = mpr121_read16(regAddr);
    float capacitance = (rawData > 0) ? 104260.0 / (float)rawData : 0;
    fingerBent[i] = fingerCalibrated && (capacitance >= fingerThresholds[i] + BENT_OFFSETS[i]);
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