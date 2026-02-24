import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import re
import sys
import time  # [추가] 시간 측정을 위해 라이브러리 추가

# --- [설정 구간] ---
# PORT = '/dev/cu.usbmodem1301'   # Mac
# PORT = 'COM3'                   # Windows
PORT = '/dev/ttyACM0'            # Linux

BAUD_RATE = 115200
MAX_RADIUS = 80.0 
MAGNET_THRESHOLD = 90.0 # Z_corr 절대값이 이보다 커야 ON으로 인식

# --- [시리얼 연결] ---
try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    print(f"✅ 장치 연결 성공: {PORT}")
except Exception as e:
    print(f"❌ 연결 실패: {e}")
    sys.exit()

# --- [그래프 설정] ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='polar')

# 1. 나침반 설정
ax.set_theta_zero_location('N') # 0도 = 북쪽
ax.set_theta_direction(1)       # 반시계 방향 (CCW) 증가

# 2. 반지름 설정
ax.set_rlim(0, MAX_RADIUS)
ax.set_rticks(np.linspace(0, MAX_RADIUS, 4))
ax.set_yticklabels([]) 

# 3. 메인 화살표 (아날로그 스틱 스타일)
arrow_line, = ax.plot([], [], color='red', linewidth=5, marker='o', markevery=[-1])
center_dot, = ax.plot([0], [0], 'ko', markersize=5) # 중심점

# 4. 텍스트 정보
title_text = ax.set_title("Analog Stick Interface", va='bottom', fontsize=15, fontweight='bold')
state_text = fig.text(0.5, 0.02, "State: NEUTRAL", ha='center', fontsize=14, color='black')

# 오른쪽 아래 Interface ON/OFF 표시 (초기설정)
interface_text = fig.text(0.85, 0.03, "Interface: OFF", ha='center', fontsize=12, color='white', fontweight='bold', 
                          bbox=dict(boxstyle='square,pad=0.5', facecolor='red', alpha=0.8, edgecolor='none'))

# 전역 변수
current_heading = 0.0
current_displacement = 0.0
current_z_corr = 0.0 

# [추가] 상태 관리를 위한 변수
ui_state = "OFF"         # "OFF", "WAITING", "ON"
detection_start_time = 0 # 감지 시작 시간 저장

def update(frame):
    global current_heading, current_displacement, current_z_corr
    global ui_state, detection_start_time
    
    # 시리얼 읽기
    while ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            # 정규표현식 파싱
            match_head = re.search(r"Heading:([-0-9.]+)", line)
            match_disp = re.search(r"Displacement:([-0-9.]+)", line)
            match_z = re.search(r"Z_corr:([-0-9.]+)", line)
            
            if match_head: current_heading = float(match_head.group(1))
            if match_disp: current_displacement = float(match_disp.group(1))
            if match_z: current_z_corr = float(match_z.group(1))
        except:
            pass
    
    # --- [1. 상태 머신 로직 (State Machine)] ---
    is_magnet_detected = abs(current_z_corr) >= MAGNET_THRESHOLD
    current_time = time.time()

    if is_magnet_detected:
        if ui_state == "OFF":
            # 자석이 처음 감지됨 -> WAITING 상태로 진입
            ui_state = "WAITING"
            detection_start_time = current_time
        
        elif ui_state == "WAITING":
            # 0.2초 경과 체크
            if current_time - detection_start_time >= 0.2:
                ui_state = "ON"  # 0.2초 지났으면 진짜 ON
            # 0.2초 안 지났으면 WAITING 유지
            
        elif ui_state == "ON":
            pass # 이미 ON이면 유지
            
    else:
        # 자석이 사라지면 즉시 OFF 리셋 (WAITING 중이어도 리셋됨)
        ui_state = "OFF"
        detection_start_time = 0

    # --- [2. 그래프 및 UI 업데이트] ---
    rad = np.deg2rad(current_heading)
    display_disp = min(current_displacement, MAX_RADIUS)
    
    # 2-1. UI 상태별 처리
    if ui_state == "OFF":
        display_disp = 0
        arrow_line.set_data([], []) # 화살표 숨기기
        
        # 빨간색 OFF 박스
        interface_text.set_text("Interface: OFF")
        interface_text.set_color('white')
        interface_text.set_bbox(dict(facecolor='red', alpha=0.8, edgecolor='none'))
        state_text.set_text("State: NO MAGNET")

    elif ui_state == "WAITING":
        display_disp = 0 # 아직 ON 아님 -> 화살표 숨기기 (혹은 보이게 하려면 여기 수정)
        arrow_line.set_data([], []) 
        
        # 파란색 WAITING 박스
        remaining = 0.2 - (current_time - detection_start_time)
        interface_text.set_text(f"WAITING... {remaining:.1f}s")
        interface_text.set_color('white')
        interface_text.set_bbox(dict(facecolor='blue', alpha=0.8, edgecolor='none'))
        state_text.set_text("State: DETECTING...")

    elif ui_state == "ON":
        # 초록색 ON 박스
        interface_text.set_text("Interface: ON")
        interface_text.set_color('white')
        interface_text.set_bbox(dict(facecolor='green', alpha=0.8, edgecolor='none'))
        
        # 데드존(중앙) 처리
        is_moving = display_disp > (MAX_RADIUS * 0.35)
        
        if is_moving:
            arrow_line.set_data([0, rad], [0, display_disp])
            arrow_line.set_color('red') # 움직임 활성 (빨강)
            state_text.set_text(f"Moving: {current_heading:.1f}°")
        else:
            arrow_line.set_data([0, rad], [0, display_disp])
            arrow_line.set_color('gray') # 중앙 대기 (회색)
            state_text.set_text("State: NEUTRAL")

    # 제목 업데이트
    title_text.set_text(f"Heading: {current_heading:.1f}°")
    
    return arrow_line, title_text, state_text, interface_text

ani = animation.FuncAnimation(fig, update, interval=30, blit=False)
plt.show()

ser.close()