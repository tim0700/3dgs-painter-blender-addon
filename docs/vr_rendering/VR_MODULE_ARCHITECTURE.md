# VR 모듈 아키텍처 가이드

> 3DGS Painter의 VR 페인팅 시스템에 대한 기술 문서입니다.
> 컴퓨터공학 학부생 수준으로 작성되었습니다.

---

## 목차

1. [시스템 개요](#시스템-개요)
2. [모듈 구조](#모듈-구조)
3. [핵심 파일 설명](#핵심-파일-설명)
4. [데이터 흐름](#데이터-흐름)
5. [OpenXR 통신 구조](#openxr-통신-구조)

---

## 시스템 개요

### VR 페인팅이란?

VR 헤드셋(Meta Quest 3)을 착용하고, VR 컨트롤러로 공중에 3D Gaussian을 그리는 기능입니다.

### 기술 스택

```
┌─────────────────────────────────────────────────────────────┐
│                    Blender 5.0                               │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   Python Addon   │ ──►│   Shared Memory (Windows)       │ │
│  │   (src/vr/*.py)  │    │   (Named: 3DGS_Gaussian_Data)   │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│              OpenXR API Layer (C++/DLL)                      │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ gaussian_layer.dll│◄──│   OpenGL Rendering              │ │
│  │ (openxr_layer/)  │    │   (VR 헤드셋에 직접 렌더링)      │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                  Meta Quest 3 (OpenXR)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 모듈 구조

### 파일 구성

```
src/vr/
├── __init__.py           # 모듈 초기화 및 등록
├── vr_session.py         # VR 세션 관리 (시작/종료)
├── vr_input.py           # 컨트롤러 입력 처리
├── vr_operators.py       # Blender 오퍼레이터 정의
├── vr_freehand_paint.py  # 자유 페인팅 로직 (핵심!)
├── vr_shared_memory.py   # C++ 레이어와 통신
├── action_maps.py        # VR 버튼 바인딩
├── vr_panels.py          # UI 패널
└── vr_action_binding.py  # 액션 바인딩 유틸리티
```

---

## 핵심 파일 설명

### 1. vr_session.py - VR 세션 관리

```python
class VRSessionManager:
    """VR 세션 상태를 관리하는 싱글톤 클래스"""

    def start_vr_session(self):
        """VR 세션 시작 - Blender의 xr_session_toggle 호출"""
        bpy.ops.wm.xr_session_toggle()

    def is_session_running(self):
        """현재 VR 세션이 실행 중인지 확인"""
        return wm.xr_session_state.is_running(bpy.context)

    def get_viewer_pose(self):
        """HMD(헤드셋) 위치와 회전 반환"""
        location = xr_state.viewer_pose_location
        rotation = xr_state.viewer_pose_rotation
```

**핵심 개념:**

- **싱글톤 패턴**: `_instance`로 하나의 인스턴스만 유지
- **XR Session State**: Blender가 제공하는 VR 상태 객체

---

### 2. vr_input.py - 컨트롤러 입력

```python
@dataclass
class ControllerState:
    """컨트롤러 한 개의 상태"""
    hand: ControllerHand        # LEFT or RIGHT
    position: Vector            # 3D 위치 (x, y, z)
    rotation: Quaternion        # 회전 (w, x, y, z 쿼터니언)
    trigger_value: float        # 트리거 값 (0.0 ~ 1.0)
    is_active: bool             # 추적 중인지

class VRInputManager:
    def get_controller_state(self, hand):
        """컨트롤러 위치/회전 정보 가져오기"""
        grip_pos = xr_state.controller_grip_location_get(ctx, index)
        grip_rot = xr_state.controller_grip_rotation_get(ctx, index)
```

**핵심 개념:**

- **Grip vs Aim**: Grip은 손잡이 위치, Aim은 레이저 방향
- **Quaternion**: 3D 회전을 표현하는 4차원 벡터 (짐벌락 방지)

---

### 3. vr_freehand_paint.py - 자유 페인팅 (★ 가장 중요)

```python
class THREEGDS_OT_VRFreehandPaint(Operator):
    """VR 자유 페인팅 오퍼레이터"""

    def invoke(self, context, event):
        """페인팅 시작 - 타이머 등록"""
        self._timer = wm.event_timer_add(1/60, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        """매 프레임 호출 - 컨트롤러 위치 샘플링"""
        if event.type == 'TIMER':
            self._on_timer_tick(context)
        return {'PASS_THROUGH'}

    def _on_timer_tick(self, context):
        """트리거 상태 확인 → 페인팅"""
        trigger_value = self._get_trigger_value()
        if trigger_value > 0.5:
            self._continue_stroke(context, position, rotation)

    def _continue_stroke(self, position, rotation):
        """현재 위치에 Gaussian 추가"""
        gaussians = generate_gaussians_at_point(position, rotation, size, color)
        self.session.stroke_painter.add_gaussians(gaussians)
```

**핵심 개념:**

- **Modal Operator**: Blender에서 지속적 이벤트를 받는 방법
- **Timer**: 1/60초마다 `_on_timer_tick` 호출
- **PASS_THROUGH**: 다른 이벤트도 처리되도록 허용

---

### 4. vr_shared_memory.py - C++ 레이어 통신 (★ IPC)

```python
# 상수 정의
SHARED_MEMORY_NAME = "Local\\3DGS_Gaussian_Data"
HEADER_SIZE = 180  # 헤더 크기 (바이트)
GAUSSIAN_SIZE = 56 # Gaussian 하나당 크기

class SharedMemoryWriter:
    def create(self):
        """Windows Named Shared Memory 생성"""
        self._mmap = mmap.mmap(-1, BUFFER_SIZE, tagname=SHARED_MEMORY_NAME)

    def write_gaussians(self, gaussians, view_matrix, proj_matrix):
        """Gaussian 데이터를 공유 메모리에 쓰기"""
        header = struct.pack('<5I', magic, version, frame_id, count, flags)
        header += view_matrix.tobytes()
        header += proj_matrix.tobytes()

        for g in gaussians:
            data += struct.pack('<3f 4f 3f 4f',
                g.position, g.rotation, g.scale, g.color)

        self._mmap.seek(0)
        self._mmap.write(header + data)
```

**핵심 개념:**

- **Shared Memory**: 두 프로세스가 메모리를 공유
- **struct.pack**: Python 데이터를 바이트로 변환 (C와 호환)
- **Little Endian (<)**: 바이트 순서 지정

#### 데이터 구조 (C++ 헤더와 동일)

```
┌───────────────────────────────────────────────────────┐
│ SharedMemoryHeader (180 bytes)                         │
├───────────────────────────────────────────────────────┤
│ magic (4B)        │ version (4B)   │ frame_id (4B)    │
│ gaussian_count(4B)│ flags (4B)     │                  │
│ view_matrix (64B) │ proj_matrix (64B)                 │
│ camera_rotation (16B) │ camera_position (12B+4B pad)  │
├───────────────────────────────────────────────────────┤
│ GaussianPrimitive[0] (56 bytes)                       │
│   position (12B) │ rotation (16B)                     │
│   scale (12B)    │ color (16B)                        │
├───────────────────────────────────────────────────────┤
│ GaussianPrimitive[1] (56 bytes)                       │
│ ...                                                   │
└───────────────────────────────────────────────────────┘
```

---

### 5. action_maps.py - VR 버튼 바인딩

```python
def disable_teleport_action():
    """텔레포트 액션을 비활성화 (트리거=페인팅만)"""
    teleport_ami = am.actionmap_items.get("teleport")
    _teleport_original_op = teleport_ami.op
    teleport_ami.op = "threegds.noop"  # 아무것도 안하는 연산자

class THREEGDS_OT_NoOp(Operator):
    """아무것도 하지 않는 연산자 (텔레포트 대체용)"""
    bl_idname = "threegds.noop"

    def execute(self, context):
        return {'FINISHED'}
```

**핵심 개념:**

- **ActionMap**: Blender VR의 버튼→기능 매핑 시스템
- **XrAttachSessionActionSets**: 세션 시작 후 액션 수정 불가 (OpenXR 제약)

---

## 데이터 흐름

### 페인팅 데이터 흐름

```
[VR 컨트롤러]
    │ 트리거 입력
    ▼
[vr_freehand_paint.py]
    │ 위치 샘플링 (60Hz)
    │ Gaussian 생성
    ▼
[vr_shared_memory.py]
    │ struct.pack()
    │ mmap.write()
    ▼
[공유 메모리: 3DGS_Gaussian_Data]
    │
    ▼
[gaussian_layer.dll (C++)]
    │ 메모리 읽기
    │ OpenGL 렌더링
    ▼
[VR 헤드셋 화면]
```

### 뷰 매트릭스 업데이트 흐름

```
[Blender 카메라]
    │ camera.matrix_world
    ▼
[vr_operators.py: _vr_matrix_update_callback()]
    │ 30Hz 타이머
    │ 뷰 매트릭스 계산
    ▼
[vr_shared_memory.py: update_matrices()]
    │ struct.pack() → mmap
    ▼
[gaussian_renderer.cpp]
    │ 캐릭터 위치 변환
    ▼
[VR에서 정확한 3D 위치]
```

---

## OpenXR 통신 구조

### Blender VR 아키텍처

```
Blender Python API
    │
    ├── window_manager.xr_session_state
    │       ├── viewer_pose_location     (HMD 위치)
    │       ├── viewer_pose_rotation     (HMD 회전)
    │       ├── controller_grip_location (컨트롤러 위치)
    │       └── actionmaps               (버튼 바인딩)
    │
    └── operators.wm.xr_session_toggle() (세션 시작/종료)

OpenXR Runtime (Meta XR)
    │
    ├── xrLocateViews()         (눈 위치 계산)
    ├── xrGetActionStateFloat() (트리거 값 읽기)
    └── xrEndFrame()            (프레임 제출)
```

### 알려진 제한사항

| 제한사항                 | 원인                                          | 현재 상태 |
| ------------------------ | --------------------------------------------- | --------- |
| B 버튼 바인딩 불가       | xrAttachSessionActionSets 이후 액션 추가 불가 | 해결 불가 |
| 텔레포트 비활성화 불안정 | 세션 중 ActionMap 수정은 race condition       | 70% 작동  |

---

## 다음 문서

- [VR 페인팅 상세 코드 분석](./VR_FREEHAND_PAINT_DETAIL.md)
- [공유 메모리 프로토콜](./VR_SHARED_MEMORY_PROTOCOL.md)
- [VR Gaussian 위치 버그 해결](./VR_GAUSSIAN_POSITION_FIX.md)
