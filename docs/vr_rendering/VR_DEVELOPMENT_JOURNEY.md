# VR 페인팅 개발 여정 (Development Journey)

> 2025년 12월 7-9일, 이틀간의 개발 과정 총정리
> 한 학기 팀프로젝트 최종 목표: **VR에서 3D Gaussian Splatting 페인팅**

---

## 📅 개발 타임라인

```
12월 7일 (토)  ────────────────────────────────────────────────►
   │
   ├─ Phase 1: 문제 분석 & 연구
   │     "왜 Blender VR에서 GLSL이 안 되지?"
   │
   ├─ Phase 2: OpenXR API Layer 설계
   │     "xrEndFrame을 후킹하면 될 것 같다!"
   │
   └─ Phase 3: C++ DLL 구현 시작
         "Quad 레이어로 일단 색상 테스트..."

12월 8일 (일)  ────────────────────────────────────────────────►
   │
   ├─ Phase 3.5: Projection Layer로 전환
   │     "Quad는 2D였네... 스테레오 3D로 바꿔야!"
   │
   ├─ Phase 4: Python ↔ C++ 통신
   │     "공유 메모리로 Gaussian 데이터 전송"
   │
   └─ Phase 5: VR 페인팅 통합
         ❌ "왜 Gaussian 위치가 이상하지...?"

12월 9일 (월)  ────────────────────────────────────────────────►
   │
   ├─ Phase 5.1: 위치 버그 수정
   │     ✅ "좌표 변환 중복 제거하니 해결!"
   │
   ├─ Phase 5.2: 트리거 입력 문제
   │     ❌ "텔레포트랑 충돌... 크래시 발생"
   │
   ├─ Phase 5.3: 크래시 해결
   │     ✅ "No-op 연산자로 안전하게 비활성화"
   │
   └─ Phase 6: 문서화 & 마무리
         ✅ "기술 문서 3개 작성, 폴더 정리"
```

---

## 🚀 Phase 1: 문제 분석 (12월 7일)

### 상황

- PC에서 3D Gaussian Splatting은 잘 작동
- **VR 헤드셋에서는 Gaussian이 안 보임**

### 발견

```
[문제] bpy.gpu draw_handler가 VR에서 호출 안 됨!

[원인] Blender wm_xr_draw.c에서 overlay pass 스킵
       → draw_handler는 overlay에서만 동작
       → VR은 overlay 렌더링 안 함
```

### 연구 결과 (5가지 방법 검토)

| 방법                   | 난이도      | 문제점               |
| ---------------------- | ----------- | -------------------- |
| draw_handler 직접 사용 | 쉬움        | ❌ VR에서 안됨       |
| gpu.offscreen + Plane  | 중간        | ❌ 2D만 가능         |
| Custom RenderEngine    | 어려움      | ❌ 전체 렌더러 대체  |
| Geometry Nodes         | 중간        | ❌ 성능 이슈         |
| **OpenXR API Layer**   | 매우 어려움 | ✅ **유일한 해결책** |

---

## 🔧 Phase 2-3: OpenXR API Layer 개발 (12월 7-8일)

### 핵심 아이디어

```
Blender ─────► OpenXR Runtime ─────► VR 헤드셋
                    │
                    ▼
           gaussian_layer.dll  ← 여기 끼어들기!
           (우리가 만든 DLL)
```

### 첫 번째 시도: XrCompositionLayerQuad

```cpp
// xrEndFrame()에서 2D Quad 주입
XrCompositionLayerQuad quadLayer;
quadLayer.type = XR_TYPE_COMPOSITION_LAYER_QUAD;
quadLayer.pose.position = {0, 0, -2.0f};  // 정면 2m 앞
```

**결과**: ✅ 빨간 사각형이 VR에 보임!
**문제**: ❌ 2D 평면이라 머리 움직이면 따라다님 (Head-locked)

### 두 번째 시도: XrCompositionLayerProjection

```cpp
// 스테레오 렌더링 (양쪽 눈)
XrCompositionLayerProjection projLayer;
projLayer.viewCount = 2;  // Left, Right
projLayer.views[0] = leftEyeView;
projLayer.views[1] = rightEyeView;
```

**결과**: ✅ World-locked 3D 렌더링 성공!

---

## 🔗 Phase 4: Python ↔ C++ 통신 (12월 8일)

### 설계

```
┌─────────────────────────────────────────────────────┐
│ Python (Blender Addon)                               │
│                                                      │
│  gaussians = paint_at_position(controller_pos)      │
│  shared_memory.write(gaussians, view_matrix)        │
│                        │                             │
└────────────────────────┼─────────────────────────────┘
                         │  mmap (Named Shared Memory)
                         │  "Local\\3DGS_Gaussian_Data"
┌────────────────────────┼─────────────────────────────┐
│                        ▼                             │
│ C++ (gaussian_layer.dll)                             │
│                                                      │
│  buffer = MapViewOfFile("Local\\3DGS_Gaussian_Data")│
│  renderer.Draw(buffer.gaussians, buffer.view_mat)   │
└─────────────────────────────────────────────────────┘
```

### 데이터 구조 (56 bytes/Gaussian)

```cpp
struct GaussianPrimitive {
    float position[3];   // 12B
    float color[4];      // 16B
    float scale[3];      // 12B
    float rotation[4];   // 16B
};
```

---

## 🐛 Phase 5: 버그와의 전쟁 (12월 8-9일)

### 버그 1: Gaussian 위치 이상

**증상**: VR에서 Gaussian이 수백 미터 밖에 렌더링됨

**디버깅 과정**:

```
1. 콘솔 로그로 위치 출력
   → Controller: (7.2, -6.7, 4.9)  ← 정상
   → 실제 렌더링: (723, -672, 491) ← 100배?!

2. 좌표 변환 추적
   Python: 월드 좌표 그대로 전송
   C++: 뷰 매트릭스로 변환
        + 카메라 오프셋 적용  ← 이게 중복!
        + VR 뷰 매트릭스 적용 ← 또 변환!

3. 해결책: C++에서 중복 변환 제거
```

**수정 코드**:

```cpp
// Before (버그)
vec3 worldPos = cameraOffset + inputPos;  // ❌ 불필요한 변환
vec4 viewPos = uViewMatrix * vec4(worldPos, 1.0);

// After (수정)
vec4 viewPos = uViewMatrix * vec4(inputPos, 1.0);  // ✅ 직접 변환
```

### 버그 2: 트리거 크래시

**증상**: TRIGGER 누르면 Blender 크래시

```
EXCEPTION_ACCESS_VIOLATION in VCRUNTIME140.dll
wm_operator_create → idp_generic_copy
```

**원인 분석**:

```
1. 텔레포트 비활성화 시도
   teleport_ami.op = ""  // 빈 문자열

2. Blender XR 시스템이 빈 연산자 생성 시도
   wm_operator_create("") → 크래시!
```

**해결책**:

```python
# No-op 연산자 생성
class THREEGDS_OT_NoOp(Operator):
    bl_idname = "threegds.noop"
    def execute(self, context):
        return {'FINISHED'}  # 아무것도 안 함

# 텔레포트를 no-op으로 대체
teleport_ami.op = "threegds.noop"  # 크래시 방지
```

**결과**: ~70% 안정 (Blender XR 구조적 한계)

---

## 📝 최종 결과물

### 코드베이스

```
src/vr/
├── vr_session.py         # VR 세션 관리
├── vr_freehand_paint.py  # 자유 페인팅 (680줄)
├── vr_shared_memory.py   # C++ 통신 (355줄)
├── action_maps.py        # 버튼 바인딩 (309줄)
└── vr_operators.py       # 오퍼레이터 (523줄)

openxr_layer/
├── src/
│   ├── xr_dispatch.cpp       # OpenXR 후킹 (372줄)
│   ├── projection_layer.cpp  # 스테레오 렌더링 (463줄)
│   ├── gaussian_renderer.cpp # GLSL 셰이더 (673줄)
│   └── shared_memory.cpp     # 공유 메모리 (100줄)
└── include/
    └── gaussian_data.h       # 데이터 구조
```

### 문서

```
docs/vr_rendering/
├── README.md                     # 개요
├── VR_MODULE_ARCHITECTURE.md     # Python 분석
├── OPENXR_LAYER_ARCHITECTURE.md  # C++ 분석
├── VR_SETUP_GUIDE.md             # 설정 가이드
└── VR_DEVELOPMENT_JOURNEY.md     # 이 문서
```

---

## 💡 배운 것들

### 기술적 인사이트

1. **OpenXR API Layer**: 런타임과 앱 사이에 끼어드는 강력한 방법
2. **IPC (공유 메모리)**: Python ↔ C++ 실시간 통신
3. **스테레오 렌더링**: 양쪽 눈에 다른 뷰 매트릭스 적용
4. **좌표 변환**: 중복 변환 = 버그의 원흉

### 교훈

1. **작은 단계로**: Quad → Projection Layer 점진적 발전
2. **로그가 생명**: `[VR DEBUG]` 없으면 디버깅 불가능
3. **문서화 중요**: 2일 뒤의 나도 모르는 코드가 됨

---

## 🎯 한계 및 미래 과제

### 현재 한계

- 텔레포트 비활성화 70% 안정 (Blender XR 구조적 한계)
- B 버튼 바인딩 불가 (OpenXR 세션 시작 후 액션 추가 불가)

### 미래 개선 방향

1. **Blender VR Addon PR**: ActionMap 수정 API 개선 요청
2. **XML 설정 파일**: 사용자별 버튼 매핑 지원
3. **성능 최적화**: Gaussian 개수 증가 시 FPS 저하 대응

---

## 🙏 마무리

이 프로젝트는 **Blender VR + OpenXR + GLSL + IPC**를 결합한
복잡한 시스템 통합 작업이었습니다.

2일 만에 작동하는 프로토타입을 완성한 것은
많은 시행착오와 디버깅의 결과입니다.

**한 학기 팀프로젝트 최종 목표: ✅ 달성!**
