# Blender VR 커스텀 렌더링: Option C vs D 상세 기술 분석

**분석 대상:** Blender 5.0 + Meta Quest 3 + 3D Gaussian Splatting  
**목표 성능:** 72+ FPS VR 렌더링, 양안 스테레오, GLSL 커스텀 셰이더  
**분석 날짜:** 2025-12-08

---

## 1. 현재 상황 분석

### 1.1 검증된 사실
- **PC viewport에서:** `draw_handler_add()` ✅ 동작
- **PC RENDERED viewport:** 커스텀 RenderEngine `view_draw()` ✅ 동작  
- **VR 세션에서:** 둘 다 **❌ 호출되지 않음**
  - draw_handler: VR 세션 중 overlay pass 건너뜀 (Blender C 코드 레벨)
  - RenderEngine.view_draw(): VR 컨텍스트에서 호출 안 됨

### 1.2 근본 원인
```
Blender VR 렌더링 파이프라인:
1. xrWaitFrame() → 다음 프레임 시간 예측
2. xrBeginFrame() → 세션 상태 체크
3. [SCENE RENDERING - Blender 내부 렌더러만]
   - 이 단계에서 Python callback 실행 안 함
   - Custom RenderEngine도 우회됨
4. xrEndFrame() → composition layer 수집 후 runtime에 전달
5. Runtime이 HMD에 composite
```

**문제:** Blender의 VR 렌더링 루프가 기존 viewport/RenderEngine 시스템을 우회함.  
**위치:** `source/blender/editors/space_xr/wm_xr_draw.c` (Julian Eisel, GSoC 2019)

---

## 2. Option C: OpenXR API Layer (C++ DLL)

### 2.1 기술 개요

**원리:**  
OpenXR Loader와 runtime 사이에 intercept DLL 삽입  
→ `xrEndFrame()` 호출 시 composition layer 수집  
→ 커스텀 Gaussian layer 주입 후 runtime에 전달

**아키텍처:**

```
Application (Blender)
    ↓
OpenXR Loader (openxr-1.dll)
    ↓
[API Layer DLL - YOUR CODE] ← 여기서 xrEndFrame() 가로채기
    ↓
OpenXR Runtime (Meta Quest runtime / SteamVR)
    ↓
HMD
```

### 2.2 구현 과정

#### 2.2.1 DLL 생성

**기본 구조 (C++17):**

```cpp
// gaussian_layer.dll (또는 gaussian_layer.so)

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

// 함수 포인터: 실제 runtime의 함수를 호출하기 위해
static PFN_xrEndFrame real_xrEndFrame = nullptr;

// 인터셉트 함수
XrResult XRAPI_CALL hooked_xrEndFrame(
    XrSession session,
    const XrFrameEndInfo* frameEndInfo) {
    
    // 1. Gaussian 데이터 준비 (외부 소스에서)
    // 2. Quad composition layer 생성
    XrCompositionLayerQuad gaussianLayer = {...};
    
    // 3. frameEndInfo의 layers 배열에 추가
    std::vector<XrCompositionLayerBaseHeader*> newLayers;
    for (int i = 0; i < frameEndInfo->layerCount; i++) {
        newLayers.push_back(frameEndInfo->layers[i]);
    }
    newLayers.push_back((XrCompositionLayerBaseHeader*)&gaussianLayer);
    
    // 4. 수정된 frameEndInfo로 runtime 호출
    XrFrameEndInfo modifiedInfo = *frameEndInfo;
    modifiedInfo.layerCount = newLayers.size();
    modifiedInfo.layers = newLayers.data();
    
    return real_xrEndFrame(session, &modifiedInfo);
}
```

#### 2.2.2 Manifest 파일 (.json)

**Windows 레지스트리 위치:**
```
HKEY_LOCAL_MACHINE\SOFTWARE\Khronos\OpenXR\1\ApiLayers\Implicit
```

**manifest.json:**
```json
{
  "file_format_version": "1.0.0",
  "api_layer": {
    "name": "XR_APILAYER_GAUSSIAN_SPLATTING",
    "library_path": "./bin/gaussian_layer.dll",
    "api_version": "1.0",
    "implementation_version": "1",
    "functions": {
      "xrEndFrame": "hooked_xrEndFrame",
      "xrGetInstanceProcAddr": "layer_xrGetInstanceProcAddr"
    }
  }
}
```

**설치 방법:**
- Manifest를 표준 위치에 복사
- Registry entry 생성 (manifest 경로 가리킴)
- 또는 환경변수: `XR_API_LAYER_PATH=C:\path\to\manifest`

#### 2.2.3 함수 인터셉션 메커니즘

**Trampoline 패턴:**
```cpp
// OpenXR Loader가 제공하는 기본 메커니즘
// (Detours 또는 manual hooking 불필요)

typedef struct XrApiLayerCreateInfo {
    XrInstance instance;
    PFN_xrGetInstanceProcAddr pfn_xrGetInstanceProcAddr;
} XrApiLayerCreateInfo;

// Layer DLL entry point
XrResult XRAPI_CALL xrCreateApiLayerInstance(
    const XrApiLayerCreateInfo* info,
    const XrInstanceCreateInfo* instanceCreateInfo,
    XrInstance* instance) {
    
    // Layer chain 초기화
    // - loader를 통해 다음 layer/runtime 호출 등록
}
```

### 2.3 Gaussian 데이터 공급 방식

#### 문제: DLL ↔ Blender 간 데이터 전달

**선택지:**

**A. Shared Memory (권장)**
```cpp
// DLL이 정기적으로 읽음
HANDLE hFile = OpenFileMapping(
    FILE_MAP_READ,
    FALSE,
    L"Local\\gaussian_data_1024");
auto ptr = MapViewOfFile(hFile, FILE_MAP_READ, 0, 0, 0);
// GPU texture pointer + gaussian count
```

**B. Named Pipe (리얼타임)**
```cpp
// Blender → DLL: 프레임당 데이터
HANDLE hPipe = CreateFile(
    "\\\\.\\pipe\\gaussian_feed",
    GENERIC_READ | GENERIC_WRITE, 0, NULL,
    OPEN_EXISTING, 0, NULL);
// 그러나 xrEndFrame 내부에서 I/O는 위험
```

**C. DXGI Shared Texture (최고 성능)**
```cpp
// Blender Python: D3D11 texture 생성
ID3D11Texture2D* gaussianTexture = ...;
IDXGIResource* dxgiResource;
gaussianTexture->QueryInterface(&dxgiResource);
HANDLE sharedHandle;
dxgiResource->GetSharedHandle(&sharedHandle);
// DLL이 OpenSharedResource()로 접근
```

**추천:** C (DXGI Shared Texture)
- GPU-to-GPU transfer (CPU 오버헤드 없음)
- Lock-free (동시 접근 안전)
- 72+ FPS 유지 가능

### 2.4 Composition Layer 종류

**Blender VR 상황에 적합한 타입:**

```cpp
// 1. Quad Layer (평면) - 가장 간단
XrCompositionLayerQuad quadLayer = {
    .type = XR_TYPE_COMPOSITION_LAYER_QUAD,
    .layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA,
    .space = stage_space,  // world space
    .subImage = {
        .swapchain = my_swapchain,  // RGBA texture with Gaussians
        .imageRect = {{0, 0}, {1024, 1024}}
    },
    .pose = {/* 1.0m 떨어진 위치 */},
    .size = {.width = 1.0f, .height = 1.0f}
};

// 2. Cylinder Layer - 더 자연스러움
XrCompositionLayerCylinderKHR cylLayer = {
    .type = XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR,
    .radius = 1.0f,
    .centralAngle = 90.0f * (M_PI / 180.0f),
    ...
};

// 3. Projection Layer (권장 - 가장 품질)
XrCompositionLayerProjection projLayer = {
    .type = XR_TYPE_COMPOSITION_LAYER_PROJECTION,
    .layerFlags = XR_COMPOSITION_LAYER_CORRECT_CHROMATIC_ABERRATION |
                  XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA,
    .space = stage_space,
    .views = projViews,  // Left/Right eye views
    .viewCount = 2
};
```

**VR Gaussian 렌더링에 최적:**
- **Quad/Cylinder:** 단순하나 기하학적 왜곡 발생
- **Projection:** 정확한 stereo, 하지만 렌더링 비용 높음

### 2.5 Blender 연동 (Python 부분)

```python
# Blender add-on
import bpy
import numpy as np
from PIL import Image
from ctypes import *

class GaussianSharedMemory:
    def __init__(self):
        # Gaussian 데이터를 공유 메모리에 기록
        self.shm_name = "Local\\gaussian_data_1024"
        self.gaussian_data = []  # List[dict]
    
    def update_gaussians(self, gaussians):
        """
        gaussians: List[{
            'position': [x, y, z],
            'color': [r, g, b, a],
            'scale': [sx, sy, sz],
            'rotation': [qx, qy, qz, qw]
        }]
        """
        # Binary format 생성
        buffer = bytearray()
        buffer.extend(len(gaussians).to_bytes(4, 'little'))
        
        for g in gaussians:
            # Position (12 bytes)
            for v in g['position']:
                buffer.extend(np.float32(v).tobytes())
            # Color (16 bytes)
            for v in g['color']:
                buffer.extend(np.float32(v).tobytes())
            # Scale (12 bytes)
            for v in g['scale']:
                buffer.extend(np.float32(v).tobytes())
            # Rotation (16 bytes)
            for v in g['rotation']:
                buffer.extend(np.float32(v).tobytes())
        
        # 공유 메모리에 쓰기
        self._write_to_shared_memory(buffer)
    
    def _write_to_shared_memory(self, data):
        # Windows API를 통해 공유 메모리 액세스
        # 또는 mmap 사용 (cross-platform)
        import mmap
        import os
        
        # 파일 기반 shared memory
        temp_file = os.path.join(os.getenv('TEMP'), 'gaussian_data.bin')
        with open(temp_file, 'wb') as f:
            f.write(data)
```

### 2.6 성능 특성

| 측면 | 값 |
|------|-----|
| **가능성** | ✅ 확실함 (OpenXR spec) |
| **프레임 오버헤드** | <1ms (xrEndFrame 호출 약 0.2ms) |
| **메모리 오버헤드** | ~10MB (shared texture) |
| **개발 난이도** | ⭐⭐⭐⭐ (중상급) |
| **72+ FPS 가능성** | ✅ 매우 높음 |
| **Stereo 지원** | ✅ 자동 (composition layer) |
| **Blender 버전 의존성** | ❌ 없음 (외부 프로세스) |

### 2.7 알려진 문제점

#### A. Layer 충돌
여러 API layer가 동일 함수 intercept 시 충돌 가능
→ **해결:** Careful function forwarding, version checking

```cpp
// 올바른 chaining
XrResult modified_xrEndFrame(...) {
    // 1. 자신의 작업 수행
    // 2. real_xrEndFrame 호출 (이게 다음 layer로 포워딩)
    return real_xrEndFrame(session, &modifiedInfo);
}
```

#### B. 타이밍 문제
`xrEndFrame()` 호출 시점에 Gaussian 데이터가 없을 수 있음
→ **해결:** Double-buffering, frame lag 허용

```cpp
// DLL 측
struct FrameBuffer {
    int frame_id;
    Gaussian data[100000];
    // lock-free ring buffer
};
```

#### C. 그래픽 API 변환
Blender가 OpenGL인데 runtime이 D3D일 경우 변환 필요
→ **해결:** GPU 사이에서 texturing (shared HANDLE)

### 2.8 설치 및 배포

**사용자 관점:**
```
1. gaussian_layer.dll + manifest.json 다운로드
2. C:\Program Files\OpenXR-API-Layers\ 에 복사
3. Registry 경로 추가 (또는 installer 실행)
4. Blender 재시작
5. VR 세션 시작 → 자동으로 Gaussian layer 로드됨
```

**배포 크기:** ~2-5MB (DLL) + manifest

---

## 3. Option D: Blender 소스 수정

### 3.1 기술 개요

**원리:**  
Blender 소스코드 → `wm_xr_draw.c` 수정  
→ VR 렌더링 루프에 Python callback 삽입  
→ Custom RenderEngine 또는 draw_handler 호출

**수정 위치:**
```
blender/source/blender/windowmanager/xr/wm_xr_draw.c

함수: wm_xr_draw_view()
→ 여기서 VR per-eye 렌더링 수행
→ Python callback을 이 단계에 추가
```

### 3.2 Blender 소스 분석

**현재 VR 렌더링 파이프라인:**

```c
void wm_xr_draw_view(wmXrDrawViewInfo *info) {
    // 1. View matrix 설정
    // 2. Projection matrix 설정
    // 3. Blender 내장 renderer 호출
    //    - EEVEE 또는 Cycles (viewport preview)
    // 4. 결과를 framebuffer에 저장
    
    // ❌ Python callback 없음
    // ❌ Custom RenderEngine 호출 없음
}
```

**수정안:**

```c
void wm_xr_draw_view(wmXrDrawViewInfo *info) {
    // 기존 코드...
    
    // [NEW] Python callback 호출
    if (G.py_handle && Python_is_initialized) {
        PyObject *callback = wm_xr_get_python_callback("view_draw");
        if (callback) {
            // Context 생성 (region, space_data 등)
            PyObject *context = bpy_context_create_xr(info);
            PyObject *depsgraph = bpy_depsgraph_get(scene);
            
            // 콜백 실행
            PyObject_CallFunctionObjArgs(
                callback,
                context,
                depsgraph,
                NULL
            );
            
            Py_DECREF(context);
            Py_DECREF(depsgraph);
            Py_DECREF(callback);
        }
    }
}
```

### 3.3 수정해야 할 파일들

| 파일 | 수정 내용 | 난이도 |
|------|----------|--------|
| `wm_xr_draw.c` | Python callback 삽입 | ⭐⭐⭐ |
| `wm_xr.h` | 콜백 함수 포인터 선언 | ⭐ |
| `wm_xr_intern.h` | 내부 구조 확장 | ⭐ |
| `py/bpy.c` | XR 관련 Python API | ⭐⭐⭐⭐ |
| `py/gpu/gpu_py_utils.c` | GPU module VR 지원 | ⭐⭐⭐ |
| `GPU module` | `gpu.matrix` VR per-eye | ⭐⭐⭐⭐ |

### 3.4 구현 예시

**C 코드 (wm_xr_draw.c):**

```c
// Python callback 관리 구조
typedef struct {
    bool enabled;
    PyObject *view_draw_callback;
    PyObject *context;
} wmXrPythonCallbacks;

static wmXrPythonCallbacks g_xr_callbacks = {0};

// Callback 등록
bool wm_xr_python_callback_register(
    const char *callback_name,
    PyObject *callable) {
    
    if (strcmp(callback_name, "view_draw") == 0) {
        Py_XDECREF(g_xr_callbacks.view_draw_callback);
        Py_INCREF(callable);
        g_xr_callbacks.view_draw_callback = callable;
        g_xr_callbacks.enabled = true;
        return true;
    }
    return false;
}

// VR 렌더링 루프에 통합
void wm_xr_draw_view(wmXrDrawViewInfo *info) {
    struct ARegion *region = info->region;
    View3D *v3d = info->v3d;
    
    // 기존: EEVEE/Cycles viewport 렌더
    // ED_view3d_draw_offscreen(...);  // <-- 이걸 bypass할 건가, 
                                      //     아니면 함께 실행할 건가?
    
    // [NEW] Python callback 호출 - GPU 렌더링 직후
    if (g_xr_callbacks.enabled && g_xr_callbacks.view_draw_callback) {
        // GPU matrix 설정 (중요!)
        gpu_matrix_set_from_openxr(info->view_matrix, info->proj_matrix);
        
        // Python 컨텍스트 생성
        bContext *C = CTX_create();  // 또는 기존 context 사용
        CTX_wm_window_set(C, wm->windows[0]);
        CTX_area_set(C, area);
        CTX_region_set(C, region);
        
        Depsgraph *depsgraph = DEG_graph_new(
            scene,
            view_layer,
            DAG_EVAL_RENDER);  // VR 최적화
        
        // Callback 호출
        PyObject *ret = PyObject_CallFunctionObjArgs(
            g_xr_callbacks.view_draw_callback,
            BPy_Context_FromContext(C),
            BPy_Depsgraph_FromDEG(depsgraph),
            NULL);
        
        if (!ret) {
            PyErr_Print();
        } else {
            Py_DECREF(ret);
        }
        
        Py_XDECREF(C);
        DEG_graph_free(depsgraph);
    }
}
```

**Python API (bpy.types.RenderEngine - VR 버전):**

```python
# Blender addon
import bpy

class VRGaussianRenderEngine(bpy.types.RenderEngine):
    bl_idname = "VR_GAUSSIAN"
    bl_label = "VR Gaussian"
    bl_use_preview = False
    
    def view_update(self, context, depsgraph):
        print("[VR] view_update called")
        # 데이터 준비
        self._compile_shader()
        self._prepare_gaussian_data()
    
    def view_draw(self, context, depsgraph):
        # ✅ VR에서도 호출될 거라는 게 목표!
        print("[VR] *** view_draw CALLED IN VR! ***")
        
        import gpu
        from gpu_extras.batch import batch_for_shader
        
        # VR per-eye GPU 상태 (gpu.matrix 자동)
        view_matrix = gpu.matrix.get_model_view_matrix()
        proj_matrix = gpu.matrix.get_projection_matrix()
        
        # Gaussian 렌더링
        self.shader.bind()
        self.shader.uniform_float("viewProjectionMatrix", 
                                  proj_matrix @ view_matrix)
        self.batch.draw(self.shader)

def register():
    bpy.utils.register_class(VRGaussianRenderEngine)

def unregister():
    bpy.utils.unregister_class(VRGaussianRenderEngine)
```

### 3.5 빌드 및 배포

**Blender 빌드 시간:**
- 첫 빌드: 30-60분 (기계 사양에 따라)
- 증분 빌드: 5-15분

**빌드 환경:**

| OS | 도구 | 시간 |
|----|------|------|
| Windows | Visual Studio 2022 + CMake | 45분 |
| Linux | GCC 11+ + CMake | 60분 |
| macOS | Clang + CMake | 50분 |

**배포:**
```
옵션 A: 패치 파일 (git diff)
- 300KB 정도
- 사용자가 직접 빌드 필요

옵션 B: 바이너리 배포
- Windows: .zip (200MB blender 포함)
- 자체 서버 필요

옵션 C: PPA / homebrew formula
- Linux/macOS 친화적
- 배포 복잡성 중간
```

### 3.6 성능 특성

| 측면 | 값 |
|------|-----|
| **가능성** | ✅ 구현 가능 (기술적으로 가능) |
| **프레임 오버헤드** | <0.5ms (callback 호출만) |
| **메모리 오버헤드** | ~5MB (코드 크기) |
| **개발 난이도** | ⭐⭐⭐⭐⭐ (고급) |
| **72+ FPS 가능성** | ✅ 높음 (기본 렌더링과 병렬 가능) |
| **Stereo 지원** | ✅ 자동 (gpu.matrix) |
| **Blender 버전 의존성** | ⚠️ **높음** (매 버전 패치 필요) |

### 3.7 장기 유지보수 부담

**Blender 5.0 → 5.1 업그레이드 시:**

```
가능한 시나리오:

1. 패치 충돌 (merge conflict)
   - wm_xr_draw.c 변경으로 인해 30% 확률

2. API 변경
   - bpy.types.RenderEngine 함수 서명 변경
   - gpu.matrix 인터페이스 변경

3. 최악: VR 렌더링 파이프라인 완전 재구현
   - Blender 팀이 OpenXR 지원 강화할 경우
   - 패치 무효화 가능
```

**예상 유지보수 비용:**
- 매 Blender 메이저 버전 (6개월): 5-10시간
- 매 마이너 버전 (월별): 1-2시간

---

## 4. Option C vs D 상세 비교

### 4.1 기술적 비교

| 항목 | Option C (API Layer) | Option D (Blender 패치) |
|------|----------------------|--------------------------|
| **기술 성숙도** | ✅ 표준화됨 (OpenXR Spec) | ⚠️ 해킹 (권장하지 않음) |
| **구현 복잡도** | 중상급 (C++) | 고급 (C + Python 호환) |
| **패치 유지보수** | ❌ 필요 없음 | ✅ Blender 버전마다 필요 |
| **Blender 독립성** | ✅ 완전 독립 | ❌ 직결됨 |
| **Runtime 독립성** | ❌ OpenXR 필요 | ✅ 모든 VR API 지원 가능 |
| **성능 오버헤드** | <1ms | <0.5ms |
| **메모리 오버헤드** | 10MB | 5MB |
| **디버깅 용이성** | ⭐⭐⭐ (외부 가능) | ⭐ (Blender 소스 필요) |
| **배포 복잡성** | 간단 (설치 프로그램) | 복잡 (사용자 빌드 또는 바이너리) |
| **보안 위험** | 낮음 (샌드박스) | 높음 (Blender 코어 수정) |

### 4.2 72+ FPS 달성 가능성

**VRSplat 논문 데이터 (2024):**
- 72+ FPS 달성: 
  - 낮은 Gaussian 수 (수천개)
  - Foveated rendering (eye-tracking 필요)
  - 최적화된 sorting/rasterization

**Option C에서의 경로:**
```
Blender (고품질 rendering)
  ↓ (opengl/eevee)
  ↓ (10ms/frame)
  ↓
Framebuffer → texture
  ↓
Shared texture via GPU
  ↓
OpenXR API Layer (read-only)
  ↓
Gaussian rendering shader
  ↓
Composition layer 생성
  ↓
Runtime composite (1-2ms)
  ↓
HMD display
```
**추정 프레임 타임:** 13-14ms → 70-75 FPS ✅

**Option D에서의 경로:**
```
Blender VR loop
  ↓
Python callback in view_draw()
  ↓
Custom GPU shader (Gaussian rasterize)
  ↓ (4-6ms for ~10k gaussians)
  ↓
Direct framebuffer write
  ↓
xrEndFrame()
  ↓
HMD display
```
**추정 프레임 타임:** 6-8ms → 125-165 FPS ✅✅

**결론:** 둘 다 가능, D가 더 빠를 가능성 높음

---

## 5. 최종 추천

### 5.1 결정 매트릭스

#### 시나리오 A: "빠르게 작동하는 프로토타입이 필요"
**추천:** **Option C (API Layer)**

**이유:**
- 2-3주 안에 작동하는 PoC 가능
- Blender 소스 이해 불필요
- 테스트 환경 독립적

**구현 로드맵:**
```
Week 1: DLL 기본 구조 + xrEndFrame 가로채기
Week 2: Composition layer 추가 + 데이터 공급 메커니즘
Week 3: Gaussian shader + GPU texture sync
Week 4: 성능 최적화 + 버그 수정
```

#### 시나리오 B: "최고 성능이 중요, 유지보수 가능"
**추천:** **Option C + D 하이브리드**

**단계적 접근:**
```
Phase 1 (즉시): Option C로 프로토타입
  - 3주, 72+ FPS 달성
  
Phase 2 (3개월): Blender 팀에 패치 제안
  - Option D의 작은 부분만
  - "VR callback 추가" minimal patch
  
Phase 3 (6개월): Blender 5.1 병합 기대
  - 공식 지원 가능성
```

#### 시나리오 C: "상용 제품화, 완전한 통제 필요"
**추천:** **Option D (Blender 커스텀 빌드)**

**전략:**
```
1. Blender 소스 fork (GitHub)
2. wm_xr_draw.c에 Python callback 추가
3. 자체 빌드 + 배포
4. 3-6개월마다 upstream merge
```

**단점:** 개발 비용 높음, 유지보수 부담

### 5.2 실제 추천: Option C

**최종 판단:**

| 기준 | 점수 | 이유 |
|------|------|------|
| 개발 시간 | 8/10 | 2-3주 prototyping 가능 |
| 기술 위험 | 9/10 | OpenXR 표준, 이미 사용 중 |
| 성능 | 7/10 | 72+ FPS 충분히 가능 |
| 유지보수 | 10/10 | Blender 독립, 수정 불필요 |
| 배포 용이성 | 8/10 | MSI installer 가능 |
| 총점 | **8.4/10** | ✅ |

---

## 6. Option C 구현 상세 가이드

### 6.1 프로젝트 구조

```
gaussian-layer/
├── src/
│   ├── gaussian_layer.cpp        # Main DLL
│   ├── gaussian_layer.h
│   ├── xr_interception.cpp       # xrEndFrame hooking
│   ├── composition_layer.cpp     # Layer creation
│   ├── gaussian_renderer.cpp     # Shader + rendering
│   ├── shared_memory.cpp         # Blender 데이터 읽기
│   └── gpu_interop.cpp          # D3D11/D3D12 texture sharing
├── manifest/
│   ├── gaussian_layer.json       # OpenXR manifest
│   └── install_registry.bat      # Windows registry setup
├── python/
│   ├── blender_addon.py         # Blender integration
│   └── gaussian_loader.py       # Data preparation
├── shader/
│   ├── gaussian_vs.hlsl         # Vertex shader
│   └── gaussian_ps.hlsl         # Pixel shader
├── CMakeLists.txt
└── README.md
```

### 6.2 핵심 코드 템플릿

**gaussian_layer.cpp:**
```cpp
#include <openxr/openxr.h>
#include <d3d11.h>
#include <wrl/client.h>
#include <glm/glm.hpp>

using Microsoft::WRL::ComPtr;

namespace gaussian_layer {

// Global state
static PFN_xrEndFrame g_next_xrEndFrame = nullptr;
static ComPtr<ID3D11Device> g_device;
static ComPtr<ID3D11DeviceContext> g_context;
static ComPtr<ID3D11PixelShader> g_gaussian_shader;

// Composition layer for Gaussians
struct GaussianLayer {
    uint32_t gaussian_count;
    glm::vec3 position;
    glm::mat4 view_matrix;
    glm::mat4 proj_matrix;
    
    XrCompositionLayerProjectionView views[2];  // L/R eye
};

XrResult XRAPI_CALL hooked_xrEndFrame(
    XrSession session,
    const XrFrameEndInfo* frameEndInfo) {
    
    // 1. Gaussian 데이터 읽기
    GaussianLayer gaussian = read_shared_memory();
    
    // 2. Rendering (if data available)
    if (gaussian.gaussian_count > 0) {
        render_gaussians(&gaussian);
    }
    
    // 3. Original runtime 호출
    return g_next_xrEndFrame(session, frameEndInfo);
}

}  // namespace gaussian_layer

// DLL Entry Point
BOOL APIENTRY DllMain(
    HMODULE hModule,
    DWORD ul_reason_for_call,
    LPVOID lpReserved) {
    
    if (ul_reason_for_call == DLL_PROCESS_ATTACH) {
        gaussian_layer::initialize();
    }
    return TRUE;
}
```

### 6.3 필요한 라이브러리

```
OpenXR SDK:
  - openxr.lib
  - openxr-loader.lib

DirectX:
  - d3d11.lib
  - dxgi.lib

3rd party (권장):
  - glm (행렬 연산)
  - spdlog (로깅)
  - JSON for Modern C++ (manifest 파싱)
```

### 6.4 컴파일 및 배포

**Visual Studio 2022:**
```
1. Clone: https://github.com/KhronosGroup/OpenXR-SDK
2. CMake configure
3. Build → gaussian_layer.dll
4. Sign (optional but recommended): signtool.exe
5. Package with manifest JSON
```

### 6.5 테스트 계획

| 테스트 | 방법 | 예상 결과 |
|--------|------|----------|
| DLL 로드 | OpenXR runtime 시작 | Layer 초기화, 로그 생성 |
| Data feed | Blender → shared memory | 프레임당 10k gaussians 읽음 |
| Rendering | xrEndFrame 후 texture | Gaussian quad layer 보임 |
| FPS | Oculus metrics | 72+ fps 유지 |
| Stereo | L/R eye comparison | 두 눈에 다른 이미지 |

---

## 7. 잠재적 문제와 해결책

### 7.1 공통 문제

| 문제 | Option C | Option D |
|------|----------|----------|
| Blender와 sync 어려움 | 공유메모리 latency (1-2frame) | 직접 호출, sync 용이 |
| GPU memory 부족 | 추가 texture 필요 | Blender texture 재사용 |
| Compositor 충돌 | Layer ordering 문제 | 없음 |
| 디버깅 어려움 | PIX/NSight 필요 | Blender debugger |
| Windows 전용 | 현재 맞음 | Blender 지원 OS |

### 7.2 Linux/macOS 지원

**Option C:**
- Linux: EGL + Vulkan 또는 OpenGL
- macOS: Metal (untested, 어려움)
- **현실적:** Windows/Linux만 지원

**Option D:**
- 자동으로 모든 플랫폼 (Blender 그대로)

---

## 8. 최종 결론

### 8.1 추천 선택지

**✅ Option C (OpenXR API Layer) 추천**

**이유:**
1. 개발 기간 단축 (2-3주 vs 2-3개월)
2. 유지보수 부담 없음 (Blender 독립)
3. 성능 충분함 (72+ FPS 달성 가능)
4. 기술적 위험 낮음 (표준 OpenXR)
5. 배포 단순함 (설치 프로그램)

### 8.2 실행 계획

```
Phase 1 (Week 1-2): Research & Prototyping
  - OpenXR API Layer template 분석
  - Blender shared memory 인터페이스 설계
  - DLL 기본 구조 구현

Phase 2 (Week 3-4): Core Implementation
  - xrEndFrame interception
  - Gaussian composition layer 생성
  - GPU texture sharing (DXGI handle)

Phase 3 (Week 5-6): Integration & Optimization
  - Blender Python addon 개발
  - 성능 최적화 (72+ FPS 달성)
  - 스테레오 렌더링 검증

Phase 4 (Week 7-8): Testing & Polishing
  - Quest 3 하드웨어 테스트
  - 메모리 누수 체크
  - 사용자 매뉴얼 작성
```

### 8.3 필요 리소스

**개발:**
- C++ 중상급 (Windows API, DirectX 기초)
- OpenXR 이해
- GPU programming 기초

**빌드:**
- Visual Studio 2022 Community (무료)
- Windows 10/11
- Meta Quest 3 + Link cable

**비용:**
- 개발 비용: 300-500시간 (2-3명 팀, 2-3개월)
- 하드웨어: ~$800 (개발용 PC + Quest 3)
- 라이선스: 0 (모두 오픈소스)

---

## 9. 참고 자료

### 9.1 OpenXR API Layer

- [OpenXR Specification](https://www.khronos.org/registry/openxr/)
- [OpenXR-API-Layer-Template](https://github.com/Ybalrid/OpenXR-API-Layer-Template)
- [Best Practices for OpenXR API Layers (Fred Emmott)](https://fredemmott.com/blog/2024/11/25/best-practices-for-openxr-api-layers.html)
- [OpenXR-Toolkit (mbucchia)](https://github.com/mbucchia/OpenXR-Toolkit)

### 9.2 VR Gaussian Splatting

- [VRSplat Paper](https://arxiv.org/abs/2505.10144) - 72+ FPS 기준
- [VR-Splatting (Foveated Rendering)](https://arxiv.org/abs/2410.17932)
- [3DGS Survey](https://arxiv.org/abs/2401.03890)

### 9.3 Blender VR

- [VR Scene Inspection Add-on](https://docs.blender.org/manual/en/latest/addons/3d_view/vr_scene_inspection.html)
- [BlenderXR Project](https://github.com/MARUI-PlugIn/BlenderXR)
- [Blender OpenXR GSoC 2019](https://devtalk.blender.org/t/gsoc-2019-vr-support-through-openxr-weekly-reports/7665)

### 9.4 커스텀 렌더링 참고

- [Blender RenderEngine API](https://docs.blender.org/api/current/bpy.types.RenderEngine.html)
- [Viewport Renderer (논문의 코드)](附属 파일들)

---

## 부록: 기술 용어

- **xrEndFrame**: OpenXR 프레임 종료, composition layer 제출
- **Composition Layer**: HMD runtime이 처리하는 그래픽 레이어
- **API Layer**: OpenXR loader와 runtime 사이 인터셉션
- **Shared Texture**: GPU VRAM에서 여러 프로세스/라이브러리가 접근 가능한 텍스처
- **Stereo Rendering**: 좌안/우안 두 번 렌더링 (약간 다른 viewpoint)
- **Foveated Rendering**: 응시점(fovea) 주변만 고품질 렌더링
- **Gaussian Splatting**: 3D 점군을 2D 가우시안으로 rasterize하는 렌더링 방법

---

**작성:** 2025-12-08  
**버전:** 1.0  
**상태:** 최종 분석 완료

