# OpenXR API 레이어 아키텍처 가이드

> OpenXR API Layer (C++/DLL)에 대한 기술 문서입니다.

---

## 목차

1. [OpenXR API Layer란?](#openxr-api-layer란)
2. [디렉토리 구조](#디렉토리-구조)
3. [핵심 파일 설명](#핵심-파일-설명)
4. [렌더링 파이프라인](#렌더링-파이프라인)
5. [공유 메모리 통신](#공유-메모리-통신)
6. [빌드 및 설치](#빌드-및-설치)

---

## OpenXR API Layer란?

### 개념

OpenXR은 VR/AR 애플리케이션을 위한 표준 API입니다.
**API Layer**는 OpenXR 런타임과 애플리케이션 사이에 끼어들어
함수 호출을 가로채고 추가 기능을 삽입하는 방법입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Blender (애플리케이션)                     │
│                       xrEndFrame() 호출                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              gaussian_layer.dll (우리 레이어)                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ 1. Blender의 xrEndFrame() 가로채기                       ││
│  │ 2. 공유 메모리에서 Gaussian 데이터 읽기                   ││
│  │ 3. OpenGL로 Gaussian을 VR에 렌더링                       ││
│  │ 4. 원래 OpenXR 런타임으로 전달                            ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Meta Quest Runtime (OpenXR 런타임)              │
│                     VR 헤드셋에 표시                          │
└─────────────────────────────────────────────────────────────┘
```

### 왜 API Layer를 사용하나요?

Blender VR 내부에서 커스텀 셰이더를 실행하기 어렵기 때문입니다.
API Layer를 통해:

- Blender를 수정하지 않고 VR 렌더링에 개입
- 3D Gaussian Splatting 같은 고급 기법 구현 가능

---

## 디렉토리 구조

```
openxr_layer/
├── CMakeLists.txt          # CMake 빌드 설정
├── README.md               # 간단한 설명
│
├── include/                # 헤더 파일
│   ├── gaussian_data.h     # 공유 메모리 데이터 구조
│   ├── gaussian_renderer.h # Gaussian 렌더러
│   ├── projection_layer.h  # 스테레오 렌더링
│   ├── xr_dispatch.h       # OpenXR 함수 후킹
│   ├── shared_memory.h     # 공유 메모리 읽기
│   └── gpu_context.h       # OpenGL 컨텍스트
│
├── src/                    # 소스 파일
│   ├── main.cpp            # DLL 진입점
│   ├── xr_dispatch.cpp     # ★ OpenXR 함수 가로채기
│   ├── projection_layer.cpp# ★ VR 스테레오 렌더링
│   ├── gaussian_renderer.cpp# ★ GLSL 셰이더
│   ├── shared_memory.cpp   # Python과 통신
│   ├── composition_layer.cpp # Quad 레이어
│   └── gpu_context.cpp     # GPU 초기화
│
├── manifest/               # OpenXR 레이어 등록
│   └── XR_APILAYER_3DGS.json
│
├── external/               # 외부 라이브러리
│   └── OpenXR SDK 헤더들
│
└── build/                  # 빌드 출력
    └── Release/gaussian_layer.dll
```

---

## 핵심 파일 설명

### 1. xr_dispatch.cpp - OpenXR 함수 후킹 (★ 진입점)

```cpp
/**
 * OpenXR API 함수를 가로채는 핵심 로직
 *
 * xrEndFrame()을 후킹하여 우리 렌더링을 삽입합니다.
 */

// xrEndFrame이 호출될 때 실행됨
XrResult gaussian_xrEndFrame(
    XrSession session,
    const XrFrameEndInfo* frameEndInfo)
{
    // 1. 공유 메모리에서 Gaussian 데이터 읽기
    g_sharedMemory.Read();

    // 2. 각 눈(left/right)에 Gaussian 렌더링
    projLayer.LocateViews(displayTime);

    for (int eye = 0; eye < 2; eye++) {
        projLayer.BeginRenderEye(eye);

        renderer.RenderFromPrimitivesWithMatrices(
            gaussians, count,
            viewMatrix, projMatrix, ...);

        projLayer.EndRenderEye();
    }

    // 3. 렌더링된 레이어를 frameEndInfo에 추가
    modifiedInfo.layerCount = layerCount + 1;  // 우리 레이어 추가

    // 4. 원래 런타임으로 전달
    return pfn_xrEndFrame(session, &modifiedInfo);
}
```

**핵심 개념:**

- **Function Hooking**: 라이브러리 함수 호출을 가로채는 기법
- **XrCompositionLayerProjection**: VR에서 3D 스테레오 렌더링을 위한 레이어

---

### 2. projection_layer.cpp - 스테레오 VR 렌더링

```cpp
/**
 * VR 스테레오(좌/우 눈) 렌더링 관리
 *
 * OpenXR Swapchain을 생성하고 각 눈에 렌더링합니다.
 */

class ProjectionLayer {
    // 각 눈의 스왑체인 (더블 버퍼링)
    XrSwapchain m_swapchains[2];  // [0]=왼쪽, [1]=오른쪽

    // 각 눈의 뷰/프로젝션 매트릭스
    float m_viewMatrices[2][16];
    float m_projMatrices[2][16];

    // 눈 위치 계산
    bool LocateViews(XrTime displayTime) {
        // OpenXR에서 각 눈의 3D 위치/방향 가져오기
        pfn_xrLocateViews(session, &viewLocateInfo, &viewState,
                          viewCount, views);

        // 뷰 매트릭스 계산 (눈 위치의 역변환)
        ComputeViewMatrix(views[eye].pose, m_viewMatrices[eye]);

        // 프로젝션 매트릭스 계산 (FOV 기반)
        ComputeProjectionMatrix(views[eye].fov, 0.1f, 100.0f,
                                m_projMatrices[eye]);
    }

    // 특정 눈으로 렌더링 시작
    GLuint BeginRenderEye(uint32_t eyeIndex) {
        // 해당 눈의 스왑체인 이미지 획득
        pfn_xrAcquireSwapchainImage(m_swapchains[eyeIndex], ...);

        // FBO에 바인딩하여 렌더링 준비
        glBindFramebuffer(GL_FRAMEBUFFER, m_fbos[eyeIndex]);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
};
```

**핵심 개념:**

- **Swapchain**: GPU가 렌더링하는 동안 다른 이미지를 표시하는 버퍼 시스템
- **FBO (Framebuffer Object)**: 텍스처에 렌더링하기 위한 OpenGL 객체
- **View Matrix**: 카메라(눈) 위치/방향의 역변환
- **Projection Matrix**: 3D→2D 투영 (원근법)

---

### 3. gaussian_renderer.cpp - GLSL 셰이더 렌더링

```cpp
/**
 * 3D Gaussian을 OpenGL로 렌더링
 *
 * 각 Gaussian을 빌보드 쿼드로 그립니다.
 */

// Vertex Shader (정점 셰이더)
static const char* VERTEX_SHADER = R"(
#version 330 core

// 인스턴스당 데이터
layout(location = 0) in vec3 aPos;        // Gaussian 위치
layout(location = 1) in vec4 aColor;      // 색상 (RGBA)
layout(location = 2) in vec3 aScale;      // 크기
layout(location = 4) in vec2 aQuadPos;    // 쿼드 코너 (-1~1)

uniform mat4 uViewMatrix;
uniform mat4 uProjMatrix;

void main() {
    // 1. 월드 좌표를 클립 좌표로 변환
    vec4 posClip = uProjMatrix * uViewMatrix * vec4(aPos, 1.0);

    // 2. 빌보드: 화면에 항상 정면을 향하도록
    float sizeNDC = clamp(aScale.x * 0.1, 0.02, 0.3);
    posClip.xy += aQuadPos * sizeNDC * posClip.w;

    gl_Position = posClip;
    vColor = aColor;
    vCoordXY = aQuadPos;
}
)";

// Fragment Shader (픽셀 셰이더)
static const char* FRAGMENT_SHADER = R"(
#version 330 core

in vec4 vColor;
in vec2 vCoordXY;
out vec4 fragColor;

void main() {
    // 원형 마스크: 중심에서 멀어질수록 투명
    float dist = length(vCoordXY);
    if (dist > 1.0) discard;

    // 가우시안 폴오프 (부드러운 가장자리)
    float alpha = exp(-dist * dist * 2.0);
    fragColor = vec4(vColor.rgb, vColor.a * alpha);
}
)";

// 렌더링 함수
void RenderFromPrimitivesWithMatrices(
    const GaussianPrimitive* gaussians,
    uint32_t count,
    const float* viewMatrix,
    const float* projMatrix, ...)
{
    // 1. 셰이더 활성화
    glUseProgram(m_shaderProgram);

    // 2. 유니폼 전달 (뷰/프로젝션 매트릭스)
    glUniformMatrix4fv(m_viewMatrixLoc, 1, GL_FALSE, viewMatrix);
    glUniformMatrix4fv(m_projMatrixLoc, 1, GL_FALSE, projMatrix);

    // 3. Gaussian 데이터를 GPU 버퍼에 업로드
    // ... (인스턴스 데이터 준비)

    // 4. 인스턴스 렌더링
    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, count);
}
```

**핵심 개념:**

- **Instanced Rendering**: 같은 메시를 여러 번 그릴 때 효율적인 방법
- **Billboard**: 항상 카메라를 향하는 쿼드 (회전 무시)
- **Gaussian Falloff**: `exp(-x²)`로 부드러운 가장자리

---

### 4. gaussian_data.h - 공유 메모리 프로토콜

```cpp
/**
 * Python ↔ C++ 간 데이터 교환 형식
 *
 * Python (vr_shared_memory.py)과 정확히 일치해야 함!
 */

// 단일 Gaussian (56 바이트)
struct GaussianPrimitive {
    float position[3];   // 12 bytes: x, y, z
    float color[4];      // 16 bytes: r, g, b, a
    float scale[3];      // 12 bytes: sx, sy, sz
    float rotation[4];   // 16 bytes: w, x, y, z (쿼터니언)
};

// 공유 메모리 헤더 (180 바이트)
struct SharedMemoryHeader {
    uint32_t magic;           // 0x33444753 ("3DGS")
    uint32_t version;         // 1
    uint32_t frame_id;        // 증가하는 프레임 ID
    uint32_t gaussian_count;  // Gaussian 개수
    uint32_t flags;           // 예약됨
    float view_matrix[16];    // 4x4 뷰 매트릭스 (64 bytes)
    float proj_matrix[16];    // 4x4 프로젝션 매트릭스 (64 bytes)
    float camera_rotation[4]; // 카메라 회전 (16 bytes)
    float camera_position[3]; // 카메라 위치 (12 bytes)
    float _padding;           // 정렬 패딩 (4 bytes)
};

// 전체 메모리 레이아웃
struct SharedMemoryBuffer {
    SharedMemoryHeader header;                    // 180 bytes
    GaussianPrimitive gaussians[MAX_GAUSSIANS];   // N * 56 bytes
};
```

**핵심 개념:**

- **alignas(4)**: 4바이트 정렬로 CPU/GPU 효율 향상
- **static_assert**: 컴파일 타임에 크기 검증

---

## 렌더링 파이프라인

### 전체 흐름

```
[VR 프레임 사이클]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ xrWaitFrame()                                                │
│   → VR 런타임이 다음 프레임 시간 제공                          │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ xrBeginFrame()                                               │
│   → 렌더링 시작 신호                                          │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ [Blender VR 렌더링]                                           │
│   → 일반 Blender 씬 렌더링                                    │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ xrEndFrame() → gaussian_xrEndFrame() ← 우리가 후킹!          │
│                                                              │
│   1. SharedMemory에서 Gaussian 읽기                          │
│   2. 각 눈에 Gaussian 렌더링 (OpenGL)                        │
│   3. XrCompositionLayerProjection 생성                       │
│   4. 원래 xrEndFrame()에 레이어 추가해서 전달                 │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ [VR 헤드셋에 표시]                                            │
│   Blender 렌더링 + Gaussian 렌더링 합성                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 빌드 및 설치

### 요구사항

- Windows 10/11
- Visual Studio 2019/2022 (C++ 빌드 도구)
- CMake 3.16+
- OpenXR SDK 1.0+

### 빌드 방법

```powershell
# 1. 빌드 디렉토리 생성
cd openxr_layer
mkdir build
cd build

# 2. CMake 구성
cmake .. -G "Visual Studio 17 2022" -A x64

# 3. 빌드
cmake --build . --config Release

# 결과: build/Release/gaussian_layer.dll
```

### 설치 방법

1. **레이어 등록** (레지스트리에 추가)

```powershell
# 관리자 권한으로 실행
reg add "HKLM\SOFTWARE\Khronos\OpenXR\1\ApiLayers\Implicit" `
    /v "C:\경로\XR_APILAYER_3DGS.json" /t REG_DWORD /d 0
```

2. **manifest 파일 수정** (`XR_APILAYER_3DGS.json`)

```json
{
  "file_format_version": "1.0.0",
  "api_layer": {
    "name": "XR_APILAYER_3DGS_gaussian",
    "library_path": "./gaussian_layer.dll", // DLL 경로
    "api_version": "1.0",
    "implementation_version": "1",
    "description": "3DGS Gaussian Rendering Layer"
  }
}
```

3. **Blender 재시작** 후 VR 시작

### 디버깅

```powershell
# 로그 파일 확인
Get-Content $env:LOCALAPPDATA\XR_APILAYER_3DGS\dispatch.log
Get-Content $env:LOCALAPPDATA\XR_APILAYER_3DGS\projection.log
```

---

## 다음 문서

- [Python VR 모듈 아키텍처](./VR_MODULE_ARCHITECTURE.md)
- [공유 메모리 프로토콜 상세](./VR_SHARED_MEMORY_PROTOCOL.md)
- [VR 설정 가이드](./VR_SETUP_GUIDE.md)
