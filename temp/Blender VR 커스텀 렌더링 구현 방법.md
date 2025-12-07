# **Blender VR 환경 내 3D Gaussian Splatting 커스텀 렌더링 아키텍처 및 구현 전략 보고서**

## **1\. 서론 (Introduction)**

본 연구 보고서는 Blender 5.0 환경에서 Meta Quest 3(Oculus Link)를 사용하여 VR 세션 내에 3D Gaussian Splatting(이하 3DGS)을 렌더링하기 위한 기술적 구현 방안을 심층적으로 분석한다. 사용자는 기존의 Python API인 draw\_handler\_add() 및 RenderEngine.view\_draw()가 VR 세션의 렌더링 루프에서 호출되지 않는 문제를 확인하였으며, 이에 대한 해결책으로 \*\*Option C (OpenXR API Layer)\*\*와 **Option D (Blender 소스 코드 수정)** 두 가지 접근 방식을 제안하였다.

본 보고서는 Blender의 내부 VR 파이프라인(wm\_xr\_draw.c)과 OpenXR 사양(Specification)을 면밀히 분석하여 각 옵션의 기술적 타당성, 성능 오버헤드, 유지보수성, 배포 용이성을 평가한다. 특히, 72 FPS 이상의 고성능 VR 렌더링, 양안 스테레오스코픽(Stereoscopic) 지원, 그리고 Blender의 표준 씬(Scene) 지오메트리와의 올바른 심도 합성(Depth Composition)을 달성하기 위한 구체적인 아키텍처를 제시한다.

분석 결과, \*\*Option C (OpenXR API Layer)\*\*가 유지보수성, 배포 용이성, 그리고 시스템 안정성 측면에서 가장 우수한 전략으로 식별되었다. 이는 Blender의 코어 소스를 수정하지 않고도 렌더링 파이프라인을 후킹(Hooking)하여 고성능 커스텀 렌더링을 주입할 수 있는 유일한 표준화된 방법론이다.

## ---

**2\. 기술적 배경 및 문제 분석 (Technical Context)**

### **2.1 Blender VR 렌더링 파이프라인의 특수성**

Blender의 일반적인 뷰포트 렌더링과 VR 렌더링은 근본적으로 다른 실행 경로를 가진다. 이를 이해하는 것이 draw\_handler\_add가 실패하는 원인을 파악하는 핵심이다.

1. **Window Manager 기반 렌더링 (PC 모니터):** 표준 Blender 뷰포트는 운영체제의 윈도우 이벤트 시스템에 의해 구동된다. 사용자가 마우스를 움직이거나 애니메이션이 재생될 때 wm\_draw.c가 호출되며, 이때 Python의 draw\_handler 콜백들이 GPU\_matrix 상태를 상속받아 실행된다. 이 과정은 Python의 GIL(Global Interpreter Lock)을 보유한 상태에서 메인 스레드에서 순차적으로 발생한다.  
2. **OpenXR 기반 렌더링 (VR 헤드셋):** VR 렌더링은 GHOST\_Xr 컨텍스트에 의해 관리되며, OpenXR 런타임(Oculus)의 프레임 타이밍에 종속된다.  
   * 소스 파일: source/blender/windowmanager/xr/intern/wm\_xr\_draw.c  
   * 핵심 함수: wm\_xr\_draw\_views() 1  
   * 이 루프는 헤드셋의 포즈(Pose)를 대기하는 xrWaitFrame과 렌더링을 시작하는 xrBeginFrame, 그리고 제출하는 xrEndFrame 사이에서 엄격한 시간 제한(11ms/90Hz) 내에 실행되어야 한다.

### **2.2 Python 콜백이 VR에서 작동하지 않는 이유**

Blender 개발진은 VR 렌더링 루프의 성능 안정성을 위해 의도적으로 Python API의 개입을 배제하였다.3

* **성능 격리 (Performance Isolation):** Python 코드는 인터프리터 오버헤드와 가비지 컬렉션(GC)으로 인해 예측 불가능한 지연(Latency)을 유발할 수 있다. VR에서 단 몇 밀리초의 지연도 멀미(Motion Sickness)를 유발하는 '저더(Judder)' 현상으로 이어진다.  
* **컨텍스트 분리:** VR 렌더링은 오프스크린 버퍼(Offscreen Framebuffer)에 직접 그려지며, SpaceView3D와 같은 에디터 공간의 컨텍스트와는 분리된 RenderEngine 파이프라인을 따른다. bpy.types.SpaceView3D에 등록된 핸들러는 에디터 윈도우의 그리기 단계에 종속되어 있으므로, 별도의 스레드나 컨텍스트에서 도는 VR 렌더 루프에는 트리거되지 않는다.

## ---

**3\. 심층 분석: Option C \- OpenXR API Layer (C++ DLL/SO)**

이 접근 방식은 Blender 프로세스와 OpenXR 런타임(Oculus) 사이에 "미들웨어" 계층을 삽입하는 방식이다. Blender의 소스 코드를 건드리지 않고, 렌더링 제출 단계인 xrEndFrame을 가로채어(Intercept) 커스텀 그래픽스를 합성한다.

### **3.1 OpenXR API Layer의 개념 및 작동 원리**

OpenXR API Layer는 OpenXR 로더(Loader)에 의해 로드되는 동적 라이브러리(DLL)이다. 이는 데코레이터 패턴(Decorator Pattern)과 유사하게 작동하며, 애플리케이션(Blender)이 OpenXR 함수를 호출할 때, 이 호출을 먼저 받아 처리한 후 실제 런타임으로 전달하거나, 반환값을 조작할 수 있다.5

* **상호작용 흐름:**  
  1. Blender가 xrCreateInstance를 호출할 때, 로더는 레지스트리에 등록된 API Layer를 감지하고 로드한다.  
  2. Layer는 함수 테이블(Dispatch Table)을 훅킹하여 xrEndFrame과 같은 핵심 함수에 대한 포인터를 자신의 내부 함수로 바꿔치기한다.  
  3. Blender가 렌더링을 마친 후 xrEndFrame을 호출하면, API Layer의 xrEndFrame\_Hook 함수가 실행된다.  
  4. 이 시점에서 Layer는 Blender가 제출한 렌더링 결과물(Projection Layer)을 확인하고, 그 위에 3DGS 렌더링 결과물을 덧씌운 후 실제 런타임에 전달한다.

### **3.2 xrEndFrame 인터셉트를 통한 Composition Layer 주입 구현**

가장 핵심적인 구현 로직은 xrEndFrame 내부에서 발생한다. 3DGS를 렌더링하고 Blender의 결과물과 합성하기 위해 다음과 같은 절차를 수행한다.6

1. **포즈 데이터 추출:** XrFrameEndInfo 구조체 내의 layers 배열을 순회하여 Blender가 제출한 XrCompositionLayerProjection을 찾는다. 여기서 현재 프레임의 뷰포트 포즈(View Pose)와 FOV 정보를 추출한다. 이 정보는 3DGS 렌더러의 카메라 매트릭스를 설정하는 데 사용된다.  
2. **커스텀 렌더링 실행:** 추출한 포즈를 기반으로 API Layer 내부의 자체 렌더러(Vulkan 또는 OpenGL)를 사용하여 3D Gaussians를 오프스크린 텍스처(Swapchain)에 렌더링한다. 이때 알파 채널을 포함하여 투명도를 유지해야 한다.  
3. **레이어 재구성 (Layer Injection):**  
   * 새로운 std::vector\<const XrCompositionLayerBaseHeader\*\>를 생성한다.  
   * 첫 번째 요소로 Blender의 오리지널 레이어를 추가한다 (배경).  
   * 두 번째 요소로 3DGS가 렌더링된 새로운 XrCompositionLayerProjection을 추가한다 (전경).  
   * 이때 XR\_COMPOSITION\_LAYER\_BLEND\_TEXTURE\_SOURCE\_ALPHA\_BIT 플래그를 활성화하여 Blender 레이어 위에 알파 블렌딩되도록 설정한다.7  
4. **하위 호출:** 수정된 레이어 리스트를 담은 XrFrameEndInfo를 사용하여 체인의 다음 xrEndFrame을 호출한다.

### **3.3 Windows 환경에서의 구축 및 등록 방법**

Windows 11 환경에서 OpenXR API Layer를 개발하고 등록하는 구체적인 절차는 다음과 같다.

1. **개발 환경:** Visual Studio 2022, C++17 이상, OpenXR SDK, Vulkan SDK (또는 OpenGL 헤더).  
2. **프로젝트 설정:** DLL(Dynamic Link Library) 프로젝트로 설정한다.  
3. **엔트리 포인트:** xrNegotiateLoaderApiLayerInterface 함수를 \_\_declspec(dllexport)로 내보내야 한다. 이 함수는 로더와 레이어 간의 버전 협상 및 함수 테이블 교환을 담당한다.6  
4. **매니페스트(Manifest) 파일 작성:** JSON 형식의 메타데이터 파일이 필요하다.  
   JSON  
   {  
       "file\_format\_version": "1.0.0",  
       "api\_layer": {  
           "name": "XR\_APILAYER\_CUSTOM\_gaussian\_splat",  
           "library\_path": ".\\\\MyGaussianLayer.dll",  
           "api\_version": "1.0",  
           "implementation\_version": "1",  
           "description": "Renders 3DGS on top of Blender",  
           "disable\_environment": "DISABLE\_MY\_LAYER"  
       }  
   }

5. **레지스트리 등록:** Windows 레지스트리에 JSON 파일의 절대 경로를 등록한다.  
   * 경로: HKEY\_LOCAL\_MACHINE\\SOFTWARE\\Khronos\\OpenXR\\1\\ApiLayers\\Implicit  
   * 값 이름: JSON 파일의 전체 경로 (예: C:\\MyLayer\\layer.json)  
   * 값 데이터: 0 (DWORD) \- 활성화를 의미.9

### **3.4 성능 오버헤드와 한계점**

* **오버헤드:** API Layer 자체의 호출 오버헤드는 무시할 수 있는 수준(마이크로초 단위)이다. 그러나 3DGS 렌더링 자체가 GPU 자원을 Blender와 경합하게 된다. 4백만 개 이상의 Splat을 렌더링할 경우 GPU 점유율 상승으로 인해 Blender의 뷰포트 렌더링 해상도가 동적으로 낮아지거나 프레임 드랍이 발생할 수 있다.  
* **한계점 \- 상호작용의 단절:** API Layer는 독립된 프로세스 공간(Blender 프로세스 내에 로드되지만 논리적으로 분리됨)에서 실행되므로, Blender의 내부 데이터(예: 사용자가 3DGS 오브젝트를 이동시키기 위해 조작하는 Empty 오브젝트의 좌표)에 직접 접근할 수 없다. 이를 해결하기 위해 **IPC(Inter-Process Communication)**, 특히 공유 메모리(Shared Memory) 기법이 필수적이다.

### **3.5 구현 예시 및 레퍼런스**

* **OpenXR-API-Layer-Template (Ybalrid/mbucchia):** API Layer 개발을 위한 가장 표준적인 C++ 보일러플레이트이다. 함수 후킹 및 매니페스트 생성 빌드 스크립트를 포함한다.6  
* **OpenXR Toolkit:** 기존 렌더링 파이프라인에 업스케일링(FSR/DLSS)이나 포비에이티드 렌더링을 주입하는 대표적인 오픈소스 프로젝트로, xrEndFrame 후킹의 실제 구현 사례를 참고하기에 최적이다.10

### **3.6 유지보수 용이성**

이 방식의 가장 큰 장점은 **Blender 버전 독립성**이다. Blender가 업데이트되어 내부 소스 코드가 변경되더라도, OpenXR 표준을 준수하는 한 API Layer는 계속 작동한다. 사용자는 Blender를 재설치하거나 패치할 필요 없이 DLL과 JSON 파일만 설치하면 된다.

## ---

**4\. 심층 분석: Option D \- Blender 소스 코드 수정**

이 방식은 Blender의 소스 코드(C/C++)를 직접 수정하여 VR 렌더링 루프 내에 3DGS 렌더링 함수를 직접 호출하는 방식이다.

### **4.1 wm\_xr\_draw.c 구조와 VR 렌더링 파이프라인 분석**

source/blender/windowmanager/xr/intern/wm\_xr\_draw.c 파일은 Blender의 VR 세션 드로잉을 담당한다. 핵심 함수인 wm\_xr\_draw\_view는 다음과 같은 역할을 수행한다 12:

1. OpenXR로부터 받은 뷰 행렬(View Matrix)과 투영 행렬(Projection Matrix)을 Blender의 뷰포트 데이터(RegionView3D)에 적용한다.  
2. ED\_view3d\_draw\_offscreen\_simple 함수를 호출하여 Blender의 3D 씬을 오프스크린 버퍼에 그린다.  
3. 렌더링된 텍스처를 OpenXR 스왑체인(Swapchain)에 복사하거나 제출한다.

### **4.2 Python 콜백 추가 가능성**

이론적으로는 wm\_xr\_draw\_view 함수 내부에서 Python C API (PyObject\_Call)를 사용하여 특정 Python 함수를 호출하도록 코드를 수정할 수 있다.

* **기술적 난관:** VR 렌더링 스레드는 메인 스레드와 다를 수 있으며, Python의 GIL 상태를 관리해야 한다. 렌더링 루프 도중 Python 코드를 실행하면 GIL 락(Lock) 대기 시간이나 GC(Garbage Collection)로 인해 프레임 타임이 급격히 튀는 스파이크(Spike)가 발생할 수 있다. 이는 VR 경험을 심각하게 저해한다. 따라서, **Python 콜백을 직접 VR 루프에 넣는 것은 강력히 비추천**된다.

### **4.3 빌드 및 배포 복잡성**

* **빌드 환경:** Blender 전체 소스 코드를 빌드하려면 Visual Studio 2022, CUDA Toolkit, OptiX, 그리고 수 기가바이트에 달하는 사전 컴파일된 라이브러리(SVN)를 설정해야 한다.14 이는 일반 사용자나 가벼운 개발자에게 매우 높은 진입 장벽이다.  
* **배포:** 수정 사항을 적용하려면 전체 blender.exe를 다시 빌드하여 배포해야 한다. 이는 사실상 Blender의 커스텀 포크(Fork) 버전을 유지보수해야 함을 의미한다.

### **4.4 버전 업데이트 시 유지보수 부담**

Blender는 개발 속도가 매우 빠르며, 내부 API(DNA, RNA, GHOST)가 자주 변경된다. 5.0 버전에 맞춰 수정한 코드는 5.1이나 5.2에서 컴파일 에러를 일으킬 확률이 매우 높다. 매 업데이트마다 git merge 충돌을 해결하고 다시 빌드해야 하는 막대한 기술 부채(Technical Debt)가 발생한다.

## ---

**5\. 비교 분석 (Comparative Analysis)**

다음은 두 가지 옵션을 핵심 지표별로 비교한 결과이다.

| 비교 항목 | Option C: OpenXR API Layer | Option D: Blender 소스 수정 |
| :---- | :---- | :---- |
| **구현 난이도** | **매우 높음** (OpenXR, C++, 그래픽스 API 직접 제어 필요) | **높음** (Blender 거대 코드베이스 이해 및 빌드 환경 구축 필요) |
| **성능 (72+ FPS)** | **최상** (C++ 네이티브 실행, Blender 루프와 독립적 최적화 가능) | **상** (Blender 내부 오버헤드에 종속될 수 있음) |
| **유지보수성** | **우수** (Blender 업데이트에 영향받지 않음, OpenXR 표준 준수) | **나쁨** (매 버전마다 재빌드 및 코드 병합 필요) |
| **배포 용이성** | **우수** (설치 프로그램 또는 스크립트로 DLL/JSON 배포 가능) | **매우 나쁨** (수 기가바이트의 커스텀 Blender 바이너리 배포 필요) |
| **데이터 접근성** | **낮음** (IPC/Shared Memory 등을 통해 Blender 데이터 수신 필요) | **최상** (Blender 내부 메모리 구조체 DNA 직접 접근 가능) |
| **리스크** | 그래픽스 API 컨텍스트 공유(OpenGL/DX) 처리 복잡성 | Python GIL 문제로 인한 VR 멀미 유발 가능성 |

## ---

**6\. 최종 목표 달성을 위한 추천 및 상세 구현 가이드**

### **6.1 추천: Option C (OpenXR API Layer) \+ Shared Memory Hybrid**

**결론적으로 Option C가 최종 목표(고성능, 유지보수, 배포)를 달성하기 위한 가장 적합한 아키텍처이다.** Option D는 내부 데이터 접근은 쉽지만, 사용자에게 커스텀 Blender 설치를 강요한다는 점에서 확장성이 떨어진다.

본 보고서는 **C++ OpenXR API Layer로 렌더링 엔진을 구축하고, Blender Python Add-on이 공유 메모리(Shared Memory)를 통해 제어 데이터를 전송하는 하이브리드 아키텍처**를 제안한다.

### **6.2 상세 구현 아키텍처**

#### **단계 1: OpenXR API Layer 개발 (C++)**

1. **프로젝트 셋업:** OpenXR-API-Layer-Template을 기반으로 Visual Studio 프로젝트를 생성한다.  
2. **xrEndFrame 훅 구현:**  
   * Blender 레이어(XrCompositionLayerProjection)에서 View Pose와 FOV를 추출한다.  
   * 추출된 카메라 정보를 바탕으로 3DGS 렌더러(예: 3DGS.cpp의 Vulkan 백엔드 또는 CUDA Rasterizer)를 실행한다.  
   * 3DGS는 투명한 배경을 가진 별도의 텍스처(Swapchain)에 렌더링된다.  
3. **심도 합성 (Depth Composition) \- 핵심 기술:**  
   * 단순히 3DGS를 위에 그리면 Blender 오브젝트(예: 큐브) 뒤에 있어야 할 Gaussian이 앞에 그려지는 문제가 발생한다.15  
   * 해결책: **XR\_KHR\_composition\_layer\_depth** 익스텐션을 활용한다.7  
   * API Layer는 Blender가 제출한 깊이 버퍼(Depth Buffer) 핸들을 가져온다.  
   * 3DGS 픽셀 셰이더에서 Blender의 깊이 값을 샘플링하여, if (gaussian\_depth \> blender\_depth) discard; 로직을 수행함으로써 완벽한 오클루전(Occlusion)을 구현한다.

#### **단계 2: Blender Python Add-on 개발 (Controller)**

1. **데이터 수집:** bpy.app.handlers.depsgraph\_update\_post 핸들러를 사용하여 매 프레임마다 3DGS 모델의 위치(World Matrix), 로드할 파일 경로, 가시성 상태 등을 수집한다.3  
2. **IPC 통신:** Python의 mmap 모듈을 사용하여 Windows 명명된 공유 메모리(Named Shared Memory)에 데이터를 쓴다. 구조체 포맷(예: struct { float matrix; char path; })을 정의하여 C++ Layer와 맞춘다.

#### **단계 3: 데이터 동기화 및 렌더링**

1. API Layer는 매 프레임(xrWaitFrame 시점 등) 공유 메모리를 읽어 3DGS 모델의 위치를 업데이트한다.  
2. 사용자가 Blender에서 Empty 오브젝트를 움직이면, API Layer의 Gaussian 모델이 실시간으로 따라 움직이게 된다.

### **6.3 필요한 도구 및 라이브러리**

* **IDE:** Microsoft Visual Studio 2022 (C++ Desktop Development 워크로드).  
* **SDK:**  
  * OpenXR SDK (NuGet 또는 GitHub 소스).  
  * Vulkan SDK (권장) 또는 CUDA Toolkit (NVIDIA 전용 고성능 필요 시).  
  * Python 3.10+ (Blender 내장).  
* **라이브러리:**  
  * **3DGS 렌더러:** gsplat (CUDA 기반, 고성능) 또는 3DGS.cpp (Vulkan 기반, 호환성 우수).19 OpenXR과의 통합을 위해서는 Vulkan 기반인 3DGS.cpp를 포팅하여 사용하는 것이 컨텍스트 공유(Interop) 측면에서 유리할 수 있다.  
  * **glm:** 수학 연산 라이브러리.

### **6.4 예상 작업량 및 위험 요소 (Risk Assessment)**

**예상 작업량:**

* **환경 구축 및 Hello World Layer:** 3-5일 (OpenXR 훅킹 검증).  
* **3DGS 렌더러 통합:** 2-3주 (기존 오픈소스 렌더러를 API Layer 내부로 포팅 및 최적화).  
* **Depth Buffer 공유 및 오클루전 구현:** 1-2주 (Graphics API 간 텍스처 공유 문제 해결).  
* **Python 브릿지 및 UI:** 3-5일.  
* **총계:** 숙련된 그래픽스 엔지니어 기준 약 **1\~1.5개월**.

**위험 요소:**

1. **Graphics API 불일치:** Oculus Link는 Windows에서 주로 DirectX 11/12를 사용한다. 3DGS 렌더러가 CUDA/Vulkan 기반일 경우, **DirectX-Vulkan Interop** 또는 **DirectX-CUDA Interop** 구현이 필수적이며 이는 난이도가 매우 높다.2  
   * *완화 전략:* XR\_KHR\_vulkan\_enable2 등을 사용하여 처음부터 Vulkan 세션을 강제하거나, Blender가 사용하는 그래픽스 바인딩을 감지하여 그에 맞는 인터롭 텍스처를 생성해야 한다.  
2. **지연 시간(Latency):** 4백만 개 이상의 Gaussian을 정렬(Sorting)하는 작업은 CPU/GPU 부하가 크다. 90Hz 방어를 위해 **Radix Sort**와 같은 고속 정렬 알고리즘이 필수적이다.22  
3. **Blender의 Depth 제출 여부:** Blender 버전이나 설정에 따라 Depth Layer 제출이 비활성화되어 있을 수 있다. 이 경우 오클루전이 불가능하므로, Blender 설정에서 이를 강제하거나 확인하는 로직이 필요하다.23

## ---

**7\. 결론**

Blender 5.0 VR 환경에서 3D Gaussian Splatting을 72 FPS 이상으로 안정적으로 렌더링하고 사용자 배포를 용이하게 하기 위해서는 \*\*Option C (OpenXR API Layer)\*\*가 가장 합리적인 선택이다. 특히 **공유 메모리를 통한 Python 제어**와 **XR\_KHR\_composition\_layer\_depth를 활용한 깊이 합성**이 기술적 성공의 핵심 열쇠(Key Success Factor)이다.

개발자는 우선 OpenXR-API-Layer-Template을 사용하여 Blender VR 화면 위에 간단한 삼각형을 띄우는 것부터 시작하여, 단계적으로 3DGS 렌더러를 통합해 나가는 애자일(Agile) 접근 방식을 권장한다.

#### **참고 자료**

1. Code Layout \- Blender Developer Documentation, 12월 8, 2025에 액세스, [https://developer.blender.org/docs/features/code\_layout/](https://developer.blender.org/docs/features/code_layout/)  
2. GSoC 2019: VR support through OpenXR \- Weekly Reports \- Blender Devtalk, 12월 8, 2025에 액세스, [https://devtalk.blender.org/t/gsoc-2019-vr-support-through-openxr-weekly-reports/7665](https://devtalk.blender.org/t/gsoc-2019-vr-support-through-openxr-weekly-reports/7665)  
3. VR camera in Blender realtime \[working /WIP\] \- Python Support, 12월 8, 2025에 액세스, [https://blenderartists.org/t/vr-camera-in-blender-realtime-working-wip/1131920](https://blenderartists.org/t/vr-camera-in-blender-realtime-working-wip/1131920)  
4. Virtual Reality \- OpenXR \- Blender Developer, 12월 8, 2025에 액세스, [https://developer.blender.org/docs/features/gpu/viewports/xr/](https://developer.blender.org/docs/features/gpu/viewports/xr/)  
5. 1 Introduction — OpenXR Tutorial documentation, 12월 8, 2025에 액세스, [https://openxr-tutorial.com/linux/opengl/1-introduction.html](https://openxr-tutorial.com/linux/opengl/1-introduction.html)  
6. Ybalrid/OpenXR-API-Layer-Template \- GitHub, 12월 8, 2025에 액세스, [https://github.com/Ybalrid/OpenXR-API-Layer-Template](https://github.com/Ybalrid/OpenXR-API-Layer-Template)  
7. OpenXR app best practices \- Mixed Reality \- Microsoft Learn, 12월 8, 2025에 액세스, [https://learn.microsoft.com/en-us/windows/mixed-reality/develop/native/openxr-best-practices](https://learn.microsoft.com/en-us/windows/mixed-reality/develop/native/openxr-best-practices)  
8. 3 Graphics — OpenXR Tutorial documentation, 12월 8, 2025에 액세스, [https://openxr-tutorial.com/linux/opengl/3-graphics.html](https://openxr-tutorial.com/linux/opengl/3-graphics.html)  
9. Best Practices for OpenXR API Layers on Windows | Fred Emmott, 12월 8, 2025에 액세스, [https://fredemmott.com/blog/2024/11/25/best-practices-for-openxr-api-layers.html](https://fredemmott.com/blog/2024/11/25/best-practices-for-openxr-api-layers.html)  
10. Quickstart | OpenXR Toolkit, 12월 8, 2025에 액세스, [https://mbucchia.github.io/OpenXR-Toolkit/](https://mbucchia.github.io/OpenXR-Toolkit/)  
11. mbucchia/OpenXR-Toolkit: A collection of useful features to ... \- GitHub, 12월 8, 2025에 액세스, [https://github.com/mbucchia/OpenXR-Toolkit](https://github.com/mbucchia/OpenXR-Toolkit)  
12. Fix: Viewport: Allow overriding solid mode background type \#149656 \- Blender Projects, 12월 8, 2025에 액세스, [https://projects.blender.org/blender/blender/pulls/149656](https://projects.blender.org/blender/blender/pulls/149656)  
13. blender/source/blender/windowmanager/intern/wm\_files.c at master \- GitHub, 12월 8, 2025에 액세스, [https://github.com/dfelinto/blender/blob/master/source/blender/windowmanager/intern/wm\_files.c](https://github.com/dfelinto/blender/blob/master/source/blender/windowmanager/intern/wm_files.c)  
14. Building Blender \- Blender Developer Documentation, 12월 8, 2025에 액세스, [https://developer.blender.org/docs/handbook/building\_blender/](https://developer.blender.org/docs/handbook/building_blender/)  
15. New OpenXR Validation Layer Helps Developers Build Robustly Portable XR Applications, 12월 8, 2025에 액세스, [https://www.khronos.org/blog/new-openxr-validation-layer-helps-developers-build-robustly-portable-xr-applications](https://www.khronos.org/blog/new-openxr-validation-layer-helps-developers-build-robustly-portable-xr-applications)  
16. 5 Extensions — OpenXR Tutorial documentation, 12월 8, 2025에 액세스, [https://openxr-tutorial.com/linux/opengl/5-extensions.html](https://openxr-tutorial.com/linux/opengl/5-extensions.html)  
17. S4-231860\_r1.docx \- 3GPP, 12월 8, 2025에 액세스, [https://www.3gpp.org/ftp/tsg\_sa/wg4\_codec/TSGS4\_126\_Chicago/Inbox/Drafts/Video/S4-231860\_r1.docx](https://www.3gpp.org/ftp/tsg_sa/wg4_codec/TSGS4_126_Chicago/Inbox/Drafts/Video/S4-231860_r1.docx)  
18. Gotchas — Blender Python API, 12월 8, 2025에 액세스, [https://docs.blender.org/api/4.0/info\_gotcha.html](https://docs.blender.org/api/4.0/info_gotcha.html)  
19. shg8/3DGS.cpp: A cross-platform, high performance renderer for Gaussian Splatting using Vulkan Compute. Supports Windows, Linux, macOS, iOS, and visionOS \- GitHub, 12월 8, 2025에 액세스, [https://github.com/shg8/3DGS.cpp](https://github.com/shg8/3DGS.cpp)  
20. gsplat: An Open-Source Library for Gaussian Splatting \- arXiv, 12월 8, 2025에 액세스, [https://arxiv.org/html/2409.06765v1](https://arxiv.org/html/2409.06765v1)  
21. \[TRACKER\] OpenXR in Godot 4 · Issue \#69647 \- GitHub, 12월 8, 2025에 액세스, [https://github.com/godotengine/godot/issues/69647](https://github.com/godotengine/godot/issues/69647)  
22. VRSplat: Fast and Robust Gaussian Splatting for Virtual Reality \- arXiv, 12월 8, 2025에 액세스, [https://arxiv.org/html/2505.10144v1](https://arxiv.org/html/2505.10144v1)  
23. DEVICE\_LOST when trying to use XR\_KHR\_composition\_layer\_depth | Meta Community Forums \- 1227351, 12월 8, 2025에 액세스, [https://communityforums.atmeta.com/discussions/dev-openxr/device-lost-when-trying-to-use-xr-khr-composition-layer-depth/1227351](https://communityforums.atmeta.com/discussions/dev-openxr/device-lost-when-trying-to-use-xr-khr-composition-layer-depth/1227351)