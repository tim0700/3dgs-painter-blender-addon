# Phase 2 Implementation Summary

**Date**: 2025-12-03  
**Status**: ✅ COMPLETE

## 📋 Overview

Phase 2 구현은 **Subprocess Actor 패턴**을 통해 Windows TBB DLL 충돌 문제를 해결하고, PyTorch/CUDA를 Blender 내에서 안전하게 사용할 수 있는 인프라를 구축했습니다.

### 핵심 문제 및 해결

| 문제                           | 원인                                                        | 해결                                                                          |
| ------------------------------ | ----------------------------------------------------------- | ----------------------------------------------------------------------------- |
| WinError 1114                  | Blender의 `tbb12.dll`과 PyTorch `c10.dll` 충돌              | Subprocess Actor 패턴으로 프로세스 격리                                       |
| Queue unpickle 시 torch import | pickle이 torch 모듈을 역직렬화할 때 main process에서 import | `_sanitize_for_pickle()` 함수로 순수 Python 타입으로 변환                     |
| PyTorch CPU 버전 덮어쓰기      | pip가 PyPI에서 최신 CPU 버전으로 업그레이드                 | 정확한 버전 지정 (`torch==2.6.0+cu124`) + `--upgrade-strategy only-if-needed` |

---

## 📁 구현된 파일 구조

```
src/
├── __init__.py                    # ✅ 수정: Subprocess 감지, 의존성 경로 설정
├── operators.py                   # ✅ 수정: 테스트/설치 오퍼레이터 추가
├── preferences.py                 # ✅ 수정: 설치 UI 패널
├── requirements/                  # ✅ 신규: 플랫폼별 의존성 파일
│   ├── win_cuda.txt
│   ├── win_cpu.txt
│   ├── mac_mps.txt
│   └── linux_cuda.txt
├── generator_process/             # ✅ 신규: Subprocess Actor 인프라
│   ├── __init__.py               # NPRGenerator, RunInSubprocess 데코레이터
│   ├── actor.py                  # Actor 베이스 클래스, _sanitize_for_pickle
│   └── future.py                 # Future 클래스 (비동기 결과)
└── npr_core/
    ├── dependencies.py            # ✅ 신규: 의존성 체크 함수
    └── installer.py               # ✅ 신규: PackageInstaller 클래스
```

---

## 🎯 Completed Tasks

### 1. ✅ Subprocess Actor 패턴 구현

**`actor.py`** - Dream Textures 참조 구현

```python
# 핵심 메커니즘
is_actor_process = current_process().name == "__actor__"

# Frontend (Blender main process)
def _send(self, action, *args, **kwargs) -> Future:
    """Queue를 통해 subprocess로 메시지 전송"""

# Backend (PyTorch subprocess)
def _receive(self):
    """Queue에서 메시지 수신 및 처리"""
```

**`_sanitize_for_pickle()`** - TBB 충돌 방지의 핵심

```python
def _sanitize_for_pickle(obj):
    """
    torch/numpy 객체를 순수 Python 타입으로 변환.
    Queue unpickle 시 torch import를 방지하여 TBB DLL 충돌 회피.

    변환 규칙:
    - torch.Tensor → list (via .tolist())
    - numpy.ndarray → list (via .tolist())
    - numpy.float32 → float
    - dict/list/tuple → 재귀 변환
    """
```

### 2. ✅ Future 패턴 구현

**`future.py`** - 비동기 결과 처리

```python
class Future:
    def result(self, timeout=None):
        """블로킹 대기로 결과 반환"""

    def add_done_callback(self, callback):
        """완료 시 콜백 호출"""

    def check(self) -> bool:
        """논블로킹 완료 체크"""
```

### 3. ✅ NPRGenerator Actor

**`generator_process/__init__.py`**

| 메서드                        | 설명                   | 반환 타입 |
| ----------------------------- | ---------------------- | --------- |
| `get_torch_info()`            | PyTorch/CUDA 버전 정보 | `dict`    |
| `check_dependencies()`        | 패키지 설치 상태       | `dict`    |
| `test_cuda_computation(size)` | CUDA 연산 테스트       | `dict`    |

### 4. ✅ 의존성 설치 시스템

**`installer.py`**

```python
class PackageInstaller:
    def install_all(self, cuda_version=None, progress_callback=None):
        """
        설치 순서:
        1. PyTorch + torchvision (정확한 CUDA 버전)
        2. Base requirements (--upgrade-strategy only-if-needed)
        3. gsplat (optional)
        """
```

**핵심 수정 사항**:

-   PyTorch 먼저 설치 (`--force-reinstall --no-deps`)
-   정확한 버전 지정: `torch==2.6.0+cu124`
-   `--upgrade-strategy only-if-needed`: 이미 설치된 torch 보호

### 5. ✅ 테스트 오퍼레이터

**`operators.py`**

| bl_idname                         | 기능                             |
| --------------------------------- | -------------------------------- |
| `threegds.test_subprocess`        | Subprocess에서 PyTorch 정보 확인 |
| `threegds.test_subprocess_cuda`   | CUDA 행렬 연산 테스트            |
| `threegds.kill_subprocess`        | Subprocess 종료                  |
| `threegds.install_dependencies`   | 의존성 설치                      |
| `threegds.uninstall_dependencies` | 의존성 제거                      |

---

## 🧪 설치 후 테스트 방법

### 방법 1: Blender UI에서 테스트 (권장)

1. **Blender 열기** → `Edit` → `Preferences` → `Add-ons`
2. **3DGS Painter** 검색 → 확장하여 Preferences 패널 열기
3. **"Install Dependencies"** 버튼 클릭 (5-15분 소요)
4. **Blender 재시작**
5. Preferences 패널에서 테스트 버튼 사용:
    - **"Test Subprocess PyTorch"**: PyTorch 버전 및 CUDA 정보 확인
    - **"Test Subprocess CUDA"**: GPU 연산 테스트

### 방법 2: Python Console에서 오퍼레이터 호출

Blender에서 **Python Console** 열기 (`Scripting` 워크스페이스):

```python
# 1. Subprocess에서 PyTorch 정보 확인
import bpy
bpy.ops.threegds.test_subprocess()
# Info: PyTorch 2.6.0+cu124, CUDA: True, Device: NVIDIA GeForce RTX ...

# 2. CUDA 연산 테스트
bpy.ops.threegds.test_subprocess_cuda()
# Info: CUDA Test: cuda, 1000x1000, compute: 5.23ms, transfer: 0.45ms

# 3. Subprocess 종료
bpy.ops.threegds.kill_subprocess()
```

### 예상 출력 (정상 설치)

```
PyTorch Info:
  torch_version: 2.6.0+cu124
  cuda_available: True
  cuda_version: 12.4
  device_count: 1
  device_name: NVIDIA GeForce RTX 2070 SUPER
  devices:
    - index: 0
      name: NVIDIA GeForce RTX 2070 SUPER
      total_memory_gb: 8.0
      compute_capability: 7.5

CUDA Test:
  success: True
  device: cuda
  size: 1000
  compute_time_ms: 5.23
  transfer_time_ms: 0.45
```

---

## 🔧 문제 해결

### "Missing Dependencies" 표시됨

```
원인: Blender 재시작 필요
해결: 의존성 설치 후 반드시 Blender 재시작
```

### Test Subprocess에서 CPU만 나옴

```
원인: PyTorch CPU 버전이 설치됨
해결:
1. Preferences에서 "Uninstall Dependencies" 클릭
2. CUDA 버전을 명시적으로 선택 (Auto-detect 대신)
3. "Install Dependencies" 다시 클릭
4. Blender 재시작
```

### WinError 1114 발생

```
원인: Main process에서 torch import 시도
해결:
- 이 에러가 Subprocess 테스트 중 발생하면 정상 (subprocess로 우회)
- Main process에서 직접 import torch 하지 말 것
```

---

## 📊 성능 지표

| 항목                        | 결과                   |
| --------------------------- | ---------------------- |
| Subprocess 시작 시간        | ~2초 (첫 호출)         |
| get_torch_info()            | ~50ms                  |
| test_cuda_computation(1000) | ~5-15ms                |
| Queue IPC 오버헤드          | ~10-20ms (작은 데이터) |

---

## 🎯 Success Criteria

-   [x] Windows TBB DLL 충돌 우회 (Subprocess 격리)
-   [x] PyTorch CUDA 버전 정상 설치 (2.6.0+cu124)
-   [x] Subprocess에서 CUDA 연산 성공
-   [x] Queue 통신 시 torch import 방지 (`_sanitize_for_pickle`)
-   [x] 설치/제거 UI 작동
-   [x] 테스트 오퍼레이터 작동

---

## 🚀 Next Steps (Phase 3 & 4)

### Phase 3: Viewport Rendering

-   GLSL 기반 Gaussian Splatting 뷰포트 렌더러
-   draw_handler 통합
-   실시간 프리뷰

### Phase 4: Painting Interaction

-   Modal 페인팅 오퍼레이터
-   SharedMemory IPC (Queue 대체, ~80ms → <1ms)
-   gsplat Deformation 통합

---

## 📝 기술 참고사항

### TBB DLL 충돌 상세

```
경로: blender.shared/tbb12.dll (Blender 번들)
충돌: PyTorch c10.dll이 다른 버전의 TBB 요구
에러: OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed
해결: multiprocessing spawn context로 별도 프로세스에서 PyTorch 로드
```

### Subprocess Detection

```python
# src/__init__.py
from multiprocessing import current_process
is_actor = current_process().name == "__actor__"

if is_actor:
    # Subprocess: 의존성 로드, PyTorch 사용 가능
    _load_dependencies()
else:
    # Main process: PyTorch import 금지
    # Generator를 통해서만 PyTorch 기능 접근
```

### 의존성 경로

```
Windows: %APPDATA%\Blender Foundation\Blender\5.0\scripts\addons\threegds_painter\.python_dependencies
macOS: ~/Library/Application Support/Blender/5.0/scripts/addons/threegds_painter/.python_dependencies
Linux: ~/.config/blender/5.0/scripts/addons/threegds_painter/.python_dependencies
```

---

**Phase 2: ✅ COMPLETE**  
**Ready for Phase 3: Viewport Rendering**
