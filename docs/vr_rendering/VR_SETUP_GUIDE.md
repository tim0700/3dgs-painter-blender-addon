# VR 페인팅 설정 가이드

> Quest 3 VR 헤드셋으로 3DGS Painter를 사용하기 위한 설정 가이드입니다.

---

## 목차

1. [시스템 요구사항](#시스템-요구사항)
2. [설치 전 준비](#설치-전-준비)
3. [애드온 설치](#애드온-설치)
4. [OpenXR 레이어 설정](#openxr-레이어-설정)
5. [VR 페인팅 사용법](#vr-페인팅-사용법)
6. [문제 해결](#문제-해결)

---

## 시스템 요구사항

### 하드웨어

- **VR 헤드셋**: Meta Quest 3 (Quest Link 사용)
- **GPU**: NVIDIA GTX 1070 이상 (VR Ready)
- **RAM**: 16GB 이상 권장
- **USB**: USB 3.0 케이블 (Quest Link용)

### 소프트웨어

- **OS**: Windows 10/11
- **Blender**: 5.0 이상
- **Meta Quest PC App**: 최신 버전
- **OpenXR Runtime**: Oculus (기본 설정)

---

## 설치 전 준비

### 1. Quest Link 연결 확인

```
1. Meta Quest PC App 설치 및 로그인
2. Quest 3을 USB 케이블로 PC에 연결
3. 헤드셋에서 "Link 허용" 선택
4. PC App에서 연결 상태 확인 (녹색 점)
```

### 2. OpenXR 런타임 설정

```powershell
# Windows 설정 > 혼합 현실 > OpenXR
# "Oculus"가 Active Runtime인지 확인

# 또는 레지스트리 확인
reg query "HKLM\SOFTWARE\Khronos\OpenXR\1" /v "ActiveRuntime"
```

**중요**: Oculus OpenXR Runtime이 활성화되어 있어야 합니다.

---

## 애드온 설치

### 1. 애드온 다운로드

```
GitHub: https://github.com/tim0700/3dgs-painter-blender-addon
```

### 2. Blender에 설치

```
1. Blender 실행
2. Edit > Preferences > Add-ons
3. "Install..." 클릭
4. 다운로드한 ZIP 파일 선택
5. "3DGS Painter" 체크박스 활성화
```

### 3. 설치 확인

```
- Sidebar (N키) > 3DGS Painter 탭 확인
- 콘솔에서 확인:
  [3DGS Painter] Addon registered (Subprocess Actor mode)
  [3DGS Painter VR] VR module registered
```

---

## OpenXR 레이어 설정

### 1. 레이어 DLL 위치 확인

애드온 설치 후 다음 경로에 파일이 있어야 합니다:

```
%APPDATA%\Blender Foundation\Blender\5.0\scripts\addons\
    threegds_painter\
        openxr_layer\
            build\Release\gaussian_layer.dll
            manifest\XR_APILAYER_3DGS.json
```

### 2. 레이어 등록 (자동)

애드온이 처음 실행될 때 자동으로 등록을 시도합니다.
만약 수동 등록이 필요한 경우:

```powershell
# PowerShell (관리자 권한)
$jsonPath = "$env:APPDATA\Blender Foundation\Blender\5.0\scripts\addons\threegds_painter\openxr_layer\manifest\XR_APILAYER_3DGS.json"

reg add "HKLM\SOFTWARE\Khronos\OpenXR\1\ApiLayers\Implicit" `
    /v $jsonPath /t REG_DWORD /d 0 /f
```

### 3. 레이어 등록 확인

```powershell
# 등록된 레이어 목록 확인
reg query "HKLM\SOFTWARE\Khronos\OpenXR\1\ApiLayers\Implicit"

# 출력에 XR_APILAYER_3DGS.json이 포함되어야 함
```

---

## VR 페인팅 사용법

### 1. VR 세션 시작

```
1. Blender에서 3DGS Painter 패널 열기 (N키 > 3DGS Painter)
2. "VR Painting" 섹션에서 "Start VR Session" 클릭
3. 헤드셋을 착용하고 VR 환경 확인
```

### 2. 컨트롤러 조작

| 버튼                | 기능                        |
| ------------------- | --------------------------- |
| **오른쪽 Trigger**  | 페인팅 (누르는 동안 그리기) |
| **왼쪽 Joystick**   | 이동 (텔레포트)             |
| **오른쪽 Joystick** | 회전                        |

### 3. 페인팅 팁

- **브러시 크기**: Blender UI에서 조절 후 VR 시작
- **색상**: Blender UI에서 설정
- **스트로크**: Trigger를 누른 상태로 이동

### 4. VR 종료

```
- Blender UI에서 "Stop VR Session" 클릭
- 또는 헤드셋에서 Oculus 버튼 → 종료
```

---

## 문제 해결

### VR이 시작되지 않음

```
오류: "No XR runtime found"
해결:
1. Meta Quest PC App 실행 확인
2. Quest Link 연결 상태 확인
3. OpenXR 런타임 설정 확인 (Oculus)
```

### Gaussian이 VR에서 보이지 않음

```
문제: 페인팅은 되지만 헤드셋에서 표시 안됨
해결:
1. OpenXR 레이어 등록 확인
2. 콘솔에서 "[GaussianRender]" 로그 확인
3. 레이어 DLL 경로 확인
```

### 위치가 이상함

```
문제: Gaussian이 이상한 위치에 렌더링됨
해결:
1. Blender 카메라 위치 확인
2. 뷰 매트릭스 업데이트 확인
3. 콘솔에서 "[VR Matrix]" 로그 확인
```

### 텔레포트와 충돌

```
문제: Trigger 누르면 텔레포트도 같이 됨
현상: 약 70% 확률로 텔레포트 비활성화 성공
이유: Blender 5.0 OpenXR 제한 (세션 중 ActionMap 수정 불안정)
```

### 크래시 발생

```
오류: EXCEPTION_ACCESS_VIOLATION in VCRUNTIME140.dll
해결:
- 최신 버전으로 업데이트
- VR 시작/종료 사이에 충분한 시간 대기
```

---

## 로그 파일 위치

디버깅에 유용한 로그 파일들:

| 로그          | 경로                               |
| ------------- | ---------------------------------- |
| Blender 콘솔  | Window > Toggle System Console     |
| OpenXR 레이어 | `%LOCALAPPDATA%\XR_APILAYER_3DGS\` |

---

## 다음 문서

- [VR 모듈 아키텍처](./VR_MODULE_ARCHITECTURE.md)
- [OpenXR 레이어 아키텍처](./OPENXR_LAYER_ARCHITECTURE.md)
