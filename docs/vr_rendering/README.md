# VR Rendering 개발 문서

> **최종 업데이트**: 2025-12-09  
> **목표**: Blender에서 3D Gaussian Splatting을 VR 헤드셋에 렌더링

---

## 📋 현재 상태

| 항목               | 상태                   |
| ------------------ | ---------------------- |
| PC GLSL Viewport   | ✅ 작동                |
| VR 컨트롤러 추적   | ✅ 작동                |
| VR 페인팅          | ✅ 작동 (TRIGGER)      |
| VR Gaussian 렌더링 | ✅ 작동 (OpenXR Layer) |
| 텔레포트 비활성화  | ⚠️ 70% 안정            |

---

## 📚 핵심 문서

| 문서                                                     | 설명                              |
| -------------------------------------------------------- | --------------------------------- |
| [VR 모듈 아키텍처](./VR_MODULE_ARCHITECTURE.md)          | Python 코드 (`src/vr/`) 전체 분석 |
| [OpenXR 레이어 아키텍처](./OPENXR_LAYER_ARCHITECTURE.md) | C++ DLL (`openxr_layer/`) 상세    |
| [VR 설정 가이드](./VR_SETUP_GUIDE.md)                    | Quest 3 설정 및 사용법            |

---

## 📁 폴더 구조

```
docs/vr_rendering/
├── README.md                     ← 현재 파일
├── VR_MODULE_ARCHITECTURE.md     ← Python 아키텍처
├── OPENXR_LAYER_ARCHITECTURE.md  ← C++ 아키텍처
├── VR_SETUP_GUIDE.md             ← 설정 가이드
│
├── research/                     ← 개발 히스토리 (연구 문서)
│   ├── 3D Gaussian.md
│   ├── Blender VR Custom Shader Rendering.md
│   ├── Blender VR Gaussian Splatting Rendering.md
│   └── ...
│
└── debug/                        ← 디버그 로그
    ├── vrBugReport/
    └── VRControlProblem/
```

---

## 🔑 핵심 발견

1. **draw_handler가 VR에서 안 되는 이유**: Blender `wm_xr_draw.c`에서 overlay 스킵
2. **OpenXR API Layer**: 최종 솔루션 - `xrEndFrame()` 후킹
3. **ActionMap 제한**: 세션 중 수정 시 race condition

---

## 📚 참고 자료

- [OpenXR-API-Layer-Template](https://github.com/Ybalrid/OpenXR-API-Layer-Template)
- [Blender VR Source](https://fossies.org/dox/blender-4.5.1/wm__xr__draw_8cc_source.html)
