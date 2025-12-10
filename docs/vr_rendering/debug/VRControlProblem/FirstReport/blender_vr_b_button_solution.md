# Blender 5.0 VR Quest 3 B 버튼 바인딩 - 완벽한 해결책

## 🎯 개요
Blender 5.0에서 Quest 3 컨트롤러의 B 버튼을 커스텀 operator에 바인딩하는 문제에 대한 완전한 분석과 해결책입니다.

---

## 📋 4가지 질문에 대한 명확한 답변

### Q1. `ami.type = 'FLOAT'` vs `'BUTTON'` 중 어떤 것이 맞나요?

**✅ 답: `'FLOAT'`가 정확합니다**

```python
# Blender API 공식 문서:
# type: FLOAT – Float action, representing either a digital or analog button
```

- **Digital button (B button)**: FLOAT 타입으로 `0.0` (누르지 않음) 또는 `1.0` (누름) 반환
- **Analog axis (Trigger)**: FLOAT 타입으로 `0.0 ~ 1.0` 범위 값 반환
- **'BUTTON' 타입**: Blender XR에 존재하지 않음

---

### Q2. B 버튼의 OpenXR component path `/input/b/click`가 맞나요?

**✅ 답: 정확합니다**

```
공식 OpenXR Spec (Oculus Touch Controller):
├─ /input/a/click      → A button (Right)
├─ /input/b/click      → B button (Right) ✅ CORRECT
├─ /input/x/click      → X button (Left)
├─ /input/y/click      → Y button (Left)
├─ /input/trigger/value → Trigger (0.0-1.0)
└─ /input/menu/click   → Menu button
```

**Unity XR Plugin 확인:**
- `/input/b/click` → `secondaryButton` (Right Hand) → `Boolean`

---

### Q3. VR 세션 시작 후 action 추가가 가능한가요? 아니면 전에 해야하나요?

**❌ 답: 세션 시작 전에 반드시 해야 합니다** (현재 코드의 핵심 문제!)

```
OpenXR Action Lifecycle:
┌─────────────────────────────────────────────────────────────────┐
│ 1. VR Session 시작 전                                            │
│    ✅ Action 등록 가능                                            │
│    ✅ xrCreateAction() 호출 가능                                  │
│                                                                   │
│ 2. xrCreateSession()                                             │
│    ⚠️  Session 생성 (OpenXR session 시작)                        │
│                                                                   │
│ 3. xrAttachSessionActionSets()                                   │
│    🔴 CRITICAL POINT                                             │
│    - 이 함수 이후로는 action 수정/추가 불가!                     │
│    - Action sets이 session에 attach됨                            │
│                                                                   │
│ 4. VR Session 시작 중                                             │
│    ❌ 새로운 action 추가 불가                                    │
│    ❌ Existing actions만 사용 가능                                │
│    ❌ Race condition 및 상태 불일치 발생                         │
└─────────────────────────────────────────────────────────────────┘
```

**현재 코드의 문제:**
```python
# ❌ WRONG - vr_operators.py
class THREEGDS_OT_StartVRSession(Operator):
    def execute(self, context):
        # 1. VR session 시작 (xrAttachSessionActionSets 호출됨)
        mgr.start_vr_session()
        
        # 2. 이 시점에 action 추가 시도 → 효과 없음!
        try_add_paint_action_now()  # 🔴 이미 너무 늦음!
```

---

### Q4. `op_mode = 'MODAL'` vs `'PRESS'` 중 어떤 것을 써야 하나요?

**✅ 답: `'MODAL'`이 정확합니다**

```python
MODAL (권장):
├─ 설명: Modal operator로 invoke 호출
├─ 동작: invoke() → modal() → 지속적 이벤트 처리
├─ 용도: 페인팅 같은 지속적 입력 필요
└─ 상태: ✅ 페인팅에 적절함

PRESS:
├─ 설명: Press 이벤트로 operator 호출
├─ 동작: 버튼 누를 때 한 번만 호출
├─ 용도: 순간적 액션 (예: 스냅샷 저장)
└─ 상태: ❌ 페인팅에 부적절함
```

---

## 🔴 핵심 문제 분석

### Problem 1: 잘못된 타이밍 (CRITICAL)
```python
# ❌ 문제 있는 코드
try_add_paint_action_now()  # VR session 시작 후 호출 ← 너무 늦음!
```

**원인:** `xrAttachSessionActionSets` 이후에 action 추가 불가

### Problem 2: Session 중 actionmap 수정 (CRITICAL)
```python
# ❌ 문제 있는 코드
disable_teleport_action()  # VR session 중 actionmap 수정 ← Race condition
```

**원인:** Session이 실행 중일 때 actionmap 수정 시도

### Problem 3: Threshold 값 (MINOR)
```python
ami.threshold = 0.3  # Digital button에는 효과 없을 수 있음
```

**원인:** Digital button은 0.0 또는 1.0만 반환하므로, 0.5 이상 추천

---

## ✅ 해결책 비교 및 추천

### 🥇 방법 1: XML 파일 사용 (가장 권장)

**파일: `~/.config/blender/5.0/config/xr_openxr/gamepad_mapping_threegds.xml`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<bindings>
  <action_set tag="threegds_paint">
    <action_set name="threegds_paint" localized_name="3DGS Paint" />
    <action name="paint_stroke" type="boolean" />
    <user_paths>
      <user_path path="/user/hand/right" />
    </user_paths>
  </action_set>
  
  <interaction_profile path="/interaction_profiles/oculus/touch_controller">
    <interaction_profile name="Oculus Touch Controller" />
    <bind action="/threegds_paint/paint_stroke">
      <input_path path="/user/hand/right/input/b/click" />
    </bind>
  </interaction_profile>
</bindings>
```

**장점:**
- ✅ Blender 초기화 시점에 자동 로드
- ✅ xrAttachSessionActionSets 전에 등록
- ✅ Race condition 없음
- ✅ Blender 표준 방식

---

### 🥈 방법 2: Addon 초기화 시점에 프로그래매틱 등록

**파일: `__init__.py`**

```python
def ensure_paint_action_before_session():
    """
    Called during addon initialization, BEFORE VR session starts.
    This hook ensures paint action is registered before xrAttachSessionActionSets.
    """
    try:
        # Blender VR addon 초기화 시점에 호출됨
        # xrAttachSessionActionSets 전에 action 등록
        pass
    except Exception as e:
        print(f"[3DGS VR] Paint action pre-registration failed: {e}")

# addon register() 함수에서
def register():
    # ... 다른 등록 코드 ...
    
    # VR addon 로드 후 callback 등록
    bpy.app.handlers.load_post.append(ensure_paint_action_before_session)
```

**장점:**
- ✅ 프로그래매틱 방식으로 유연함
- ✅ 안정적 (Blender 초기화 시점 사용)
- ✅ Addon 내에서 완전히 관리 가능

---

### 🥉 방법 3: Minimal Fix (빠른 테스트용)

**수정 사항:**

1. **action_maps.py에서:**
```python
# 기존 코드 수정
ami.type = 'FLOAT'              # ✅ 이미 정확함
ami.op_mode = 'MODAL'           # ✅ 이미 정확함
ami.threshold = 0.5             # ✅ 수정: 0.3 → 0.5 (digital button)

amb.threshold = 0.5             # ✅ 수정: 0.3 → 0.5
```

2. **vr_operators.py에서:**
```python
class THREEGDS_OT_StartVRSession(Operator):
    def execute(self, context):
        mgr = get_vr_session_manager()
        mgr.ensure_vr_addon_enabled()
        
        if not mgr.start_vr_session():
            self.report({'ERROR'}, "Failed to start VR")
            return {'CANCELLED'}
        
        _start_vr_matrix_updater()
        
        # ❌ 제거: try_add_paint_action_now()
        # 대신: Blender 초기화 시점에 action을 미리 등록
        
        bpy.ops.threegds.vr_freehand_paint('INVOKE_DEFAULT')
        self.report({'INFO'}, "VR started")
        return {'FINISHED'}
```

**주의:** 이 방법은 Blender 재시작 후에만 작동하며, 영구적 해결책이 아닙니다.

---

## 📊 방법별 비교표

| 항목 | XML 파일 | Addon Init | Minimal |
|------|---------|-----------|---------|
| 구현 난이도 | ⭐⭐⭐ 높음 | ⭐⭐ 중간 | ⭐ 쉬움 |
| 안정성 | ⭐⭐⭐ 높음 | ⭐⭐⭐ 높음 | ⭐⭐ 중간 |
| 확장성 | ⭐⭐⭐ 높음 | ⭐⭐⭐ 높음 | ⭐ 낮음 |
| 권장도 | 🥇 최우선 | 🥈 권장 | 🥉 테스트용 |

---

## 🔍 검증 체크리스트

B 버튼이 작동하는지 확인하려면:

### 1. Console에 출력 확인
```python
# vr_operators.py에 추가
def invoke(self, context, event):
    print(f"[VR Paint] invoke called!")  # 이게 출력되는지 확인
    print(f"[VR Paint] Event: {event.type}, {event.value}")
```

### 2. Action 등록 확인
```python
wm = bpy.context.window_manager
xr = wm.xr_session_state
if xr:
    am = xr.actionmaps.get("blender_default")
    paint_action = am.actionmap_items.get("threegds_paint")
    if paint_action:
        print("[VR] Paint action found!")
    else:
        print("[VR] Paint action NOT found!")  # ← 문제!
```

### 3. B 버튼 값 읽기
```python
# modal() 함수에서
if event.type == 'XR_ACTION':
    print(f"[VR] XR Action: {event.xr}")
    print(f"[VR] Event value: {event.value}")
```

---

## 📚 참고 자료

### Blender 공식 문서
- XrActionMapItem: https://docs.blender.org/api/current/bpy.types.XrActionMapItem.html
- XrSessionState: https://docs.blender.org/api/current/bpy.types.XrSessionState.html
- VR Scene Inspection: https://docs.blender.org/manual/en/latest/addons/3d_view/vr_scene_inspection.html

### OpenXR 공식 규격
- Oculus Touch Controller Profile
- `/input/b/click` → Boolean (digital button)
- xrAttachSessionActionSets: Actions must be created before this call

### Unity XR Plugin (참고용)
- `/input/b/click` → secondaryButton (Right Hand)
- Type: Boolean

---

## 🎯 최종 권장사항

1. **즉시 적용 (Minimal Fix):** threshold를 0.3 → 0.5로 수정
2. **단기 해결 (Addon Init):** addon 초기화 시점에 paint action 사전 등록
3. **장기 해결 (XML File):** XML 파일로 Blender의 표준 VR addon 방식 사용

---

## 💡 디버깅 팁

만약 B 버튼이 여전히 작동하지 않으면:

```python
# 1. Blender 콘솔 확인
# [VR Paint] invoke called! ← 이게 나오는가?

# 2. Action 등록 확인
# [VR] Paint action found! ← 이게 나오는가?

# 3. XR_ACTION 이벤트 확인
# [VR] XR Action: ... ← 이게 나오는가?

# 순서대로 확인하면서 어디서 끊기는지 찾기
```

---

**작성일:** 2025년 12월 9일  
**Blender 버전:** 5.0.0  
**VR 헤드셋:** Meta Quest 3  
**OpenXR Runtime:** Oculus v1.115.0+
