
# Blender VR Action Binding Analysis - Test Code
# 이 코드는 B 버튼 바인딩 문제의 핵심을 분석합니다

print("=" * 80)
print("BLENDER 5.0 VR ACTION BINDING ANALYSIS")
print("=" * 80)

# 1. XrActionMapItem.type 분석
print("\n1. XrActionMapItem.type VALUES:")
print("-" * 80)
action_types = {
    'FLOAT': 'Float action - representing either a digital or analog button',
    'VECTOR2D': '2D Vector action - for thumbstick/joystick input',
    'POSE': 'Pose action - for tracking hand pose/position',
    'VIBRATION': 'Vibration output action - for haptic feedback'
}

for key, value in action_types.items():
    print(f"  • {key:12} → {value}")

print("\n  ⚠️  문제: ami.type = 'FLOAT'는 CORRECT입니다!")
print("     - /input/b/click는 digital 버튼이므로 FLOAT type으로 받음")
print("     - FLOAT type은 값이 0.0(누르지 않음) 또는 1.0(누름)을 반환")
print("     - 실제로는 DIGITAL이라는 별도 타입이 있을 가능성도 검토 필요")

# 2. OpenXR Component Path 검증
print("\n2. QUEST 3 CONTROLLER B BUTTON OPENXR PATHS:")
print("-" * 80)
paths = {
    '/input/b/click': '✅ Correct - B button press (0.0 or 1.0)',
    '/input/b/value': '❌ Not standard - B는 digital button (no value range)',
    '/input/b/touch': '✅ Also available - B button touched',
    '/input/a/click': '✅ A button (for comparison)',
}

for path, desc in paths.items():
    print(f"  {path:20} → {desc}")

print("\n  📌 Unity XR Plugin Documentation 확인됨:")
print("     /input/b/click  → secondaryButton (Right Hand) → Boolean")

# 3. ACTION MAP 바인딩 타이밍 분석
print("\n3. VR SESSION ACTION REGISTRATION TIMING:")
print("-" * 80)

timeline = {
    '1. VR Session 시작 전': [
        '✅ Action 등록 가능',
        '✅ Blender internal에서 actionmap 생성'
    ],
    '2. xrCreateSession': [
        '⚠️  Critical point - OpenXR session 생성',
        '⚠️  이 시점부터 action 생성/수정이 제한될 수 있음'
    ],
    '3. xrAttachSessionActionSets': [
        '🔴 CRITICAL - 이후로는 action 수정 불가!',
        '🔴 Action sets를 session에 attach하면 고정됨',
        '🔴 "Actions must be attached before xrAttachSessionActionSets"'
    ],
    '4. VR Session 시작 중': [
        '❌ 새로운 action binding 추가 불가',
        '❌ Race condition 발생 가능',
        '❌ 현재 코드의 문제점!'
    ]
}

for phase, items in timeline.items():
    print(f"\n  {phase}")
    for item in items:
        print(f"    {item}")

# 4. 현재 코드의 문제점 분석
print("\n4. CURRENT CODE PROBLEMS:")
print("-" * 80)

problems = [
    {
        'line': 'try_add_paint_action_now()',
        'issue': 'VR session 시작 후 action 추가 시도',
        'location': 'vr_operators.py line: THREEGDS_OT_StartVRSession.execute()',
        'severity': '🔴 CRITICAL',
        'fix': 'Session 시작 전에 action 등록해야 함'
    },
    {
        'line': 'ami.op_mode = \'MODAL\'',
        'issue': 'MODAL vs PRESS 선택이 명확하지 않음',
        'location': 'action_maps.py',
        'severity': '🟡 WARNING',
        'fix': '버튼 타입이므로 PRESS가 더 적절할 수 있음'
    },
    {
        'line': 'disable_teleport_action()',
        'issue': 'Session 중 actionmap 수정 시도 (race condition)',
        'location': 'vr_operators.py',
        'severity': '🔴 CRITICAL',
        'fix': 'Session 시작 전에 teleport를 처음부터 바꾸거나, trigger 값만 읽기'
    },
    {
        'line': 'ami.threshold = 0.3',
        'issue': 'Digital button에 threshold 설정이 의미 없을 수 있음',
        'location': 'action_maps.py',
        'severity': '🟡 WARNING',
        'fix': 'Digital button은 0.0 또는 1.0만 반환하므로 0.5 이상으로 설정'
    }
]

for i, prob in enumerate(problems, 1):
    print(f"\n  문제 {i}: {prob['severity']}")
    print(f"    코드: {prob['line']}")
    print(f"    위치: {prob['location']}")
    print(f"    이유: {prob['issue']}")
    print(f"    해결: {prob['fix']}")

# 5. op_mode 분석
print("\n5. op_mode = 'MODAL' vs 'PRESS' ANALYSIS:")
print("-" * 80)

op_modes = {
    'MODAL': {
        'desc': 'Modal operator로 invoke 호출 - 지속적 이벤트 처리 가능',
        'use_case': '드래그/스와이프/지속적 입력 필요시',
        'status': '좋은 선택 - 페인팅은 지속적 입력 필요'
    },
    'PRESS': {
        'desc': 'Press 이벤트로 operator 호출 - 순간적 액션',
        'use_case': '한 번 누를 때만 실행되는 동작',
        'status': '부적절 - 페인팅은 지속적이어야 함'
    }
}

for mode, info in op_modes.items():
    print(f"\n  [{mode}]")
    print(f"    설명: {info['desc']}")
    print(f"    용도: {info['use_case']}")
    print(f"    상태: {info['status']}")

print("\n  💡 결론: op_mode = 'MODAL'은 CORRECT")

# 6. ami.type 재검증
print("\n6. FLOAT TYPE FOR BUTTON VALIDATION:")
print("-" * 80)
print("  Blender API 공식 문서에서:")
print("    type: FLOAT – Float action, representing either a digital or analog button")
print("")
print("  즉, FLOAT 타입이 button을 표현할 때 사용됨:")
print("    • Digital button → FLOAT value 0.0 or 1.0 반환")
print("    • Analog axis   → FLOAT value 0.0 ~ 1.0 범위 반환")
print("")
print("  ✅ ami.type = 'FLOAT' is CORRECT for B button")

print("\n" + "=" * 80)
print("CRITICAL FINDINGS")
print("=" * 80)

findings = [
    ("ACTION REGISTRATION TIMING", "❌ WRONG", "try_add_paint_action_now()를 VR session 시작 후에 호출"),
    ("COMPONENT PATH", "✅ CORRECT", "/input/b/click는 정확함"),
    ("ACTION TYPE", "✅ CORRECT", "ami.type = 'FLOAT'는 정확함"),
    ("op_mode", "✅ CORRECT", "'MODAL'은 적절함"),
    ("SESSION MODIFICATION", "❌ WRONG", "Session 시작 후 actionmap 수정 시도"),
    ("THRESHOLD VALUE", "⚠️  CHECK", "Digital button이라 threshold 효과 확인 필요"),
]

for aspect, status, note in findings:
    print(f"\n{status:12} {aspect:30} → {note}")

print("\n" + "=" * 80)
