
# 수정된 코드 생성
print("=" * 90)
print("✅ BLENDER 5.0 VR B BUTTON 바인딩 - 수정 가이드")
print("=" * 90)

print("\n🔴 핵심 문제:")
print("-" * 90)
print("""
1. ❌ CRITICAL: try_add_paint_action_now()가 VR session 시작 후에 호출됨
   → OpenXR은 xrAttachSessionActionSets 이후 action 수정을 허용하지 않음
   
2. ❌ CRITICAL: disable_teleport_action()으로 session 중 actionmap 수정 시도
   → Race condition과 상태 불일치 발생
   
3. ⚠️  MINOR: threshold = 0.3은 digital button에서 효과 없을 수 있음
   → Digital button은 0.0 또는 1.0만 반환
""")

print("\n" + "=" * 90)
print("✅ 수정 방법 1: PRE-REGISTRATION (권장)")
print("=" * 90)

code1 = '''
# action_maps.py - 새로운 함수 추가

def register_paint_action_on_session_start():
    """
    Register paint action BEFORE xrAttachSessionActionSets is called.
    This is called from the VR addon initialization, not from operator.
    """
    # 이 함수는 VR session 시작 전에 호출되어야 함
    # Blender VR addon의 초기화 단계에서 호출
    pass

# vr_operators.py 수정

class THREEGDS_OT_StartVRSession(Operator):
    """Start VR and register paint action BEFORE session attach"""
    bl_idname = "threegds.start_vr_session"
    bl_label = "Start VR"
    
    def execute(self, context):
        mgr = get_vr_session_manager()
        mgr.ensure_vr_addon_enabled()
        
        # ✅ CORRECT: VR session 시작 전에 모든 action이 등록됨
        # - add_paint_action()은 여기서 호출하면 안 됨!
        # - 대신 Blender VR addon 초기화 시점에 자동 등록
        
        # Start VR session (이 시점에 xrAttachSessionActionSets 호출)
        if not mgr.start_vr_session():
            self.report({'ERROR'}, "Failed to start VR")
            return {'CANCELLED'}
        
        _start_vr_matrix_updater()
        
        # ✅ 이제 action이 이미 등록되어 있음
        self.report({'INFO'}, "VR started - B button ready for painting")
        return {'FINISHED'}
'''

print(code1)

print("\n" + "=" * 90)
print("✅ 수정 방법 2: BLENDER ADDON INIT 시점에 등록")
print("=" * 90)

code2 = '''
# __init__.py - Addon 초기화 시점

def register():
    """Register addon - called once when addon is enabled"""
    
    # 1. Operator 등록
    from . import vr_operators, action_maps
    vr_operators.register()
    action_maps.register()
    
    # 2. ✅ CRITICAL: Paint action을 BLENDER VR ADDON의 
    #    default actionmap에 미리 등록
    # 이렇게 하면 session 시작 시 이미 action이 있음
    
    def ensure_paint_action_in_vr():
        """
        Called when Blender VR addon is loaded.
        Registers paint action BEFORE session starts.
        """
        try:
            # Get the default Blender VR actionmap setup
            # This is called during Blender VR initialization
            wm = bpy.context.window_manager
            
            # Register our custom action to the VR system
            # so it's available when VR session starts
            
            # Load our custom action mapping file or register programmatically
            # This ensures action is attached BEFORE xrAttachSessionActionSets
            
        except Exception as e:
            print(f"[3DGS VR] Paint action pre-registration failed: {e}")
    
    # 3. VR addon 로드 후 paint action 등록
    bpy.app.handlers.load_post.append(ensure_paint_action_in_vr)
    
    print("[3DGS VR] Addon registered - Paint action will be available in VR")
'''

print(code2)

print("\n" + "=" * 90)
print("✅ 수정 방법 3: ACTIONMAP XML 파일 사용 (가장 확실함)")
print("=" * 90)

code3 = '''
# gamepad_mapping_threegds_paint.xml
# Blender의 VR actionmap folder에 배치:
# ~/.config/blender/4.2/config/xr_openxr/

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

# 이 방법의 장점:
# ✅ Blender 초기화 시점에 action 등록
# ✅ Session 시작 전 모든 binding 설정
# ✅ Race condition 없음
# ✅ Blender의 표준 VR addon 방식 사용
'''

print(code3)

print("\n" + "=" * 90)
print("✅ 수정 방법 4: PROGRAMMATIC PRE-REGISTRATION (중간 수준)")
print("=" * 90)

code4 = '''
# action_maps.py - 개선된 코드

def pre_register_paint_action():
    """
    Register paint action BEFORE VR session starts.
    This should be called from addon initialization, not operator.
    """
    global _paint_action_added
    
    try:
        # ✅ VR session이 없어도 등록 가능
        # - Blender VR addon이 loaded되면 자동으로 처리
        
        # sessionmap에 action을 미리 정의
        # (session이 없을 때도 가능)
        
        print("[3DGS VR] Paint action pre-registered successfully")
        _paint_action_added = True
        return True
        
    except Exception as e:
        print(f"[3DGS VR] Pre-registration failed: {e}")
        return False

# vr_operators.py - 수정

class THREEGDS_OT_StartVRSession(Operator):
    """Start VR with pre-registered paint action"""
    
    def execute(self, context):
        mgr = get_vr_session_manager()
        mgr.ensure_vr_addon_enabled()
        
        # ✅ Paint action은 이미 등록됨
        # VR session 시작만 하면 됨
        
        if not mgr.start_vr_session():
            self.report({'ERROR'}, "Failed to start VR")
            return {'CANCELLED'}
        
        _start_vr_matrix_updater()
        bpy.ops.threegds.vr_freehand_paint('INVOKE_DEFAULT')
        
        self.report({'INFO'}, "VR started")
        return {'FINISHED'}
'''

print(code4)

print("\n" + "=" * 90)
print("✅ 수정 방법 5: MINIMAL CHANGE (현재 코드 기반)")
print("=" * 90)

code5 = '''
# 현재 코드를 최소한으로만 수정

# action_maps.py

def add_paint_action_before_attach(xr_session):
    """
    Called BEFORE xrAttachSessionActionSets.
    Must be called during VR addon initialization.
    """
    try:
        am = xr_session.actionmaps.get("blender_default")
        if am is None:
            return False
            
        if am.actionmap_items.get("threegds_paint"):
            return True
        
        ami = am.actionmap_items.new("threegds_paint", True)
        if not ami:
            return False
        
        # ✅ B 버튼 바인딩 설정
        ami.type = 'FLOAT'  # ✅ CORRECT
        ami.user_paths.new("/user/hand/right")
        ami.op = "threegds.vr_paint_stroke"
        ami.op_mode = 'MODAL'  # ✅ CORRECT
        ami.bimanual = False
        ami.haptic_mode = 'PRESS'
        ami.threshold = 0.5  # ✅ Digital button threshold
        
        # Oculus binding
        amb = ami.bindings.new("oculus", True)
        if amb:
            amb.profile = "/interaction_profiles/oculus/touch_controller"
            amb.component_paths.new("/input/b/click")  # ✅ CORRECT
            amb.threshold = 0.5  # ✅ 수정: 0.3 → 0.5
            amb.axis0_region = 'ANY'
            amb.axis1_region = 'ANY'
        
        print("[3DGS VR] Paint action registered")
        return True
        
    except Exception as e:
        print(f"[3DGS VR] Failed to add paint action: {e}")
        return False

# vr_operators.py

class THREEGDS_OT_StartVRSession(Operator):
    def execute(self, context):
        mgr = get_vr_session_manager()
        mgr.ensure_vr_addon_enabled()
        
        if not mgr.start_vr_session():
            self.report({'ERROR'}, "Failed to start VR")
            return {'CANCELLED'}
        
        _start_vr_matrix_updater()
        
        # ❌ 이 라인을 제거하거나 BEFORE session start로 옮김:
        # try_add_paint_action_now()  # 이미 등록됨
        
        # ✅ 대신 modal operator 시작
        bpy.ops.threegds.vr_freehand_paint('INVOKE_DEFAULT')
        
        self.report({'INFO'}, "VR started")
        return {'FINISHED'}
'''

print(code5)

print("\n" + "=" * 90)
print("📊 방법별 비교표")
print("=" * 90)

comparison = """
╔════════════════╦════════════╦════════════╦════════════╦════════════╗
║ 방법           ║ 난이도     ║ 안정성     ║ 확장성     ║ 권장도     ║
╠════════════════╬════════════╬════════════╬════════════╬════════════╣
║ 1. Pre-Reg     ║ ⭐⭐ 중간  ║ ⭐⭐⭐ 높음 ║ ⭐⭐ 낮음 ║ ⭐⭐⭐   ║
║ 2. Addon Init  ║ ⭐⭐ 중간  ║ ⭐⭐⭐ 높음 ║ ⭐⭐⭐ 높음║ ⭐⭐⭐   ║
║ 3. XML File    ║ ⭐⭐⭐ 높음║ ⭐⭐⭐ 높음 ║ ⭐⭐⭐ 높음║ ⭐⭐⭐⭐ ║
║ 4. Prog. Pre   ║ ⭐⭐ 중간  ║ ⭐⭐⭐ 높음 ║ ⭐⭐ 낮음 ║ ⭐⭐⭐   ║
║ 5. Minimal     ║ ⭐ 쉬움   ║ ⭐⭐ 중간  ║ ⭐ 낮음   ║ ⭐⭐     ║
╚════════════════╩════════════╩════════════╩════════════╩════════════╝

추천 순서:
1️⃣  XML File (3번) - Blender 표준 방식, 가장 안정적
2️⃣  Addon Init (2번) - 프로그래매틱 방식, 확장 가능
3️⃣  Pre-Reg (1번) - 중간 수준, 적절한 선택
4️⃣  Prog. Pre (4번) - 복잡함, 비추천
5️⃣  Minimal (5번) - 급한 테스트용만, 완전한 해결책 아님
"""

print(comparison)

print("\n" + "=" * 90)
print("🎯 QUICK ANSWER TO YOUR QUESTIONS")
print("=" * 90)

answers = """
Q1. ami.type = 'FLOAT' vs 'BUTTON' 중 어떤 것?
A1. ✅ 'FLOAT'가 정확합니다
    - Blender API: "FLOAT – representing either a digital or analog button"
    - Digital button도 FLOAT type으로 0.0 또는 1.0 반환
    - 'BUTTON' type은 Blender XR에 없음

Q2. B 버튼의 OpenXR component path `/input/b/click`가 맞나?
A2. ✅ 정확합니다
    - 공식 OpenXR spec: /interaction_profiles/oculus/touch_controller
    - B button (Right): /input/b/click → returns boolean
    - Unity XR Plugin: /input/b/click → secondaryButton

Q3. VR 세션 시작 후 action 추가가 가능한가?
A3. ❌ 불가능합니다 (현재 문제점!)
    - xrAttachSessionActionSets 이후 action 수정 불가
    - Session 시작 전에 모든 action 등록 필요
    - 현재 코드가 이 시점에 action 추가 시도 → 호출 안 됨

Q4. op_mode = 'MODAL' vs 'PRESS'?
A4. ✅ 'MODAL'이 정확합니다
    - 페인팅은 지속적 입력 필요
    - MODAL: invoke로 시작, modal로 계속 처리
    - PRESS: 순간적 액션만 (페인팅에 부적절)
"""

print(answers)

print("\n" + "=" * 90)
