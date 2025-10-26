import OpenGL.EGL as egl

# デフォルトディスプレイ取得
display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
if display == egl.EGL_NO_DISPLAY:
    print("EGL display not available")
else:
    print("EGL display available")

# EGL初期化
major, minor = egl.EGLint(), egl.EGLint()
if egl.eglInitialize(display, major, minor) == 0:
    print("EGL failed to initialize")
else:
    print(f"EGL initialized: version {major.value}.{minor.value}")
