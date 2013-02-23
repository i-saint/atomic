#ifndef ist_Graphics_h
#define ist_Graphics_h

#include "Config.h"
#include "Base.h"
#include "Math.h"
#include "GraphicsCommon/Image.h"

#ifdef ist_with_OpenGL
#include "GraphicsGL/i3dgl.h"
#endif // ist_with_OpenGL

#ifdef ist_with_OpenGLES
#include "GraphicsGLES/i3dgles.h"
#endif // ist_with_OpenGL

#ifdef ist_with_DirectX11
#include "GraphicsDX11//i3ddx11.h"
#endif // ist_with_DirectX11

#endif // ist_Graphics_h
