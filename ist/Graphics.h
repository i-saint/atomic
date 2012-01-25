#ifndef __ist_Graphics__
#define __ist_Graphics__

#ifdef __ist_with_OpenGL__
#include "GraphicsGL/i3dgl.h"
#endif // __ist_with_OpenGL__

#ifdef __ist_with_DirectX11__
#include "GraphicsDX11//i3ddx11.h"
#endif // __ist_with_DirectX11__

#endif // __ist_Graphics__
