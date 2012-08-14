#ifndef __ist__
#define __ist__

//feature macros:
//
//#define __ist_enable_assert__
//#define __ist_enable_file_log__
//#define __ist_enable_graphics_assert__
//
//#define __ist_with_OpenGL__
//#define __ist_with_DirectX11__
//#define __ist_with_OpenCL__
//#define __ist_with_zlib__
//#define __ist_with_oggvorbis__

#ifdef _DEBUG
#pragma comment(lib, "istd.lib")
#else
#pragma comment(lib, "ist.lib")
#endif // _DEBUG

#include "Config.h"
#include "Base.h"
#include "Debug.h"
#include "Concurrency.h"
#include "Math.h"
#include "Graphics.h"
#include "GraphicsUtil.h"
#include "Window.h"
#include "GPGPU.h"

#endif // __ist__
