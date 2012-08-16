#ifndef __ist_Config_h__

#define __ist_enable_assert__
#define __ist_enable_graphics_assert__

#define __ist_with_EASTL__
#define __ist_with_OpenGL__
#define __ist_with_OpenGLES__
#define __ist_with_DirectX11__
#define __ist_with_zlib__
#define __ist_with_png__
//#define __ist_with_jpeg__
#define __ist_with_OpenAL__
#define __ist_with_oggvorbis__

#if defined(_WIN64)
#   define istWindows
#   define istWin64
#   define istx86
#   define istx86_64
#elif defined(_WIN32)
#   define istWindows
#   define istWin32
#   define istx86
#elif defined(__ANDROID__)
#   define istAndroid
#   define istARM
#else
#   error
#endif


#ifdef __ist_with_zlib__
#   define ZLIB_DLL
#   include "zlib/zlib.h"
#   pragma comment(lib, "zdll.lib")
#endif // __ist_with_zlib__

#ifdef __ist_with_png__
#   include <libpng/png.h>
#   pragma comment(lib,"libpng15.lib")
#endif // __ist_with_png__
#ifdef __ist_with_jpeg__
#   include <jpeglib.h>
#   include <jerror.h>
#   pragma comment(lib,"libjpeg.lib")
#endif // __ist_with_jpeg__


#endif // __ist_Config_h__
