#ifndef __ist_Config_h__

#ifndef istMasterBuild
#   define __ist_enable_assert__
#   define __i3d_enable_assert__
#endif // istMasterBuild
#ifdef istDebugBuild
#   define __ist_enable_memory_leak_check__
#   define __i3d_enable_resource_leak_check__
#endif // istDebugBuild

#define __ist_with_EASTL__
#define __ist_with_OpenGL__
//#define __ist_with_OpenGLES__
//#define __ist_with_DirectX11__
#define __ist_with_zlib__
#define __ist_with_png__
//#define __ist_with_jpeg__
#define __ist_with_gli__ // dds ファイル対応
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


//#define istExportSymbols

#ifdef istWindows
#   if defined(istExportSymbols)
#       define istInterModule __declspec(dllexport)
#   elif defined(istImportSymbols)
#       define istInterModule __declspec(dllimport)
#   else
#       define istInterModule
#   endif // istExportSymbols
#else // istWindows
#   define istInterModule
#endif // istWindows



#define WIN32_LEAN_AND_MEAN             // Windows ヘッダーから使用されていない部分を除外します。
#define _SCL_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS

#pragma warning(disable: 4819) // コードページ問題 (glm)
#pragma warning(disable: 4251) // __export つけろ問題

#include <stdint.h>
#include <stdio.h>

#ifdef __ist_with_EASTL__
#   include <EASTL/algorithm.h>
#   include <EASTL/sort.h>
#   include <EASTL/vector.h>
#   include <EASTL/list.h>
#   include <EASTL/set.h>
#   include <EASTL/map.h>
#   include <EASTL/string.h>
namespace stl = eastl;
#else // __ist_with_EASTL__
#   include <vector>
#   include <list>
#   include <map>
#   include <string>
#   include <algorithm>
namespace stl = std;
#endif // __ist_with_EASTL__

#ifdef __ist_with_DirectX11__
#   include <D3D11.h>
#   include <D3DX11.h>
#endif // __ist_with_DirectX11__

#ifdef __ist_with_OpenGL__
#   include <GL/glew.h>
#   ifdef istWindows
#       include <GL/wglew.h>
#       pragma comment(lib, "glew32.lib")
#       pragma comment(lib, "opengl32.lib")
#   endif // istWindows
#endif // __ist_with_OpenGL__

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

#ifdef __ist_with_OpenAL__
#   include <AL/al.h>
#   include <AL/alc.h>
#endif // __ist_with_OpenAL__

#include "ist/Base/Decl.h"
#include "ist/Base/Types.h"

#endif // __ist_Config_h__
