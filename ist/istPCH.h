// stdafx.h : 標準のシステム インクルード ファイルのインクルード ファイル、または
// 参照回数が多く、かつあまり変更されない、プロジェクト専用のインクルード ファイル
// を記述します。
//

#pragma once

#define WIN32_LEAN_AND_MEAN             // Windows ヘッダーから使用されていない部分を除外します。
#define _SCL_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4819)

#include <stdint.h>
#include <stdio.h>

#include "ist/Config.h"

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
