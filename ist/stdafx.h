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
#include <GL/glew.h>
#include <GL/wglew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <D3D11.h>
#include <D3DX11.h>
#include <AL/al.h>
#include <AL/alc.h>

#include <EASTL/algorithm.h>
#include <EASTL/sort.h>
#include <EASTL/vector.h>
#include <EASTL/list.h>
#include <EASTL/set.h>
#include <EASTL/map.h>
#include <EASTL/string.h>
namespace stl = eastl;

#define __ist_enable_assert__
#define __ist_enable_graphics_assert__

#define __ist_with_OpenGL__
#define __ist_with_DirectX11__
#define __ist_with_zlib__
#define __ist_with_oggvorbis__


// TODO: プログラムに必要な追加ヘッダーをここで参照してください。
