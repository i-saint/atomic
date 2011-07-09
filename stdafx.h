// stdafx.h : 標準のシステム インクルード ファイルのインクルード ファイル、または
// 参照回数が多く、かつあまり変更されない、プロジェクト専用のインクルード ファイル
// を記述します。
//

#pragma once

#define GLM_FORCE_INLINE    // glm のインライン化と SSE 化
#define GLM_FORCE_SSE2      //
#define GLM_GTX_simd_vec4   //
#define GLM_GTX_simd_mat4   //

#include <stdio.h>
#include <GL/glew.h>
#include <GL/wglew.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//#include <Cg/cgGL.h>
#include <windows.h>
#include <mmsystem.h>
#include <xnamath.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <boost/timer.hpp>
#include <boost/thread.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/regex.hpp>
#include <boost/noncopyable.hpp>
#include <boost/static_assert.hpp>
#include <EASTL/algorithm.h>
#include <EASTL/sort.h>
#include <EASTL/vector.h>
#include <EASTL/list.h>
#include <EASTL/set.h>
#include <EASTL/map.h>
#include <EASTL/string.h>
#include <iostream>
#include <sstream>
#include <fstream>

namespace stl = eastl;

// TODO: プログラムに必要な追加ヘッダーをここで参照してください。
