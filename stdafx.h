// stdafx.h : 標準のシステム インクルード ファイルのインクルード ファイル、または
// 参照回数が多く、かつあまり変更されない、プロジェクト専用のインクルード ファイル
// を記述します。
//

#pragma once

#pragma warning(disable: 4819)

#define _SCL_SECURE_NO_WARNINGS

#include <windows.h>
#include <stdio.h>
#include <GL/glew.h>
#include <GL/wglew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cuda.h>
#include <cudaGL.h>
#include <cuda_runtime.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include <cutil.h>
#include <cutil_math.h>
#include <cutil_inline_runtime.h>
#include <math_constants.h>

#include <thrust/host_vector.h>

#include <mmsystem.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <boost/thread.hpp>
#include <boost/noncopyable.hpp>
#include <boost/static_assert.hpp>
#include <boost/mem_fn.hpp>
#include <boost/function.hpp>
#include <EASTL/algorithm.h>
#include <EASTL/sort.h>
#include <EASTL/vector.h>
#include <EASTL/list.h>
#include <EASTL/set.h>
#include <EASTL/map.h>
#include <EASTL/string.h>

#include <tbb/tbb.h>

#include <iostream>

namespace stl = eastl;

#include <windows.h>
#include <windowsx.h>

// leak check on Debug configuration
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif // _DEBUG

#include "features.h"
