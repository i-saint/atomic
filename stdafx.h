// stdafx.h : 標準のシステム インクルード ファイルのインクルード ファイル、または
// 参照回数が多く、かつあまり変更されない、プロジェクト専用のインクルード ファイル
// を記述します。
//

#pragma once

#pragma warning(disable: 4819)

#define _SCL_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS

#define GLM_FORCE_SSE2

#include <windows.h>
#include <stdio.h>
#include <GL/glew.h>
#include <GL/wglew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


#include <mmsystem.h>
#include <stdlib.h>
#include <intrin.h>

#include <tbb/tbb.h>
#include <iostream>


#include <windows.h>
#include <windowsx.h>

// leak check on Debug configuration
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif // _DEBUG

#include "ist/ist.h"
#include "ist/Graphics.h"
#include "ist/Sound.h"
#include "features.h"
#include "types.h"
