#ifndef __ist_i3ddx11__
#define __ist_i3ddx11__
#ifdef __ist_with_DirectX11__

#ifdef _DEBUG
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "d3dx11d.lib")
#pragma comment(lib, "d3dx9d.lib")
#pragma comment(lib, "dxerr.lib")
#pragma comment(lib, "dxguid.lib")
#else // _DEBUG
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "d3dx11.lib")
#pragma comment(lib, "d3dx9.lib")
#pragma comment(lib, "dxerr.lib")
#pragma comment(lib, "dxguid.lib")
#endif // _DEBUG
#include <D3D11.h>
#include <D3DX11.h>
#include "ist/Base/Types.h"
#include "i3ddx11Types.h"
#include "i3ddx11DeviceResource.h"
#include "i3ddx11Device.h"
#include "i3ddx11DeviceContext.h"
#include "i3ddx11Buffer.h"
#include "i3ddx11Shader.h"
#include "i3ddx11Util.h"
#include "i3dudx11Camera.h"
#include "i3dudx11Font.h"

#endif // __ist_with_DirectX11__
#endif // __ist_i3ddx11__
