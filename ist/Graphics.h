#ifndef __ist_Graphics__
#define __ist_Graphics__

#include <glm/glm.hpp>
#include "Base/Types.h"

#ifdef __ist_with_OpenGL__
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#include <GL/glew.h>
#include <GL/wglew.h>
#include "GraphicsGL/i3dglTypes.h"
#include "GraphicsGL/i3dglDeviceResource.h"
#include "GraphicsGL/i3dglDevice.h"
#include "GraphicsGL/i3dglDeviceContext.h"
#include "GraphicsGL/i3dglBuffer.h"
#include "GraphicsGL/i3dglShader.h"
#include "GraphicsGL/i3dglUtil.h"
#include "GraphicsGL/i3duglCamera.h"
#include "GraphicsGL/i3duglFont.h"
#endif // __ist_with_OpenGL__

#ifdef __ist_with_DirectX11__
#ifdef _DEBUG
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "d3dx11d.lib")
#pragma comment(lib, "d3dx9d.lib")
#pragma comment(lib, "dxerr.lib")
#pragma comment(lib, "dxguid.lib")
#else
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "d3dx11.lib")
#pragma comment(lib, "d3dx9.lib")
#pragma comment(lib, "dxerr.lib")
#pragma comment(lib, "dxguid.lib")
#endif

#include <GL/glew.h>
#include <D3D11.h>
#include <D3DX11.h>
#include "GraphicsDX11/i3ddx11Types.h"
#include "GraphicsDX11/i3ddx11DeviceResource.h"
#include "GraphicsDX11/i3ddx11Device.h"
#include "GraphicsDX11/i3ddx11DeviceContext.h"
#include "GraphicsDX11/i3ddx11Buffer.h"
#include "GraphicsDX11/i3ddx11Shader.h"
#include "GraphicsDX11/i3ddx11Util.h"
#include "GraphicsDX11/i3dudx11Camera.h"
#include "GraphicsDX11/i3dudx11Font.h"
#endif // __ist_with_DirectX11__

#endif // __ist_Graphics__
