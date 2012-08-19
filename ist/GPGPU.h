#ifndef __ist_GPGPU_h__
#define __ist_GPGPU_h__

#include "Base/Types.h"

#ifdef __ist_with_OpenCL__
#pragma comment(lib, "OpenCL.lib")
#include "GPUCL/gpuclDevice.h"
#endif // __ist_with_OpenCL__

#endif // __ist_GPGPU_h__
