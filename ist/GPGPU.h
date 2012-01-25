#ifndef __ist_GPGPU__
#define __ist_GPGPU__

#include "Base/Types.h"

#ifdef __ist_with_OpenCL__
#pragma comment(lib, "OpenCL.lib")
#include "GPUCL/gpuclDevice.h"
#endif // __ist_with_OpenCL__

#endif // __ist_GPGPU__
