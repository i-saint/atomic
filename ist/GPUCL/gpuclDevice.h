#ifndef __ist_gpucl_Device__
#define __ist_gpucl_Device__

#if defined (__APPLE__) || defined(MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif 
#include "ist/GraphicsGL/i3dgl.h"
#include "gpuclDeviceResource.h"

namespace ist {
namespace gpucl {

class DeviceResource;
class Program;
class Kernel;

// OpenGL と連携したい場合、OpenGL の初期化の後に初期化する必要がある
class Device
{
private:
    cl_context m_context;
    cl_platform_id m_platform;
    cl_device_id m_device;
    cl_command_queue m_queue;
    stl::vector<DeviceResource*> m_resources;

public:
    Device();
    ~Device();

    cl_device_id getDeviceID() { return m_device; }
    cl_context getContext() { return m_context; }

    Program* createProgram(const char **source, const uint32 *length, const char *compile_options);
    Kernel* createKernel(Program *program, const char *entry_point);

    void deleteResource(DeviceResource *r);

    void queueKernel(Kernel **kernels, uint32 num_kernels);
};


} // namespace gpucl
} // namespace ist
#endif // __ist_gpucl_Device__
