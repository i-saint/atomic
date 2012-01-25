#ifndef __ist_gpucl_Device__
#define __ist_gpucl_Device__

#if defined (__APPLE__) || defined(MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif 

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

    Program* createProgram(const char **source, const uint32 *length, const char *compile_options);
    Kernel* createKernel(Program *program, const char *entry_point);

    void queueKernel(Kernel **kernels, uint32 num_kernels);
};


class DeviceResource
{
private:
    Device *m_dev;

public:
    DeviceResource();
    virtual ~DeviceResource();
    void addRef();
    void release();
};

class Program : public DeviceResource
{
private:
    cl_program m_program;

    Program();
    ~Program();
    void initialize(const char **source, const uint32 *length, const char *compile_options);

public:
    cl_program getProgram() { return m_program; }
};

class Kernel : public DeviceResource
{
private:
    cl_kernel m_kernel;

    Kernel();
    ~Kernel();
    void initialize(Program *program, const char *entry_point);

public:
    cl_kernel getKernel() { return m_kernel; }
};

} // namespace gpucl
} // namespace ist
#endif // __ist_gpucl_Device__
