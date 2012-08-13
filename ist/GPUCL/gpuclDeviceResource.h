#ifndef __ist_gpucl_DeviceResource__
#define __ist_gpucl_DeviceResource__

#include "stdafx.h"
#include "ist/ist.h"
#include "gpuclDevice.h"

namespace ist {
namespace gpucl {

#define GPUCL_DECLARE_DEVICE_RESOURCE(classname) \
private:\
    template<class T> friend T* ::call_destructor(T *v);\
    friend class Device;

class Device;



enum IGPU_ACCESS_MODE {
    IGPU_READ_ONLY  = CL_MEM_READ_ONLY,
    IGPU_WRITE_ONLY = CL_MEM_WRITE_ONLY,
    IGPU_READ_WRITE = CL_MEM_READ_WRITE,
};

class DeviceResource
{
GPUCL_DECLARE_DEVICE_RESOURCE(DeviceResource);
private:
    Device *m_dev;
    int32 m_ref_count;

protected:
    Device* getDevice();

public:
    DeviceResource(Device *dev);
    virtual ~DeviceResource();
    void addRef();
    void release();
};


class Program : public DeviceResource
{
GPUCL_DECLARE_DEVICE_RESOURCE(Program);
typedef DeviceResource super;
private:
    cl_program m_program;

    Program(Device *dev, const char **source, const uint32 *length, const char *compile_options="-cl-fast-relaxed-math");
    ~Program();

public:
    cl_program getProgram() { return m_program; }
};


class Kernel : public DeviceResource
{
GPUCL_DECLARE_DEVICE_RESOURCE(Kernel);
typedef DeviceResource super;
private:
    cl_kernel m_kernel;

    Kernel(Device *dev, Program *program, const char *entry_point);
    ~Kernel();

public:
    cl_kernel getKernel() { return m_kernel; }
};


class BufferBase : public DeviceResource
{
GPUCL_DECLARE_DEVICE_RESOURCE(BufferBase);
typedef DeviceResource super;
protected:
    cl_mem m_buffer;

    BufferBase(Device *dev);
    ~BufferBase();

public:
    cl_mem getMemory() { return m_buffer; }
};

class Buffer : public BufferBase
{
GPUCL_DECLARE_DEVICE_RESOURCE(Buffer);
typedef BufferBase super;
private:
    Buffer(Device *dev, IGPU_ACCESS_MODE access_mode, uint32 data_size, void *data=NULL);
    ~Buffer();
};

class GLBuffer : public BufferBase
{
GPUCL_DECLARE_DEVICE_RESOURCE(GLBuffer);
typedef BufferBase super;
private:
    GLBuffer(Device *dev, i3dgl::Buffer *glbuf, IGPU_ACCESS_MODE access_mode);
    ~GLBuffer();
};

} // namespace gpucl
} // namespace ist
#endif // __ist_gpucl_DeviceResource__
