#include "stdafx.h"
#include "gpuclDevice.h"
#include "gpuclDeviceResource.h"

namespace ist {
namespace gpucl {


DeviceResource::DeviceResource( Device *dev )
    : m_dev(dev)
    , m_ref_count(1)
{
}

DeviceResource::~DeviceResource()
{
}

void DeviceResource::addRef()
{
    ++m_ref_count;
}

void DeviceResource::release()
{
    if(--m_ref_count==0) { m_dev->deleteResource(this); }
}


Program::Program( Device *dev, const char **source, const uint32 *length, const char *compile_options )
    : super(dev)
    , m_program(NULL)
{
    cl_int error_code;
    m_program = clCreateProgramWithSource(getDevice()->getContext(), 1, source, length, &error_code);
    error_code = clBuildProgram(m_program, 0, NULL, compile_options, NULL, NULL);
    if(error_code != CL_SUCCESS) {
        char log[1024];
        clGetProgramBuildInfo(m_program, getDevice()->getDeviceID(), CL_PROGRAM_BUILD_LOG,  sizeof(log), log, NULL );
        istPrint(log);
    }
}

Program::~Program()
{
    if(m_program) { clReleaseProgram(m_program); }
}


Kernel::Kernel( Device *dev, Program *program, const char *entry_point )
    : super(dev)
    , m_kernel(NULL)
{
    cl_int error_code;
    m_kernel = clCreateKernel(program->getProgram(), entry_point, &error_code);
}

Kernel::~Kernel()
{
    if(m_kernel) { clReleaseKernel(m_kernel); }
}



BufferBase::BufferBase( Device *dev )
    : super(dev)
    , m_buffer(NULL)
{
}

BufferBase::~BufferBase()
{
}


Buffer::Buffer( Device *dev, IGPU_ACCESS_MODE access_mode, uint32 data_size, void *data )
    : super(dev)
{
    cl_int error_code;
    m_buffer = clCreateBuffer(getDevice()->getContext(), access_mode, data_size, data, &error_code);
}

Buffer::~Buffer()
{
    clReleaseMemObject(m_buffer);
}


GLBuffer::GLBuffer( Device *dev, i3dgl::Buffer *glbuf, IGPU_ACCESS_MODE access_mode )
    : super(dev)
{
    cl_int error_code;
    m_buffer = clCreateFromGLBuffer(getDevice()->getContext(), access_mode, glbuf->getHandle(), &error_code);
}

GLBuffer::~GLBuffer()
{
    clReleaseMemObject(m_buffer);
}

} // namespace gpucl
} // namespace ist
