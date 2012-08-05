#include "stdafx.h"
#include "types.h"
#include "ist/Base.h"
#include "CudaBuffer.h"

namespace atomic {

    CudaBuffer::~CudaBuffer()
    {
        istFree(m_host_buf);
        cudaFree(m_device_buf);
    }

    void CudaBuffer::setCapacity( uint32 byte )
    {
        istAssert(m_host_buf==NULL, "re-allocation is not supported.\n");
        m_capacity = byte;
        m_host_buf = istAlignedMalloc(m_capacity, 16);
        cudaMalloc(&m_device_buf, m_capacity);
    }

    void CudaBuffer::copyHostToDevice()
    {
        cudaMemcpy(m_device_buf, m_host_buf, m_capacity, cudaMemcpyHostToDevice);
    }

    void CudaBuffer::copyHostToDevice( uint32 byte )
    {
        istAssert(byte<=m_capacity, "max capacity exceeded.\n");
        cudaMemcpy(m_device_buf, m_host_buf, byte, cudaMemcpyHostToDevice);
    }

} // namespace atomic
