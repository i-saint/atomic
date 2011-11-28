#ifndef __atomic_SPH_internal_cuh__
#define __atomic_SPH_internal_cuh__


class DeviceBufferObject
{
private:
    cudaGraphicsResource *m_resources[1];
    cudaStream_t m_cuda_stream;
    size_t m_size;
    void *m_dbuffer;

public:
    DeviceBufferObject()
    {
        m_resources[0] = NULL;
        m_cuda_stream = NULL;
        m_size = 0;
        m_dbuffer = NULL;
    }

    ~DeviceBufferObject()
    {
        unregisterBuffer();
    }

    bool isRegistered() const { return m_resources[0]!=NULL; }

    void registerBuffer(int vbo_handle, cudaGraphicsMapFlags usage=cudaGraphicsMapFlagsNone)
    {
        CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(m_resources, vbo_handle, cudaGraphicsMapFlagsNone) );
    }

    void unregisterBuffer()
    {
        if(isRegistered()) {
            CUDA_SAFE_CALL( cudaGraphicsUnregisterResource(m_resources[0]) );
            m_resources[0] = NULL;
        }
    }

    void* mapBuffer()
    {
        CUDA_SAFE_CALL( cudaStreamCreate(&m_cuda_stream) );
        CUDA_SAFE_CALL( cudaGraphicsMapResources(1, m_resources, m_cuda_stream) );
        CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer(&m_dbuffer, &m_size, m_resources[0]) );
        return m_dbuffer;
    }

    void unmapBuffer()
    {
        CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, m_resources, m_cuda_stream) );
        CUDA_SAFE_CALL( cudaStreamDestroy(m_cuda_stream) );
        m_cuda_stream = NULL;
        m_dbuffer = NULL;
    }
};

#endif // __atomic_SPH_internal_cuh__
