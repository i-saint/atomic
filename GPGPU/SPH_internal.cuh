#ifndef __atomic_SPH_internal_cuh__
#define __atomic_SPH_internal_cuh__


// OpenGL のバッファオブジェクトを CUDA にマッピングするヘルパ
class DeviceBufferObject
{
private:
    cudaGraphicsResource *m_resources;
    cudaStream_t m_cuda_stream;
    size_t m_size;
    void *m_dbuffer;

public:
    DeviceBufferObject()
    {
        m_resources = NULL;
        m_cuda_stream = NULL;
        m_size = 0;
        m_dbuffer = NULL;
    }

    ~DeviceBufferObject()
    {
        unregisterBuffer();
    }

    bool isRegistered() const { return m_resources!=NULL; }

    void registerBuffer(int vbo_handle, cudaGraphicsMapFlags usage=cudaGraphicsMapFlagsNone)
    {
        CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(&m_resources, vbo_handle, cudaGraphicsMapFlagsNone) );
    }

    void unregisterBuffer()
    {
        if(isRegistered()) {
            CUDA_SAFE_CALL( cudaGraphicsUnregisterResource(m_resources) );
            m_resources = NULL;
        }
    }

    void* mapBuffer()
    {
        CUDA_SAFE_CALL( cudaStreamCreate(&m_cuda_stream) );
        CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &m_resources, m_cuda_stream) );
        CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer(&m_dbuffer, &m_size, m_resources) );
        return m_dbuffer;
    }

    void unmapBuffer()
    {
        CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &m_resources, m_cuda_stream) );
        CUDA_SAFE_CALL( cudaStreamDestroy(m_cuda_stream) );
        m_cuda_stream = NULL;
        m_dbuffer = NULL;
    }
};



struct sphParam
{
    float smooth_len;
    float pressure_stiffness;
    float rest_density;
    float particle_mass;
    float viscosity;
    float density_coef;
    float grad_pressure_coef;
    float lap_viscosity_coef;
    float wall_stiffness;
};


struct sphGridParam
{
    float4 grid_dim;
    float4 grid_dim_rcp;
    float4 grid_pos;
};

#endif // __atomic_SPH_internal_cuh__
