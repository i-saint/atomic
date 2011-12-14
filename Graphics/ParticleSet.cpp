#include "stdafx.h"
#include "types.h"
#include "ParticleSet.h"

namespace atomic {

    ParticleSet::~ParticleSet()
    {
        delete[] m_host_particles;
        cudaFree(m_device_particles);
    }

    void ParticleSet::setCapacity( uint32 max_particles )
    {
        if(m_host_particles!=NULL) {
            IST_ASSERT("re-allocation is not supported.\n");
        }
        m_capacity = max_particles;
        m_host_particles = new vec4[m_capacity];
        cudaMalloc(&m_device_particles, sizeof(particle_t)*m_capacity);
    }

    void ParticleSet::copyHostToDevice()
    {
        cudaMemcpy(m_device_particles, m_host_particles, sizeof(particle_t)*m_capacity, cudaMemcpyHostToDevice);
    }

    void ParticleSet::copyHostToDevice(size_t num)
    {
        if(num > m_capacity) {
            IST_ASSERT("max capacity exceeded.\n");
        }
        cudaMemcpy(m_device_particles, m_host_particles, sizeof(particle_t)*num, cudaMemcpyHostToDevice);
    }

} // namespace atomic
