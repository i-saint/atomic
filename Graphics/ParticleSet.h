#ifndef __atomic_Graphics_ParticleSet__
#define __atomic_Graphics_ParticleSet__
namespace atomic {

    class ParticleSet
    {
    private:
        typedef vec4 particle_t;
        particle_t *m_host_particles;
        particle_t *m_device_particles;
        uint32 m_capacity;

    public:
        ParticleSet() : m_host_particles(NULL), m_device_particles(NULL), m_capacity(0) {}
        ~ParticleSet();

        uint32 getCapacity() const { return m_capacity; }
        particle_t* getHostParticles() { return m_host_particles; }
        particle_t* getDeviceParticles() { return m_device_particles; }

        void setCapacity(uint32 max_particles);
        void copyHostToDevice();
        void copyHostToDevice(size_t num);
    };

} // namespace atomic
#endif // __atomic_Graphics_ParticleSet__
