#ifndef __atomic_Graphics_CudaBuffer__
#define __atomic_Graphics_CudaBuffer__
namespace atomic {

    class CudaBuffer
    {
    private:
        void *m_host_buf;
        void *m_device_buf;
        uint32 m_capacity;

    public:
        CudaBuffer() : m_host_buf(NULL), m_device_buf(NULL), m_capacity(0) {}
        ~CudaBuffer();

        uint32 getCapacity() const  { return m_capacity; }
        void* getHostBuffer()       { return m_host_buf; }
        void* getDeviceBuffer()     { return m_device_buf; }

        void setCapacity(uint32 byte);
        void copyHostToDevice();
        void copyHostToDevice(uint32 byte);
    };

} // namespace atomic
#endif // __atomic_Graphics_CudaBuffer__
