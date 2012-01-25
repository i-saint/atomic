#include "stdafx.h"
#include "ist/ist.h"
#include "gpuclDevice.h"

#if defined (__APPLE__) || defined(MACOSX)
#define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
#define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#endif

namespace ist {
namespace gpucl {


Device::Device()
    : m_context(NULL)
    , m_platform(NULL)
    , m_device(NULL)
    , m_queue(NULL)
{
    cl_int error_code = 0;

    // select platform
    {
        cl_uint num_platforms; 
        clGetPlatformIDs(0, NULL, &num_platforms);
        if(num_platforms == 0) {
            istPrint("No OpenCL platform found!\n\n");
            return;
        }

        stl::vector<cl_platform_id> platform_ids(num_platforms);
        clGetPlatformIDs(num_platforms, &platform_ids[0], NULL);
        for(cl_uint i = 0; i < num_platforms; ++i)
        {
            char name[256];
            clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, _countof(name), &name, NULL);
            istPrint("OpenCL platform %d: %s\n", name);
        }
        m_platform = platform_ids[0];
    }

    // select device
    {
        cl_uint dev_count;
        clGetDeviceIDs(m_platform, CL_DEVICE_TYPE_GPU, 0, NULL, &dev_count);

        stl::vector<cl_device_id> devices(dev_count);
        clGetDeviceIDs(m_platform, CL_DEVICE_TYPE_GPU, dev_count, &devices[0], NULL);

        // search device supports context sharing with OpenGL
        cl_uint dev_selected = ~0;
        bool sharing_supported = false;
        for(uint32 i=0; i<dev_count; ++i) {
            size_t extensionSize;
            clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize );
            if(extensionSize > 0) {
                char* extensions = (char*)malloc(extensionSize);
                clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, extensionSize, extensions, &extensionSize);
                stl::string stdDevString(extensions);
                free(extensions);

                size_t szOldPos = 0;
                size_t szSpacePos = stdDevString.find(' ', szOldPos); // extensions string is space delimited
                while (szSpacePos != stdDevString.npos) {
                    if( strcmp(GL_SHARING_EXTENSION, stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0 ) {
                        // Device supports context sharing with OpenGL
                        dev_selected = i;
                        sharing_supported = true;
                        break;
                    }
                    do {
                        szOldPos = szSpacePos + 1;
                        szSpacePos = stdDevString.find(' ', szOldPos);
                    } while (szSpacePos == szOldPos);
                }
            }
        }

        if(dev_selected!=~0) {
            m_device = devices[dev_selected];
#if defined (__APPLE__)
            CGLContextObj kCGLContext = CGLGetCurrentContext();
            CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
            cl_context_properties props[] = 
            {
                CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup, 
                0 
            };
            m_context = clCreateContext(props, 0,0, NULL, NULL, &error_code);
#else // __APPLE__
#ifdef UNIX
            cl_context_properties props[] = 
            {
                CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), 
                CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), 
                CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
                0
            };
            m_context = clCreateContext(props, 1, &devices[uiDeviceUsed], NULL, NULL, &error_code);
#else // Win32
            cl_context_properties props[] = 
            {
                CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), 
                CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), 
                CL_CONTEXT_PLATFORM, (cl_context_properties)m_platform, 
                0
            };
            m_context = clCreateContext(props, 1, &m_device, NULL, NULL, &error_code);
#endif // UNIX
#endif // __APPLE__
        }
        else {
            m_device = devices[0];
            cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)m_platform, 0};
            m_context = clCreateContext(props, 1, &devices[dev_selected], NULL, NULL, &error_code);
        }
    }

    m_queue = clCreateCommandQueue(m_context, m_device, 0, &error_code);
}

Device::~Device()
{
    if(m_queue) { clReleaseCommandQueue(m_queue); }
    if(m_context) { clReleaseContext(m_context); }
}

} // namespace gpucl
} // namespace ist
