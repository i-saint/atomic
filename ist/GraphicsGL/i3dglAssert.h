#ifndef __ist_i3dgl_Assert__
#define __ist_i3dgl_Assert__

#ifdef __i3d_enable_assert__
    #include "../Base/Assert.h"

    #define istCheckGLError() \
    {\
        int e = glGetError();\
        if(e!=GL_NO_ERROR) {\
            IST_ASSERT("%s\n", (const char*)gluErrorString(e));\
        }\
    }
#else
    #define istCheckGLError()
#endif // __ist_enable_graphics_assert__

#endif // __ist_i3dgl_Assert__
