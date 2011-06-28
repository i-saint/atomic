#ifndef __ist_Graphic_Assert__
#define __ist_Graphic_Assert__

//#define IST_ENABLE_GRAPHICS_ASSERT

#ifdef IST_ENABLE_GRAPHICS_ASSERT
    #include "../Base/Assert.h"

    #define CheckGLError() \
    {\
        int e = glGetError();\
        if(e!=GL_NO_ERROR) {\
            IST_ASSERT("%s\n", (const char*)gluErrorString(e));\
        }\
    }
#else
    #define CheckGLError()
#endif

#endif // __ist_Graphic_Assert__

