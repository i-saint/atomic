#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"


int istmain(int argc, char* argv[])
{
    atomic::AtomicApplication app;
    atomic::uint32 window_x = 100;
#ifdef ATOMIC_ENABLE_DEBUG_FEATURE
    window_x = 2000;
#endif // ATOMIC_ENABLE_DEBUG_FEATURE
    if(app.initialize(window_x, 100, 1024, 768, L"atomic", false)) {
    //if(app.initialize(window_x, 100, 640, 480, L"atomic", false)) {
        app.mainLoop();
        app.finalize();
    }

    return 0;
}

