#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"


int istmain(int argc, char* argv[])
{
    atomic::AtomicApplication app;
    if(app.initialize()) {
    //if(app.initialize(window_x, 100, 640, 480, L"atomic", false)) {
        app.mainLoop();
        app.finalize();
    }

#ifdef _DEBUG
    _CrtDumpMemoryLeaks();
#endif // _DEBUG
    return 0;
}

