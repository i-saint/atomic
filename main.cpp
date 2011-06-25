#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"


int istmain(int argc, char* argv[])
{
    atomic::AtomicApplication app;
    //if(app.Initialize(2000, 100, 1024, 768, L"atomic", false)) {
    if(app.Initialize(2000, 100, 640, 480, L"atomic", false)) {
        app.mainLoop();
        app.Finalize();
    }

    return 0;
}

