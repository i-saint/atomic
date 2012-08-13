#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"


int istmain(int argc, char* argv[])
{
    atomic::AtomicApplication app;
    if(app.initialize(argc, argv)) {
        app.mainLoop();
    }
    app.finalize();

#ifdef _DEBUG
    _CrtDumpMemoryLeaks();
#endif // _DEBUG
    return 0;
}

