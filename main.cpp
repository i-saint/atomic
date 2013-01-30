#include "stdafx.h"
#include "types.h"
#include "Game/AtomicApplication.h"
#include "FunctionID.h"

int istmain(int argc, char* argv[])
{
    //atomic::FunctionIDEachPair([](const ist::EnumStr &es){
    //    istPrint("%d:%s\n", es.num, es.str);
    //});

    atomic::AtomicApplication app;
    if(app.initialize(argc, argv)) {
        app.mainLoop();
    }
    app.finalize();
    return 0;
}

