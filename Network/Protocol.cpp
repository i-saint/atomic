#include "stdafx.h"
#include "Protocol.h"

namespace atomic {


void PMessage::destroy()
{
    switch(type) {
    case PM_Sync:
        break;

    default:
        break;
    }
}

} // namespace atomic
