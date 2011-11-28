#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "Message.h"
#include "Fraction.h"
#include "FractionTask.h"
#include "FractionCollider.h"

namespace atomic {




Task_FractionUpdateAsync::Task_FractionUpdateAsync()
: m_owner(NULL)
{
}

Task_FractionUpdateAsync::~Task_FractionUpdateAsync()
{
}

void Task_FractionUpdateAsync::initialize(FractionSet *obj)
{
    m_owner = obj;
}

void Task_FractionUpdateAsync::exec()
{
    MessageRouter *message_router = atomicGetMessageRouter(MR_FRACTION);
    FractionSet *obj = m_owner;

    obj->updateAsync();

    //message_router->unuseAll();
    //message_router->route();
}

void Task_FractionUpdateAsync::waitForComplete()
{
    TaskScheduler::waitFor(this);
}



} // namespace atomic
