#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "Message.h"
#include "Fraction.h"
#include "FractionTask.h"
#include "FractionCollider.h"

namespace atomic
{



Task_FractionBeforeDraw::Task_FractionBeforeDraw()
: m_obj(NULL)
{
}

Task_FractionBeforeDraw::~Task_FractionBeforeDraw()
{
    for(size_t i=0; i<m_state_tasks.size(); ++i) {
        AT_DELETE(m_state_tasks[i]);
    }
    m_state_tasks.clear();
}

void Task_FractionBeforeDraw::initialize(FractionSet *obj)
{
    m_obj = obj;
}

void Task_FractionBeforeDraw::exec()
{
    MessageRouter *message_router = atomicGetMessageRouter(MR_FRACTION);
    FractionSet *obj = m_obj;

    // 生成メッセージを処理
    obj->processMessage();

    message_router->unuseAll();

    uint32 num_blocks = obj->getNumBlocks();
    // 衝突器とタスク数をブロックサイズに合わせる
    while(m_state_tasks.size()<num_blocks) {
        m_state_tasks.push_back(AT_NEW(Task_FractionState)());
    }
    for(uint32 i=0; i<num_blocks; ++i) {
        m_state_tasks[i]->initialize(obj, i);
    }
    message_router->resizeMessageBlock(num_blocks);


    // 移動タスクをスケジュール&実行完了待ち
    if(num_blocks > 0) {
        TaskScheduler::schedule((Task**)&m_state_tasks[0], num_blocks);
        TaskScheduler::waitFor((Task**)&m_state_tasks[0], num_blocks);
    }
    message_router->route();


    obj->updateGrid();
    //// 描画後タスクをキック
    //Task_FractionAfterDraw *next = FractionSet::getInterframe()->getTask_AfterDraw();
    //next->initialize(obj);
    //TaskScheduler::schedule(next);
    //TaskScheduler::waitFor(next);
}

void Task_FractionBeforeDraw::waitForComplete()
{
    if(!m_state_tasks.empty()) {
        TaskScheduler::waitFor((Task**)&m_state_tasks[0], m_state_tasks.size());
    }
}



Task_FractionAfterDraw::Task_FractionAfterDraw()
: m_obj(NULL)
{
    m_grid_task = AT_NEW(Task_FractionGrid) ();
}

Task_FractionAfterDraw::~Task_FractionAfterDraw()
{
    AT_DELETE(m_grid_task);
}

void Task_FractionAfterDraw::initialize( FractionSet *obj )
{
    m_obj = obj;
}

void Task_FractionAfterDraw::waitForComplete()
{
    TaskScheduler::waitFor(m_grid_task);
}

void Task_FractionAfterDraw::exec()
{
    // 衝突グリッド更新
    m_grid_task->initialize(m_obj);
    TaskScheduler::schedule(m_grid_task);
    TaskScheduler::waitFor(m_grid_task);
}

} // namespace atomic
