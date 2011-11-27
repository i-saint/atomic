#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "Message.h"
#include "Fraction.h"
#include "FractionTask.h"
#include "FractionCollider.h"

namespace atomic {


class Task_FractionBeforeDraw_Block : public Task
{
private:
    FractionSet *m_owner;
    size_t m_block;
public:
    Task_FractionBeforeDraw_Block() : m_owner(NULL) {}
    void initialize(FractionSet *obj, size_t block) { m_owner=obj; m_block=block; }
    void exec() { m_owner->taskBeforeDraw(m_block); }
    FractionSet* getOwner() { return m_owner; }
};

class Task_FractionSPHDensity : public Task
{
private:
    FractionSet *m_owner;
    size_t m_block;
public:
    Task_FractionSPHDensity() : m_owner(NULL) {}
    void initialize(FractionSet *obj, size_t block) { m_owner=obj; m_block=block; }
    void exec() { m_owner->sphDensity(m_block); }
    FractionSet* getOwner() { return m_owner; }
};





Task_FractionBeforeDraw::Task_FractionBeforeDraw()
: m_owner(NULL)
{
}

Task_FractionBeforeDraw::~Task_FractionBeforeDraw()
{
    for(size_t i=0; i<m_state_tasks.size(); ++i) { IST_DELETE(m_state_tasks[i]); }
    for(size_t i=0; i<m_sph_density_tasks.size(); ++i) { IST_DELETE(m_sph_density_tasks[i]); }
}

void Task_FractionBeforeDraw::initialize(FractionSet *obj)
{
    m_owner = obj;
}

void Task_FractionBeforeDraw::exec()
{
    Task_FractionCopy *task_copy = FractionSet::getInterframe()->getTask_Copy();
    Task_FractionAfterDraw *task_after = FractionSet::getInterframe()->getTask_AfterDraw();

    MessageRouter *message_router = atomicGetMessageRouter(MR_FRACTION);
    FractionSet *obj = m_owner;

    // コピー完了待ち
    task_copy->waitForComplete();

    // 生成メッセージを処理
    obj->taskBeforeDraw();

    message_router->unuseAll();

    // 衝突器とタスク数をブロックサイズに合わせる
    uint32 num_blocks = obj->getNumBlocks();
    //while(m_state_tasks.size()<num_blocks) {
    //    m_state_tasks.push_back(IST_NEW(Task_FractionBeforeDraw_Block)());
    //    m_sph_density_tasks.push_back(IST_NEW(Task_FractionSPHDensity)());
    //}
    //for(uint32 i=0; i<num_blocks; ++i) {
    //    m_state_tasks[i]->initialize(obj, i);
    //    m_sph_density_tasks[i]->initialize(obj, i);
    //}
    message_router->resizeMessageBlock(num_blocks);


    //// 移動タスクをスケジュール&実行完了待ち
    //if(num_blocks > 0) {
    //    task_after->waitForComplete();

    //    TaskScheduler::push((Task**)&m_sph_density_tasks[0], num_blocks);
    //    TaskScheduler::waitFor((Task**)&m_sph_density_tasks[0], num_blocks);

    //    TaskScheduler::push((Task**)&m_state_tasks[0], num_blocks);
    //    TaskScheduler::waitFor((Task**)&m_state_tasks[0], num_blocks);
    //}
    message_router->route();


    //// 描画後タスクをキック
    //task_after->initialize(obj);
    //task_after->kick();
    obj->taskAfterDraw();


    task_copy->initialize(obj, obj->getNext());
    task_copy->kick();
}

void Task_FractionBeforeDraw::waitForComplete()
{
    if(!m_state_tasks.empty()) {
        TaskScheduler::waitFor((Task**)&m_state_tasks[0], m_state_tasks.size());
    }
    TaskScheduler::waitFor(this);
}



Task_FractionAfterDraw::Task_FractionAfterDraw()
: m_owner(NULL)
{
}

Task_FractionAfterDraw::~Task_FractionAfterDraw()
{
}

void Task_FractionAfterDraw::initialize( FractionSet *obj )
{
    m_owner = obj;
}

void Task_FractionAfterDraw::waitForComplete()
{
    TaskScheduler::waitFor(this);
}

void Task_FractionAfterDraw::exec()
{
    // 衝突グリッド更新
    m_owner->taskAfterDraw();
}



void Task_FractionCopy::initialize( const FractionSet *obj, FractionSet *dst )
{
    m_owner = obj;
    m_dst = dst;
}

void Task_FractionCopy::exec()
{
    if(m_owner==m_dst) {
        return;
    }
    m_owner->taskCopy(m_dst);
}

void Task_FractionCopy::waitForComplete()
{
    TaskScheduler::waitFor(this);
}

} // namespace atomic
