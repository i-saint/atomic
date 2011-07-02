#include "stdafx.h"
#include "../ist/ist.h"
#include "../types.h"
#include "Message.h"
#include "Fraction.h"
#include "FractionTask.h"
#include "FractionCollider.h"

namespace atomic
{



Task_FractionUpdate::Task_FractionUpdate()
{
    m_sortx_task = AT_NEW(Task_FractionSortX) Task_FractionSortX();
    m_sorty_task = AT_NEW(Task_FractionSortY) Task_FractionSortY();
    m_sortz_task = AT_NEW(Task_FractionSortZ) Task_FractionSortZ();
}

Task_FractionUpdate::~Task_FractionUpdate()
{
    for(size_t i=0; i<m_move_tasks.size(); ++i) {
        AT_DELETE(m_move_tasks[i]);
    }
    for(size_t i=0; i<m_col_test_tasks.size(); ++i) {
        AT_DELETE(m_col_test_tasks[i]);
    }
    for(size_t i=0; i<m_col_proc_tasks.size(); ++i) {
        AT_DELETE(m_col_proc_tasks[i]);
    }
    m_move_tasks.clear();
    m_col_test_tasks.clear();
    m_col_proc_tasks.clear();
    AT_DELETE(m_sortx_task);
    AT_DELETE(m_sorty_task);
    AT_DELETE(m_sortz_task);
}

void Task_FractionUpdate::initialize(FractionSet *obj)
{
    m_obj = obj;
}

void Task_FractionUpdate::exec()
{
    TaskScheduler* scheduler = TaskScheduler::getInstance();

    // 生成メッセージを処理
    m_obj->processGenerateMessage();

    uint32 num_blocks = m_obj->getNumBlocks();
    m_blocks = num_blocks;
    // 衝突器とタスク数をブロックサイズに合わせる
    FractionSet::getInterframe()->resizeColliders(num_blocks);
    while(m_move_tasks.size()<num_blocks) {
        m_move_tasks.push_back(AT_NEW(Task_FractionMove) Task_FractionMove());
    }
    while(m_col_test_tasks.size()<num_blocks) {
        m_col_test_tasks.push_back(AT_NEW(Task_FractionCollisionTest) Task_FractionCollisionTest());
    }
    while(m_col_proc_tasks.size()<num_blocks) {
        m_col_proc_tasks.push_back(AT_NEW(Task_FractionCollisionProcess) Task_FractionCollisionProcess());
    }

    // 移動タスクをスケジュール&実行完了待ち
    for(uint32 i=0; i<num_blocks; ++i) {
        m_move_tasks[i]->initialize(m_obj, i);
    }
    scheduler->schedule((Task**)&m_move_tasks[0], num_blocks);
    scheduler->waitFor((Task**)&m_move_tasks[0], num_blocks);

    // ソートタスクをスケジュール&実行完了待ち
    m_sortx_task->initialize(m_obj);
    m_sorty_task->initialize(m_obj);
    m_sortz_task->initialize(m_obj);
    Task *sorts[] ={m_sortx_task, m_sorty_task, m_sortz_task};
    scheduler->schedule(sorts, _countof(sorts));
    scheduler->waitFor(sorts, _countof(sorts));

    // 衝突判定タスクをスケジュール&実行完了待ち
    for(uint32 i=0; i<num_blocks; ++i) {
        m_col_test_tasks[i]->initialize(m_obj, i);
    }
    scheduler->schedule((Task**)&m_col_test_tasks[0], num_blocks);
    scheduler->waitFor((Task**)&m_col_test_tasks[0], num_blocks);

    // 衝突進行タスクをスケジュール&実行完了待ち
    for(uint32 i=0; i<num_blocks; ++i) {
        m_col_proc_tasks[i]->initialize(m_obj, i);
    }
    scheduler->schedule((Task**)&m_col_proc_tasks[0], num_blocks);
    scheduler->waitFor((Task**)&m_col_proc_tasks[0], num_blocks);

}


void Task_FractionUpdate::waitForCompletion()
{
    uint32 num_blocks = m_blocks;
    TaskScheduler* scheduler = TaskScheduler::getInstance();
    scheduler->waitFor((Task**)&m_move_tasks[0], num_blocks);
    scheduler->waitFor((Task**)&m_col_test_tasks[0], num_blocks);
    scheduler->waitFor((Task**)&m_col_proc_tasks[0], num_blocks);

    Task *sorts[] ={m_sortx_task, m_sorty_task, m_sortz_task};
    scheduler->waitFor(sorts, _countof(sorts));
}


} // namespace atomic
