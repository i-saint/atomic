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
    m_grid_task = AT_NEW(Task_FractionGrid) ();
}

Task_FractionUpdate::~Task_FractionUpdate()
{
    for(size_t i=0; i<m_move_tasks.size(); ++i) {
        AT_DELETE(m_move_tasks[i]);
    }
    m_move_tasks.clear();
    for(size_t i=0; i<m_col_test_tasks.size(); ++i) {
        AT_DELETE(m_col_test_tasks[i]);
    }
    m_col_test_tasks.clear();
    for(size_t i=0; i<m_col_proc_tasks.size(); ++i) {
        AT_DELETE(m_col_proc_tasks[i]);
    }
    m_col_proc_tasks.clear();
    AT_DELETE(m_grid_task);
}

void Task_FractionUpdate::initialize(FractionSet *obj)
{
    m_obj = obj;
}

void Task_FractionUpdate::exec()
{
    // 生成メッセージを処理
    m_obj->processGenerateMessage();

    uint32 num_blocks = m_obj->getNumBlocks();
    m_blocks = num_blocks;
    // 衝突器とタスク数をブロックサイズに合わせる
    while(m_move_tasks.size()<num_blocks) {
        m_move_tasks.push_back(AT_NEW(Task_FractionMove) ());
    }
    while(m_col_test_tasks.size()<num_blocks) {
        m_col_test_tasks.push_back(AT_NEW(Task_FractionCollisionTest) ());
    }
    while(m_col_proc_tasks.size()<num_blocks) {
        m_col_proc_tasks.push_back(AT_NEW(Task_FractionCollisionProcess) ());
    }

    // 移動タスクをスケジュール&実行完了待ち
    for(uint32 i=0; i<num_blocks; ++i) {
        m_move_tasks[i]->initialize(m_obj, i);
    }
    TaskScheduler::schedule((Task**)&m_move_tasks[0], num_blocks);
    TaskScheduler::waitFor((Task**)&m_move_tasks[0], num_blocks);

    m_grid_task->initialize(m_obj);
    TaskScheduler::schedule(m_grid_task);
    TaskScheduler::waitFor(m_grid_task);

    // 衝突判定タスクをスケジュール&実行完了待ち
    for(uint32 i=0; i<num_blocks; ++i) {
        m_col_test_tasks[i]->initialize(m_obj, i);
    }
    TaskScheduler::schedule((Task**)&m_col_test_tasks[0], num_blocks);
    TaskScheduler::waitFor((Task**)&m_col_test_tasks[0], num_blocks);

    // 衝突進行タスクをスケジュール&実行完了待ち
    for(uint32 i=0; i<num_blocks; ++i) {
        m_col_proc_tasks[i]->initialize(m_obj, i);
    }
    TaskScheduler::schedule((Task**)&m_col_proc_tasks[0], num_blocks);
    TaskScheduler::waitFor((Task**)&m_col_proc_tasks[0], num_blocks);

}


void Task_FractionUpdate::waitForCompletion()
{
    uint32 num_blocks = m_blocks;
    TaskScheduler::waitFor((Task**)&m_move_tasks[0], num_blocks);
    TaskScheduler::waitFor((Task**)&m_col_test_tasks[0], num_blocks);
    TaskScheduler::waitFor((Task**)&m_col_proc_tasks[0], num_blocks);
    TaskScheduler::waitFor(m_grid_task);

}


} // namespace atomic
