#ifndef __ist_Concurrency_ParallelAlgorithm_h__
#define __ist_Concurrency_ParallelAlgorithm_h__

namespace ist {

template<class Index, class Body>
class ParallelForTask : public Task
{
public:
    ~ParallelForTask()
    {
        wait();
    }

    void start(Index first, Index last, const Body &body)
    {
        wait();
        m_first = first;
        m_last = last;
        m_body = &body;
        TaskScheduler::getInstance()->enqueue(this);
    }

    void exec()
    {
        (*m_body)(m_first, m_last);
    }

private:
    Index m_first;
    Index m_last;
    const Body *m_body;
};

template<class Index, class Step, class Body>
inline void parallel_for(Index first, Index last, Step step, const Body &body)
{
    typedef ParallelForTask<Index, Body> Task;
    Task tasks[128];
    int32 ti = 0;
    for(Index i=first; i<last; i+=step) {
        tasks[ti].start(i, std::min<Index>(i+step, last), body); // start() の最初で wait するので ti が一周しても大丈夫なはず
        ti = (ti+1)%_countof(tasks);
    }
    // scope 抜ける時デストラクタで wait
}

template<class Index, class Body>
inline void parallel_for(Index first, Index last, const Body &body)
{
    parallel_for<Index, int32, Body>(first, last, 1, body);
}

} // namespace ist

#endif // __ist_Concurrency_ParallelAlgorithm_h__
