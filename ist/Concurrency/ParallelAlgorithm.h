#ifndef ist_Concurrency_ParallelAlgorithm_h
#define ist_Concurrency_ParallelAlgorithm_h

#include "ist/Concurrency/TaskScheduler.h"
#include "ist/Concurrency/AsyncFunction.h"

namespace ist {

template<class F1, class F2>
inline void ParallelInvoke(const F1 &f1, const F2 &f2)
{
    AsyncFunction<F1> af1(f1); af1.start();
    AsyncFunction<F2> af2(f2); af2.start();
}

template<class F1, class F2, class F3>
inline void ParallelInvoke(const F1 &f1, const F2 &f2, const F3 &f3)
{
    AsyncFunction<F1> af1(f1); af1.start();
    AsyncFunction<F2> af2(f2); af2.start();
    AsyncFunction<F3> af3(f3); af3.start();
}

template<class F1, class F2, class F3, class F4>
inline void ParallelInvoke(const F1 &f1, const F2 &f2, const F3 &f3, const F4 &f4)
{
    AsyncFunction<F1> af1(f1); af1.start();
    AsyncFunction<F2> af2(f2); af2.start();
    AsyncFunction<F3> af3(f3); af3.start();
    AsyncFunction<F4> af4(f4); af4.start();
}


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
inline void ParallelFor(Index first, Index last, Step step, const Body &body)
{
    typedef ParallelForTask<Index, Body> Task;
    Task tasks[128];
    int32 ti = 0;
    for(Index i=first; i<last; i+=step) {
        tasks[ti].start(i, stl::min<Index>(i+step, last), body); // start() の最初で wait するので ti が一周しても大丈夫なはず
        ti = (ti+1)%_countof(tasks);
    }
    // scope 抜ける時デストラクタで wait
}

template<class Index, class Body>
inline void ParallelFor(Index first, Index last, const Body &body)
{
    parallel_for<Index, int32, Body>(first, last, 1, body);
}



template<class Iterator, class Compare>
class ParallelSortFunc
{
public:
    ParallelSortFunc(Iterator begin, Iterator end, const Compare &compare, int32 depth)
        : m_first(begin)
        , m_last(end)
        , m_compare(&compare)
        , m_depth(depth)
    {}

    void operator()() const
    {
        // 要素数が cutoff_elem 未満、もしくは再帰の深度 が cutoff_depth 以上の場合普通に stl::sort
        // そうでない場合 2 分割して並列に部分ソート
        const size_t cutoff_elem = 30;
        const size_t cutoff_depth = 6;

        if( m_last > m_first ) {
            size_t dist = stl::distance(m_first, m_last);
            if(dist < cutoff_elem || m_depth>=cutoff_depth) { 
                stl::sort(m_first, m_last, *m_compare);
            }
            else {
                Iterator mid = m_first;
                stl::advance(mid, dist/2);
                stl::nth_element(m_first, mid, m_last, *m_compare);
                ParallelInvoke(
                    ParallelSortFunc(m_first, mid, *m_compare, m_depth+1),
                    ParallelSortFunc(mid, m_last, *m_compare, m_depth+1) );
            }
        }
    }

private:
    Iterator m_first;
    Iterator m_last;
    const Compare *m_compare;
    int32 m_depth;
};

template<class Iterator, class Compare>
inline void ParallelSort(Iterator begin, Iterator end, const Compare &comp)
{
    ParallelSortFunc<Iterator, Compare> func(begin, end, comp, 0);
    func();
}

template<class Iterator>
inline void ParallelSort(Iterator begin, Iterator end)
{
    typedef stl::less<typename stl::iterator_traits<Iterator>::value_type> Compare;
    ParallelSortFunc<Iterator, Compare> func(begin, end, Compare(), 0);
    func();
}


} // namespace ist

#endif // ist_Concurrency_ParallelAlgorithm_h
