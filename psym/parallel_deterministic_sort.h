
#ifndef __TBB_parallel_stable_sort_H
#define __TBB_parallel_stable_sort_H

#include <tbb/tbb.h>

namespace detail {

    template<class Iterator, class Compare>
    class parallel_deterministic_sort_body
    {
    public:
        parallel_deterministic_sort_body(Iterator i_first, Iterator i_last, const Compare *i_compare, int i_depth)
            : m_first(i_first)
            , m_last(i_last)
            , m_compare(i_compare)
            , m_depth(i_depth)
        {}

        void operator()() const
        {
            // 要素数が cutoff_elem 未満、もしくは再帰の深度 が cutoff_depth 以上の場合普通に std::sort
            // そうでない場合 2 分割して並列に部分ソート
            const size_t cutoff_elem = 30;
            const size_t cutoff_depth = 7;

            if( m_last > m_first ) {
                size_t dist = std::distance(m_first, m_last);
                if(dist < cutoff_elem || m_depth>=cutoff_depth) { 
                    std::stable_sort(m_first, m_last, *m_compare);
                }
                else {
                    Iterator mid = m_first;
                    std::advance(mid, dist/2);
                    std::nth_element(m_first, mid, m_last, *m_compare);
                    tbb::parallel_invoke(
                        parallel_deterministic_sort_body(m_first, mid, m_compare, m_depth+1),
                        parallel_deterministic_sort_body(mid, m_last, m_compare, m_depth+1) );
                }
            }
        }

    private:
        Iterator m_first;
        Iterator m_last;
        const Compare *m_compare;
        int m_depth;
    };

} // namespace detail

template<class Iterator, class Compare>
inline void parallel_deterministic_sort(Iterator i_first, Iterator i_last, const Compare &i_compare)
{
    detail::parallel_deterministic_sort_body<Iterator, Compare> body(i_first, i_last, &i_compare, 0);
    body();
}

template<class Iterator>
inline void parallel_deterministic_sort(Iterator i_first, Iterator i_last)
{
    typedef std::less< typename LmGetValueTypeFromIterator<Iterator>::Type > Compare;
    parallel_deterministic_sort<Iterator, Compare>(i_first, i_last, Compare());
}
#endif

