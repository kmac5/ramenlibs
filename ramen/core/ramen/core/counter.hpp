// From Adobe source libraries. Original license follows.
/*
    Copyright 2005-2007 Adobe Systems Incorporated
    Distributed under the MIT License (see accompanying file LICENSE_1_0_0.txt
    or a copy at http://stlab.adobe.com/licenses.html)
*/

#ifndef RAMEN_CORE_COUNTER_HPP
#define RAMEN_CORE_COUNTER_HPP

#include<ramen/core/config.hpp>

#include<tbb/atomic.h>

namespace ramen
{
namespace core
{

/*!
\brief An atomic counter.
*/
class counter_t
{
public:

    typedef std::size_t value_type;
    typedef tbb::atomic<value_type> atomic_counter_type;

    counter_t() { count_ = 1;}

    explicit counter_t( value_type count)
    {
        count_ = count;
    }

    void increment() { ++count_;}

    bool decrement()
    {
        return --count_ == 0;
    }

    bool is_one() const
    {
        return count_ == 1;
    }

private:

    // non-copyable
    counter_t( const counter_t&);
    counter_t& operator=( const counter_t&);

    atomic_counter_type count_;
};

} // core
} // ramen

#endif
