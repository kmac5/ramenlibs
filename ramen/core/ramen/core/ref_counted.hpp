/*
 Copyright (c) 2013 Esteban Tovagliari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef RAMEN_CORE_REF_COUNTED_HPP
#define RAMEN_CORE_REF_COUNTED_HPP

#include<ramen/core/config.hpp>

#include<boost/smart_ptr/detail/atomic_count.hpp>

namespace ramen
{
namespace core
{


class RAMEN_CORE_API ref_counted_t
{
public:

    typedef size_t ref_count_type;

    ref_counted_t();

    inline void add_ref() const
    {
        ++num_refs_;
    }

    inline void remove_ref() const
    {
        if( --num_refs_)
            delete this;
    }

    long ref_count() const
    {
        return num_refs_;
    }

protected:

    virtual ~ref_counted_t();

    // non-copyable
    ref_counted_t( const ref_counted_t&);
    ref_counted_t& operator=( const ref_counted_t&);

private:

    mutable boost::detail::atomic_count num_refs_;
};

inline void intrusive_ptr_add_ref( const ref_counted_t *r) { r->add_ref();}
inline void intrusive_ptr_release( const ref_counted_t *r) { r->remove_ref();}

} // core
} // ramen

#endif
