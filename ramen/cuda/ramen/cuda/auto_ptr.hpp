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

#ifndef RAMEN_CUDA_AUTO_PTR_HPP
#define RAMEN_CUDA_AUTO_PTR_HPP

#include<ramen/cuda/config.hpp>

#include<ramen/cuda/cudart.hpp>

#include<boost/move/move.hpp>
#include<boost/swap.hpp>

namespace ramen
{
namespace cuda
{

struct device_ptr_policy
{
    template<class T>
    static void delete_ptr( T *ptr)
    {
        assert( ptr);

        cuda_free( ptr);
    }

    template<class T>
    static void get_device_ptr( T *x, unsigned int flags)
    {
        return x;
    }
};

struct host_ptr_policy
{
    template<class T>
    static void delete_ptr( T *ptr)
    {
        assert( ptr);

        cuda_free_host( ptr);
    }

    template<class T>
    static void get_device_ptr( T *x, unsigned int flags)
    {
        assert( x);

        return cuda_host_get_device_ptr( x, flags);
    }
};

template<class T, class P>
class auto_ptr_t
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE( auto_ptr_t)

    // for safe bool
    operator int() const;

public:

    typedef T        element_type;
    typedef T*       pointer_type;
    typedef const T* const_pointer_type;
    typedef P        policy_type;

    explicit auto_ptr_t( T *ptr = 0) : ptr_( ptr) {}

    ~auto_ptr_t()
    {
        delete_contents();
    }

    auto_ptr_t( BOOST_RV_REF( auto_ptr_t) other) throw()  : ptr_( other.release())
    {
    }

    auto_ptr_t& operator=( BOOST_RV_REF( auto_ptr_t) other)
    {
        swap( other);
        return *this;
    }

    pointer_type get()
    {
        return ptr_;
    }

    pointer_type device_ptr( unsigned int flags = 0)
    {
        assert( ptr_);

        return policy_type::get_device_ptr( ptr_, flags);
    }

    pointer_type release()
    {
        pointer_type tmp = ptr_;
        ptr_ = 0;
        return tmp;
    }

    void reset( pointer_type ptr)
    {
        if( ptr_ != ptr)
        {
            delete_contents();
            ptr_ = ptr;
        }
    }

    void swap( auto_ptr_t<T,P>& other)
    {
        boost::swap( ptr_, other.ptr_);
    }

    // safe bool conversion ( private int conversion prevents unsafe use)
    operator bool() const throw() { return ptr_ != 0;}
    bool operator!() const throw();

private:

    void delete_contents()
    {
        if( ptr_)
            policy_type::delete_ptr( ptr_);

        ptr_ = 0;
    }

    pointer_type ptr_;
};

template<class T, class P>
inline void swap( auto_ptr_t<T,P>& x, auto_ptr_t<T,P>& y)
{
    x.swap( y);
}

} // cuda
} // ramen

#endif
