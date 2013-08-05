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

#ifndef RAMEN_CORE_ALLOCATOR_INTERFACE_HPP
#define RAMEN_CORE_ALLOCATOR_INTERFACE_HPP

#include<ramen/core/ref_counted.hpp>

#include<cstddef>

#include<boost/intrusive_ptr.hpp>

namespace ramen
{
namespace core
{

/*!
\ingroup core
\brief base class for polymorphic allocators.
*/
class RAMEN_CORE_API allocator_interface_t : public ref_counted_t
{
public:

    virtual ~allocator_interface_t();

    /// allocates a block of memory of size \param size bytes.
    virtual void *allocate( std::size_t size) = 0;

    /// deallocates a block of memory.
    virtual void deallocate( void *ptr) = 0;

protected:

    allocator_interface_t();

private:

    /// delete this object.
    virtual void release() const = 0;
};

typedef boost::intrusive_ptr<allocator_interface_t> allocator_ptr_t;

} // core
} // ramen

#endif
