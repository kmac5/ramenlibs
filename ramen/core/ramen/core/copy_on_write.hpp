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

#ifndef RAMEN_COPY_ON_WRITE_HPP
#define RAMEN_COPY_ON_WRITE_HPP

#include<ramen/core/final.hpp>

#include<cassert>

#include<boost/shared_ptr.hpp>
#include<boost/move/move.hpp>
#include<boost/utility/enable_if.hpp>
#include<boost/type_traits/has_equal_to.hpp>
#include<boost/type_traits/has_not_equal_to.hpp>
#include<boost/swap.hpp>

#include<ramen/core/Concepts/RegularConcept.hpp>

namespace ramen
{
namespace core
{

/*!
\ingroup core
\brief Copy on write holder.
*/
template <typename T> // T models Regular
class copy_on_write_t : RAMEN_CORE_FINAL( copy_on_write_t<T>)
{
    BOOST_CONCEPT_ASSERT(( RegularConcept<T>));

public:

    typedef T value_type;

    /// Default constructor.
    copy_on_write_t()
    {
        object_.reset( new implementation_t(), capture_deleter_t());
    }

    /// Constructor.
    explicit copy_on_write_t( const T& x)
    {
        object_.reset( new implementation_t( x), capture_deleter_t());
    }

    explicit copy_on_write_t( BOOST_RV_REF( T) x)
    {
        object_.reset( new implementation_t( x), capture_deleter_t());
    }

    const value_type& read() const
    {
        assert( object_);

        return object_->value_;
    }

    operator const value_type& () const     { return read();}
    const value_type& operator*() const     { return read();}
    const value_type* operator->() const    { return &read();}

    value_type& write()
    {
        assert( object_);

        if( !unique_instance())
            object_.reset( new implementation_t( object_->value_), capture_deleter_t());

        return object_->value_;
    }

    bool unique_instance() const
    {
        return object_ && object_.use_count() == 1;
    }

    void swap( copy_on_write_t& other)
    {
        boost::swap( object_, other.object_);
    }

private:

    struct implementation_t
    {
        implementation_t() {}

        explicit implementation_t( const value_type& x) : value_( x) {}

        explicit implementation_t( BOOST_RV_REF( value_type) x) : value_( boost::move( x)) {}

        value_type value_;
    };

    struct capture_deleter_t
    {
        void operator()( implementation_t *x) const { delete x;}
    };

    typedef boost::shared_ptr<implementation_t> implementation_ptr_t;

    implementation_ptr_t object_;
};

template<class T>
inline void swap( copy_on_write_t<T>& x, copy_on_write_t<T>& y)
{
    x.swap( y);
}

template<class T>
typename boost::enable_if<boost::has_equal_to<T>, bool>::type
operator==( const copy_on_write_t<T>& a, const copy_on_write_t<T>& b)
{
    return a.read() == b.read();
}

template<class T>
typename boost::enable_if<boost::has_not_equal_to<T>, bool>::type
operator!=( const copy_on_write_t<T>& a, const copy_on_write_t<T>& b)
{
    return a.read() != b.read();
}

} // core
} // ramen

#endif
