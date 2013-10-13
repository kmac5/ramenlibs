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

#ifndef RAMEN_ARRAYS_ARRAY_REF_HPP
#define RAMEN_ARRAYS_ARRAY_REF_HPP

#include<ramen/arrays/config.hpp>

#include<assert.h>
#include<iterator>

#include<boost/container/vector.hpp>
#include<boost/range.hpp>

#include<ramen/core/exceptions.hpp>
#include<ramen/core/stl_allocator_adapter.hpp>

#include<ramen/containers/string8_vector.hpp>

#include<ramen/arrays/array.hpp>
#include<ramen/arrays/detail/array_models.hpp>
#include<ramen/arrays/detail/string_array_model.hpp>

namespace ramen
{
namespace arrays
{

template<class T>
struct array_ref_traits
{
    typedef core::stl_allocator_adapter_t<T> allocator_type;
    typedef detail::array_model_t<boost::container::vector<T, allocator_type> > model_type;

    typedef const T&    const_reference;
    typedef T&          reference;
    typedef const T*    const_iterator;
    typedef T*          iterator;

    static const_iterator get_begin( const array_t& array)
    {
        const model_type *model = static_cast<const model_type*>( array.model());
        return model->items_.begin().get_ptr();
    }

    static const_iterator get_end( const array_t& array)
    {
        const model_type *model = static_cast<const model_type*>( array.model());
        return model->items_.end().get_ptr();
    }

    static iterator get_begin( array_t& array)
    {
        model_type *model = static_cast<model_type*>( array.model());
        return model->items_.begin().get_ptr();
    }

    static iterator get_end( array_t& array)
    {
        model_type *model = static_cast<model_type*>( array.model());
        return model->items_.end().get_ptr();
    }
};

template<>
struct array_ref_traits<core::string8_t>
{
    typedef detail::string_array_model_t model_type;

    typedef const core::string8_t&                        const_reference;
    typedef containers::string8_vector_t::string_proxy    reference;
    typedef containers::string8_vector_t::const_iterator  const_iterator;
    typedef containers::string8_vector_t::iterator        iterator;

    static const_iterator get_begin( const array_t& array)
    {
        const model_type *model = static_cast<const model_type*>( array.model());
        return model->items_.begin();
    }

    static const_iterator get_end( const array_t& array)
    {
        const model_type *model = static_cast<const model_type*>( array.model());
        return model->items_.end();
    }

    static iterator get_begin( array_t& array)
    {
        model_type *model = static_cast<model_type*>( array.model());
        return model->items_.begin();
    }

    static iterator get_end( array_t& array)
    {
        model_type *model = static_cast<model_type*>( array.model());
        return model->items_.end();
    }
};

/*!
\ingroup arrays
\brief Constant reference to an array.
*/
template<class T>
class const_array_ref_t
{
    // for safe bool
    operator int() const;
    
public:

    typedef T                                               value_type;
    typedef typename array_ref_traits<T>::const_reference   const_reference;
    typedef typename array_ref_traits<T>::reference         reference;
    typedef typename array_ref_traits<T>::const_iterator    const_iterator;
    typedef typename array_ref_traits<T>::iterator          iterator;
    typedef typename array_t::size_type                     size_type;

    const_array_ref_t() : array_( 0) {}
    
    explicit const_array_ref_t( const array_t& array) : array_( const_cast<array_t*>( &array))
    {
        // preconditions
        assert( core::type_traits<value_type>::type() == array.type());

        if( core::type_traits<value_type>::type() != array.type())
            throw core::bad_type_cast( array.type(), core::type_traits<value_type>::type());
    }

    core::type_t type() const
    {
        assert( array_);
        
        return array_->type();
    }
    
    bool empty() const
    {
        assert( array_);
        
        return array_->empty();
    }

    size_type size() const
    {
        assert( array_);
        
        return array_->size();
    }

    size_type capacity() const
    {
        assert( array_);
        
        return array_->capacity();
    }

    const_iterator begin() const
    {
        assert( array_);
        
        return array_ref_traits<T>::get_begin( *array_);
    }

    const_iterator end() const
    {
        assert( array_);
        
        return array_ref_traits<T>::get_end( *array_);
    }

    const_reference operator[]( size_type i) const
    {
        assert( i < size());

        return begin()[i];
    }

    // safe bool conversion ( private int conversion prevents unsafe use)
    operator bool() const throw() { return array_;}
    bool operator!() const throw();
    
protected:

    array_t *array_;
};

/*!
\ingroup arrays
\brief Non-constant reference to an array.
*/
template<class T>
class array_ref_t : public const_array_ref_t<T>
{
    // for safe bool
    operator int() const;
    
public:

    typedef T                                               value_type;
    typedef typename array_ref_traits<T>::const_reference   const_reference;
    typedef typename array_ref_traits<T>::reference         reference;
    typedef typename array_ref_traits<T>::const_iterator    const_iterator;
    typedef typename array_ref_traits<T>::iterator          iterator;
    typedef typename array_t::size_type                     size_type;

    array_ref_t() : const_array_ref_t<value_type>() {}
    
    explicit array_ref_t( array_t& array) : const_array_ref_t<value_type>( array) {}

    void clear()
    {
        assert( this->array_);
        
        return this->array_->clear();
    }

    void reserve( size_type n)
    {
        assert( this->array_);
                
        this->array_->reserve( n);
    }

    void shrink_to_fit()
    {
        assert( this->array_);
        
        this->array_->shrink_to_fit();
    }

    const_iterator begin() const
    {
        assert( this->array_);

        return array_ref_traits<T>::get_begin( *( this->array_));
    }

    const_iterator end() const
    {
        return array_ref_traits<T>::get_end( *( this->array_));
    }

    iterator begin()
    {
        return array_ref_traits<T>::get_begin( *( this->array_));
    }

    iterator end()
    {
        return array_ref_traits<T>::get_end( *( this->array_));
    }

    const_reference operator[]( size_type i) const
    {
        assert( i < this->size());

        return begin()[i];
    }
    
    reference operator[]( size_type i)
    {
        assert( i < this->size());

        return begin()[i];
    }

    void fill( const_reference x)
    {
        for( int i = 0, e = this->size(); i < e; ++i)
            (*this)[i] = x;
    }

    void push_back( const_reference x)
    {
        assert( this->array_);
        
        this->array_->push_back( reinterpret_cast<const void*>( &x));
    }

    void resize( size_type n)
    {
        assert( this->array_);
        
        this->array_->resize( n);
    }

    void erase( size_type start, size_type end)
    {
        assert( this->array_);
        
        this->array_->erase( start, end);
    }

    void erase( size_type pos)
    {
        assert( this->array_);
        
        this->array_->erase( pos);
    }
    
    // safe bool conversion ( private int conversion prevents unsafe use)
    operator bool() const throw() { return this->array_;}
    bool operator!() const throw();
};

// for boost range...
template<class T>
inline typename const_array_ref_t<T>::const_iterator range_begin( const_array_ref_t<T>& x )
{
    return x.begin();
}

template<class T>
inline typename const_array_ref_t<T>::const_iterator range_end( const_array_ref_t<T>& x )
{
    return x.end();
}

template<class T>
inline typename const_array_ref_t<T>::const_iterator range_begin( const const_array_ref_t<T>& x )
{
    return x.begin();
}

template<class T>
inline typename const_array_ref_t<T>::const_iterator range_end( const const_array_ref_t<T>& x )
{
    return x.end();
}

} // arrays
} // ramen

// boost range interop
namespace boost
{

template<class T>
struct range_mutable_iterator<ramen::arrays::const_array_ref_t<T> >
{
    typedef typename ramen::arrays::const_array_ref_t<T>::const_iterator type;
};

template<class T>
struct range_const_iterator<ramen::arrays::const_array_ref_t<T> >
{
    typedef typename ramen::arrays::const_array_ref_t<T>::const_iterator type;
};

} // boost

#endif
