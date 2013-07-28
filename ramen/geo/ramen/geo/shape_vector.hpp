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

#ifndef RAMEN_GEO_SHAPE_VECTOR_HPP
#define RAMEN_GEO_SHAPE_VECTOR_HPP

#include<ramen/geo/config.hpp>

#include<ramen/geo/shape_vector_fwd.hpp>

#include<ramen/geo/shape.hpp>

namespace ramen
{
namespace geo
{

class RAMEN_GEO_API shape_vector_t
{
    BOOST_COPYABLE_AND_MOVABLE( shape_vector_t)

public:

    shape_vector_t();
    ~shape_vector_t();

    // Copy constructor
    shape_vector_t( const shape_vector_t& other);

    // Copy assignment
    shape_vector_t& operator=( BOOST_COPY_ASSIGN_REF( shape_vector_t) other)
    {
        shape_vector_t tmp( other);
        swap( tmp);
        return *this;
    }

    // Move constructor
    shape_vector_t( BOOST_RV_REF( shape_vector_t) other)
    {
        assert( other.pimpl_);

        init();
        swap( other);
    }

    // Move assignment
    shape_vector_t& operator=( BOOST_RV_REF( shape_vector_t) other) //Move assignment
    {
        assert( other.pimpl_);

        swap( other);
        return *this;
    }

    void swap( shape_vector_t& other);

    bool empty() const;
    std::size_t size() const;

    const shape_t& operator[]( unsigned int index) const;
    shape_t& operator[]( unsigned int index);

    const shape_t& back() const;
    shape_t& back();

    void push_back( const shape_t& shape);
    void push_back( BOOST_RV_REF( shape_t) shape);

    // TODO: add emplace versions here. ( est.)

    void clear();

    typedef const shape_t*  const_iterator;
    typedef shape_t*        iterator;

    const_iterator begin() const;
    const_iterator end() const;

    iterator begin();
    iterator end();

    void remove_invalid_shapes();

private:

    void init();

    struct impl;
    impl *pimpl_;
};

inline void swap( shape_vector_t& x, shape_vector_t& y)
{
    x.swap( y);
}

RAMEN_GEO_API void apply_visitor( const_shape_visitor& v, const shape_vector_t& s);
RAMEN_GEO_API void apply_visitor( shape_visitor& v, shape_vector_t& s);

} // geo
} // ramen

#endif
