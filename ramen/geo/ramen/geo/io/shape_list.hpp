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

#ifndef RAMEN_GEO_IO_SHAPE_LIST_HPP
#define RAMEN_GEO_IO_SHAPE_LIST_HPP

#include<ramen/geo/io/shape_list_fwd.hpp>

#include<ramen/geo/config.hpp>

#include<boost/cstdint.hpp>

#include<ramen/core/regex8.hpp>

#include<ramen/math/box3.hpp>

#include<ramen/geo/shape_models/shape_types.hpp>

namespace ramen
{
namespace geo
{
namespace io
{

struct RAMEN_GEO_IO_API shape_entry_t
{
    shape_entry_t( const core::string8_t& shape_name,
                   boost::uint32_t shape_type,
                   const math::box3d_t& shape_bbox = math::box3d_t());

    // gcc requires this, even if the default assignment operator would work. A bug?
    shape_entry_t& operator=( const shape_entry_t& other)
    {
        name = other.name;
        type = other.type;
        return *this;
    }

    const char *type_string() const;

    core::string8_t name;
    boost::uint32_t type;

    // unused for now.
    math::box3d_t bbox;
};

class RAMEN_GEO_IO_API shape_list_filter_t
{
public:

    shape_list_filter_t();
    explicit shape_list_filter_t( const core::regex8_t& regex);
    explicit shape_list_filter_t( boost::uint32_t type_mask);
    shape_list_filter_t( const core::regex8_t& regex, boost::uint32_t type_mask);

    ~shape_list_filter_t();

    boost::uint32_t type_mask() const;

    bool operator()( const shape_entry_t& entry) const;
    bool operator()( const core::string8_t& name, boost::uint32_t type) const;

private:

    bool regex_match( const core::string8_t& name) const;

    // Non copyable.
    shape_list_filter_t( const shape_list_filter_t& other);
    shape_list_filter_t& operator=( const shape_list_filter_t& other);

    struct impl;
    impl *pimpl_;
};

class RAMEN_GEO_IO_API shape_list_t
{
    BOOST_COPYABLE_AND_MOVABLE( shape_list_t)

public:

    shape_list_t();
    ~shape_list_t();

    // Copy constructor
    shape_list_t( const shape_list_t& other);

    // Copy assignment
    shape_list_t& operator=( BOOST_COPY_ASSIGN_REF( shape_list_t) other)
    {
        shape_list_t tmp( other);
        swap( tmp);
        return *this;
    }

    // Move constructor
    shape_list_t( BOOST_RV_REF( shape_list_t) other)
    {
        assert( other.pimpl_);

        init();
        swap( other);
    }

    // Move assignment
    shape_list_t& operator=(BOOST_RV_REF( shape_list_t) other) //Move assignment
    {
        assert( other.pimpl_);

        swap( other);
        return *this;
    }

    void swap( shape_list_t& other);

    void emplace_back( const core::string8_t& shape_name, boost::uint32_t shape_type);
    void push_back( const shape_entry_t& entry);

    typedef const shape_entry_t*  const_iterator;

    const_iterator begin() const;
    const_iterator end() const;

private:

    void init();

    struct impl;
    impl *pimpl_;
};

} // io
} // geo
} // ramen

#endif
