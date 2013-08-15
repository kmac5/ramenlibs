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

#include<ramen/containers/variant.hpp>

namespace ramen
{
namespace containers
{

template<class T>
struct variant_t::vtable_impl
{
    static core::type_t type()
    {
        return core::type_traits<T>::type();
    }

    static void destroy( const variant_t& v)
    {
        const T& data = *reinterpret_cast<const T*>( v.storage());
        data.~T();
    }

    static void clone( const variant_t& src, variant_t& dst)
    {
        new( dst.storage()) T( *reinterpret_cast<const T*>( src.storage()));
        dst.vptr_ = src.vptr_;
    }

    static bool equals( const variant_t& a, const variant_t& b)
    {
        const T& a_data = *reinterpret_cast<const T*>( a.storage());
        const T& b_data = *reinterpret_cast<const T*>( b.storage());
        return a_data == b_data;
    }
};

// Needs special handling, as core::string8_t copy constructor can throw.
template<>
struct variant_t::vtable_impl<core::string8_t>
{
    static core::type_t type()
    {
        return core::type_traits<core::string8_t>::type();
    }

    static void destroy(const variant_t& v)
    {
        const core::string8_t& data = *reinterpret_cast<const core::string8_t*>( v.storage());
        data.~string8_t();
    }

    static void clone(const variant_t& src, variant_t& dst)
    {
        try
        {
            // this can throw, leave dst untouched if it does.
            core::string8_t tmp( *reinterpret_cast<const core::string8_t*>( src.storage()));

            // From here, nothing can throw.
            new( dst.storage()) core::string8_t();
            core::string8_t& dst_str( *reinterpret_cast<core::string8_t*>( dst.storage()));
            dst_str = boost::move( tmp);
            dst.vptr_ = src.vptr_;
        }
        catch( ...)
        {
            throw;
        }
    }

    static bool equals( const variant_t& a, const variant_t& b)
    {
        const core::string8_t& a_data = *reinterpret_cast<const core::string8_t*>( a.storage());
        const core::string8_t& b_data = *reinterpret_cast<const core::string8_t*>( b.storage());
        return a_data == b_data;
    }
};

variant_t::variant_t()                              { init<bool>( false);}
variant_t::variant_t( bool x)                       { init<bool>( x);}
variant_t::variant_t( boost::int32_t x)             { init<boost::int32_t>( x);}
variant_t::variant_t( boost::uint32_t x)            { init<boost::uint32_t>( x);}
variant_t::variant_t( float x)                      { init<float>( x);}
variant_t::variant_t( const math::point2i_t& x)     { init<math::point2i_t>( x);}
variant_t::variant_t( const math::point2f_t& x)     { init<math::point2f_t>( x);}
variant_t::variant_t( const math::point3f_t& x)     { init<math::point3f_t>( x);}
variant_t::variant_t( const math::hpoint3f_t& x)    { init<math::hpoint3f_t>( x);}
variant_t::variant_t( const core::string8_t& x)     { init<core::string8_t>( x);}
variant_t::variant_t( const char *x)                { init<core::string8_t>( x);}
variant_t::variant_t( char *x)                      { init<core::string8_t>( x);}
variant_t::variant_t( const math::box2i_t& x)       { init<math::box2i_t>( x);}

variant_t::variant_t( const variant_t& other)
{
    other.vptr_->clone( other, *this);
}

variant_t::~variant_t()
{
    vptr_->destroy( *this);
}

variant_t& variant_t::operator=( const variant_t& other)
{
    vptr_->destroy( *this);
    other.vptr_->clone( other, *this);
    return *this;
}

core::type_t variant_t::type() const
{
    return vptr_->type();
}

template<class T>
void variant_t::init( const T& x)
{
    new( storage()) T( x);

    static vtable vtbl =
    {
        &vtable_impl<T>::type,
        &vtable_impl<T>::destroy,
        &vtable_impl<T>::clone,
        &vtable_impl<T>::equals
    };

    vptr_ = &vtbl;
}

bool variant_t::operator==( const variant_t& other) const
{
    return type() == other.type() && vptr_->equals( *this, other);
}

bool variant_t::operator!=( const variant_t& other) const
{
    return !( *this == other);
}

const unsigned char *variant_t::storage() const
{
    return &storage_[0];
}

unsigned char *variant_t::storage()
{
    return &storage_[0];
}

std::ostream& operator<<( std::ostream& os, const variant_t& x)
{
    switch( x.type())
    {
        case core::bool_k:
            os << get<bool>( x);
        break;

        case core::int32_k:
            os << get<boost::int32_t>( x);
        break;

        case core::uint32_k:
            os << get<boost::uint32_t>( x);
        break;

        case core::float_k:
            os << get<float>( x);
        break;

        case core::string8_k:
            os << get<core::string8_t>( x);
        break;

        //case point3f_k:
        //case point3hf_k:
        default:
            throw core::not_implemented();
    }

    return os;
}

} // core
} // ramen
