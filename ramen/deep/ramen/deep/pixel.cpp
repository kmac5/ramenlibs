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

#include<ramen/deep/pixel.hpp>

#include<vector>
#include<algorithm>

#include<boost/range/size.hpp>

#include<ramen/core/global_names.hpp>

#include<ramen/arrays/array.hpp>
#include<ramen/arrays/apply_visitor.hpp>

namespace ramen
{
namespace deep
{
namespace
{

struct pair_compare_first
{
    template<class T, class I>
    bool operator()( const std::pair<T, I>& a,
                     const std::pair<T, I>& b) const
    {
        return a.first < b.first;
    }
};

template<class T>
void make_sorted_indices_array( const arrays::const_array_ref_t<T>& array_ref,
                                arrays::array_t& indices)
{
    assert( array_ref.type() == core::float_k);
    
    std::vector<std::pair<float,boost::uint32_t> > data_indx_array;
    data_indx_array.reserve( array_ref.size());

    for( int i = 0, e = array_ref.size(); i < e; ++i)
        data_indx_array.push_back( std::pair<float,boost::uint32_t>( array_ref[i], i));

    std::stable_sort( data_indx_array.begin(), data_indx_array.end(), pair_compare_first());
    
    indices.reserve( array_ref.size());
    arrays::array_ref_t<boost::uint32_t> indx_ref( indices);

    for( int i = 0, e = array_ref.size(); i < e; ++i)
        indx_ref.push_back( data_indx_array[i].second);
}

template<class Range>
struct rearrange_visitor
{
    rearrange_visitor( const arrays::array_t& orig,
                       const Range& ind) : original( orig), indices( ind)
    {
        assert( boost::size( indices) == original.size());        
    }

    template<class T>
    void operator()( arrays::array_ref_t<T>& array)
    {
        arrays::const_array_ref_t<T> orig_ref( original);
        
        for( int i = 0, e = original.size(); i < e; ++i)
            array[i] = orig_ref[indices[i]];
    }
    
    const arrays::array_t& original;
    const Range& indices;
};

template<class Range>
void rearrange_array( arrays::array_t& array, const Range& indices)
{    
    arrays::array_t tmp( array.type(), array.size());
    rearrange_visitor<Range> v( array, indices);
    arrays::apply_visitor( v, tmp);
    tmp.swap( array);
}

} // unnamed

pixel_t::pixel_t( interpretation_k interp)
{
    clear( interp);
}

pixel_t::pixel_t( const pixel_t& other) : data_( other.data_)
{
    update_array_refs();
}

void pixel_t::swap( pixel_t& other)
{
    std::swap( data_, other.data_);
    other.update_array_refs();
    update_array_refs();
}

void pixel_t::insert_channel( const core::name_t& name, core::type_t array_type)
{
    data_.insert( name, array_type);
    update_array_refs();
}

void pixel_t::clear( interpretation_k interp)
{
    data_.clear();

    data_.insert( core::g_Z_name, core::float_k);
    
    if( interp == continuous_k)
        data_.insert( core::g_Zback_name, core::float_k);
    
    data_.insert( core::g_A_name, core::color3f_k);
    
    update_array_refs();
}

void pixel_t::update_array_refs()
{
    assert( has_channel( core::g_Z_name));
    assert( has_channel( core::g_A_name));
    
    z_data_ = arrays::array_ref_t<float>( array( core::g_Z_name));
    a_data_ = arrays::array_ref_t<color::color3f_t>( array( core::g_A_name));
    
    if( has_channel( core::g_Zback_name))
        z_back_data_ = arrays::array_ref_t<float>( array( core::g_Zback_name));
    else
        z_back_data_ = arrays::array_ref_t<float>();

    if( has_channel( core::g_C_name))
        c_data_ = arrays::array_ref_t<color::color3f_t>( array( core::g_C_name));
    else
        c_data_ = arrays::array_ref_t<color::color3f_t>();
}

void pixel_t::push_back_discrete_sample( float z, const color::color3f_t& a)
{
    z_data_.push_back( z);
    
    if( z_back_data_)
        z_back_data_.push_back( z);
    
    a_data_.push_back( a);
}

void pixel_t::push_back_discrete_sample( float z, const color::color3f_t& a, const color::color3f_t& c)
{
    assert( c_data_);
    
    push_back_discrete_sample( z, a);
    c_data_.push_back( c);
}

void pixel_t::sort()
{
    arrays::array_t indices( core::uint32_k);
    indices.reserve( z_data_.size());
    make_sorted_indices_array( z_data_, indices);
    arrays::const_array_ref_t<boost::uint32_t> index_ref( indices);

    for( arrays::named_array_map_t::iterator it( data_.begin()), e( data_.end()); it != e; ++it)
        rearrange_array( it.second(), index_ref);
    
    update_array_refs();
}

std::size_t pixel_t::num_samples() const
{
    return z_data_.size();
}

void pixel_t::flatten( color::color3f_t& c, color::color3f_t& a) const
{
    if( num_samples())
    {
        c = c_data_[0];
        a = a_data_[0];
    }
    else
    {
        c = color::color3f_t( 0.0f);
        a = color::color3f_t( 0.0f);
    }
    
    for( int i = 1, e = num_samples(); i < e; ++i)
    {
        c.x = c.x + c_data_[i].x * (1.0f - a.x);
        c.y = c.y + c_data_[i].y * (1.0f - a.y);
        c.z = c.z + c_data_[i].z * (1.0f - a.z);
        a.x = a.x + a_data_[i].x * (1.0f - a.x);
        a.y = a.y + a_data_[i].y * (1.0f - a.y);
        a.z = a.z + a_data_[i].z * (1.0f - a.z);
    }
}

} // deep
} // ramen
