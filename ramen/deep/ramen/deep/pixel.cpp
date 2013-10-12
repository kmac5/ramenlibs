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

#include<ramen/core/global_names.hpp>

#include<ramen/arrays/array.hpp>

namespace ramen
{
namespace deep
{

pixel_t::pixel_t()
{
    data_.insert( core::g_Z_name, core::float_k);
    data_.insert( core::g_A_name, core::color3f_k);
    
    z_data_ref_ = new arrays::array_ref_t<float>( data_.array( core::g_Z_name));
    z_back_data_ref_ = 0;
    a_data_ref_ = new arrays::array_ref_t<color::color3f_t>( data_.array( core::g_A_name));
    c_data_ref_ = 0;
}

pixel_t::~pixel_t()
{
    delete z_data_ref_;
    delete z_back_data_ref_;
    delete a_data_ref_;
    delete c_data_ref_;
}

pixel_t::pixel_t( const pixel_t& other) : data_( other.data_)
{
    z_data_ref_ = new arrays::array_ref_t<float>( data_.array( core::g_Z_name));
    a_data_ref_ = new arrays::array_ref_t<color::color3f_t>( data_.array( core::g_A_name));

    if( data_.has_array( core::g_Zback_name))
        z_back_data_ref_ = new arrays::array_ref_t<float>( data_.array( core::g_Zback_name));
    else
        z_back_data_ref_ = 0;
    
    if( data_.has_array( core::g_C_name))
        c_data_ref_ = new arrays::array_ref_t<color::color3f_t>( data_.array( core::g_C_name));
    else
        c_data_ref_ = 0;
}

void pixel_t::swap( pixel_t& other)
{
    std::swap( data_, other.data_);
    std::swap( z_data_ref_, other.z_data_ref_);
    std::swap( z_back_data_ref_, other.z_back_data_ref_);
    std::swap( a_data_ref_, other.a_data_ref_);
    std::swap( c_data_ref_, other.c_data_ref_);
}

} // deep
} // ramen
