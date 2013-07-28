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

#include<ramen/geo/shape_models/curves_model.hpp>

namespace ramen
{
namespace geo
{

curves_model_t::curves_model_t( unsigned int degree,
                                curve_basis_t basis,
                                curve_periodicity_t wrap) : visitable_t(),
                                                            knots_( arrays::array_t( core::float_k))
{
    assert( degree >= 1);

    degree_ = degree;
    wrap_ = wrap;
    basis_ = basis;
}

std::size_t curves_model_t::num_curves() const
{
    return const_verts_per_curve_array().size();
}

const arrays::array_t& curves_model_t::const_verts_per_curve_array() const
{
    return verts_per_curve_.read();
}

arrays::array_t& curves_model_t::verts_per_curve_array()
{
    return verts_per_curve_.write();
}

const arrays::array_t& curves_model_t::const_knots_array() const
{
    assert( basis_ == nurbs_basis_k);

    return knots_.read();
}

arrays::array_t& curves_model_t::knots_array()
{
    assert( basis_ == nurbs_basis_k);

    return knots_.write();
}

bool curves_model_t::check_consistency( const shape_attributes_t& attrs) const
{
    return true;
}

} // geo
} // ramen
