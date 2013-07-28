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

#ifndef RAMEN_COLOR_SPECTRAL_TO_XYZ_HPP
#define RAMEN_COLOR_SPECTRAL_TO_XYZ_HPP

#include<ramen/color/config.hpp>

#include<cassert>

#include<boost/range/size.hpp>

#include<ramen/color/Concepts/CieXYZFit.hpp>

#include<ramen/color/color3.hpp>

#include<ramen/range/value_type.hpp>

namespace ramen
{
namespace color
{
namespace detail
{

/*
template<class ConstIterator>
struct trapezoidal_integrate
{
    typedef typename std::iterator_traits<ConstIterator>::value_type value_type;

    explicit trapezoidal_integrate( ConstIterator end) : e( end)
    {
        // result( 0);
    }

    void operator()( ConstIterator it)
    {
        ConstIterator next( it + 1);

        if( next != e)
        {
            // ...
        }
    }

    ConstIterator e;
};
*/

} // detail

template<class CieXYZCurves, class T, class ConstRandomAccessRange>
color3_t<typename range::value_type<ConstRandomAccessRange>::type, xyz_t>
spectral_to_xyz( const CieXYZCurves& cie_xyz_curves,
                 T min_wavelength, T max_wavelenght,
                 const ConstRandomAccessRange& values)
{
    BOOST_CONCEPT_ASSERT(( CieXYZFit<CieXYZCurves>));

    typedef typename boost::range_iterator<const ConstRandomAccessRange>::type iterator_type;
    typedef typename range::value_type<ConstRandomAccessRange>::type real_type;

    std::size_t num_samples = boost::size( values);
    real_type w = min_wavelength;
    real_type w_step = max_wavelenght - min_wavelength / static_cast<real_type>( num_samples);

    color3_t<real_type, xyz_t> result( 0);

    for( iterator_type it( boost::begin( values)), e( boost::end( values)); it != e; ++it)
    {
        iterator_type next( it + 1);

        if( next != e)
        {
            real_type avg(( *it + *next) * real_type( 0.5));
            result.x += cie_xyz_curves.X( w) * avg;
            result.y += cie_xyz_curves.Y( w) * avg;
            result.z += cie_xyz_curves.Z( w) * avg;
        }

        w += w_step;
    }

    return result;
}

} // color
} // ramen

#endif
