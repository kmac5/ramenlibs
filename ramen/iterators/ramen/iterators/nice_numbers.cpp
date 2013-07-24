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

#include<ramen/iterators/nice_numbers.hpp>

#include<cassert>
#include<algorithm>

#include<ramen/math/cmath.hpp>

namespace ramen
{
namespace iterators
{
namespace
{

double nice_num( double x, bool round)
{
    int expv = math::cmath<double>::floor( math::cmath<double>::log10( x));
    double f = x / math::cmath<double>::pow( 10.0, expv); // between 1 and 10
    double nf; // nice, rounded fraction

    if (round)
    {
        if( f < 1.5)
            nf = 1.0;
        else
            if( f < 3.0)
                nf = 2.0;
            else
                if( f < 7.0)
                    nf = 5.0;
                else
                    nf = 10.0;
    }
    else
    {
        if( f <= 1.0)
            nf = 1.0;
        else
            if( f <= 2.0)
                nf = 2.;
            else
                if( f <= 5.)
                    nf = 5.;
                else
                    nf = 10.;
    }

    return nf * math::cmath<double>::pow( 10.0, expv);
}

} // unnamed

nice_numbers_t::nice_numbers_t( double imin, double imax, int num_ticks)
{
    double range = nice_num( imax - imin, false);
    d_ = nice_num( range / ( num_ticks - 1), true);
    double graphmin = math::cmath<double>::floor( imin / d_) * d_;
    double graphmax = math::cmath<double>::ceil( imax / d_) * d_;
    double nfrac = std::max( -math::cmath<double>::floor( math::cmath<double>::log10( d_)), 0.0);
    x_ = graphmin;
    max_ = graphmax + 0.5 * d_;
    end_ = false;
}

nice_numbers_t::nice_numbers_t()
{
    end_ = true;
}

nice_numbers_t& nice_numbers_t::operator++()
{
    assert( !end_);

    x_ += d_;

    if( x_ > max_)
        end_ = true;

    return *this;
}

nice_numbers_t nice_numbers_t::operator++( int)
{
    assert( !end_);

    nice_numbers_t tmp( *this);
    ++tmp;
    return tmp;
}

double nice_numbers_t::operator*() const
{
    assert( !end_);

    return x_;
}

bool nice_numbers_t::operator==( const nice_numbers_t& other) const
{
    if( end_ && other.end_)
        return true;

    if( end_ == false && other.end_ == false)
        return x_ == other.x_;

    return false;
}

bool nice_numbers_t::operator!=( const nice_numbers_t& other) const
{
    return !( *this == other);
}

bool nice_numbers_t::operator<( const nice_numbers_t& other) const
{
    if( end_ && other.end_)
        return false;

    if( end_ == false && other.end_ == false)
        return x_ < other.x_;

    if( end_)
        return true;
    else
        return false;
}

} // iterators
} // ramen
