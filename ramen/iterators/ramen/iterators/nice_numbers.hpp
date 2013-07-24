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

#ifndef RAMEN_ITERATORS_NICE_NUMBERS_HPP
#define	RAMEN_ITERATORS_NICE_NUMBERS_HPP

#include<ramen/iterators/config.hpp>

namespace ramen
{
namespace iterators
{

class RAMEN_ITERATORS_API nice_numbers_t
{
public:

    // begin iterator
    nice_numbers_t( double imin, double imax, int num_ticks);

    // end iterator
    nice_numbers_t();

    nice_numbers_t& operator++();
    nice_numbers_t operator++( int);

    double operator*() const;

    bool operator==( const nice_numbers_t& other) const;
    bool operator!=( const nice_numbers_t& other) const;
    bool operator<( const nice_numbers_t& other) const;

private:

    double x_;
    double d_;
    double max_;
    bool end_;
};

} // iterators
} // ramen

#endif
