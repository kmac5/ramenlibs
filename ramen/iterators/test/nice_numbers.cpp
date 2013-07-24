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

#include<gtest/gtest.h>

#include<stdio.h>
#include<math.h>

#include<ramen/iterators/nice_numbers.hpp>

#include<iostream>

using namespace ramen::iterators;

/*
 * Nice Numbers for Graph Labels
 * by Paul Heckbert
 * from "Graphics Gems", Academic Press, 1990
 */

/*
 * label.c: demonstrate nice graph labeling
 *
 * Paul Heckbert	2 Dec 88
 */

/*
 * nicenum: find a "nice" number approximately equal to x.
 * Round the number if round=1, take ceiling if round=0
 */

double nicenum( double x, int round)
{
    int expv;				/* exponent of x */
    double f;				/* fractional part of x */
    double nf;				/* nice, rounded fraction */

    expv = floor( log10(x));
    f = x/pow(10., expv);		/* between 1 and 10 */
    if (round)
    if (f<1.5) nf = 1.;
    else if (f<3.) nf = 2.;
    else if (f<7.) nf = 5.;
    else nf = 10.;
    else
    if (f<=1.) nf = 1.;
    else if (f<=2.) nf = 2.;
    else if (f<=5.) nf = 5.;
    else nf = 10.;
    return nf*pow(10., expv);
}

void looselabels( double min, double max, int numticks)
{
    char str[6], temp[20];
    int nfrac;
    double d;				/* tick mark spacing */
    double graphmin, graphmax;		/* graph range min and max */
    double range, x;

    /* we expect min!=max */
    range = nicenum(max-min, 0);
    d = nicenum(range/(numticks-1), 1);
    graphmin = floor(min/d)*d;
    graphmax = ceil(max/d)*d;
    nfrac = std::max(-floor(log10(d)), 0.0);	/* # of fractional digits to show */
    //sprintf(str, "%%.%df", nfrac);	/* simplest axis labels */

    //printf("graphmin=%g graphmax=%g increment=%g\n", graphmin, graphmax, d);

    nice_numbers_t it( min, max, numticks), e;

    for( x=graphmin; x<graphmax+.5*d; x+=d)
    {
        //sprintf(temp, str, x);
        //printf("(%s)\n", temp);
    }
}

TEST( NiceNumberIter, All)
{
    int num_ticks = 31;
    nice_numbers_t it( 0.77, 2.456, num_ticks), end;

    int j = 1;
    for( ; it != end; ++it, ++j)
    {
        std::cout << " " << *it;
    }

    //EXPECT_EQ( j, num_ticks + 1);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
