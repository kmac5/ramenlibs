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

#include"gtest/gtest.h"

#include<ramen/nurbs/basis.hpp>

#include<vector>

#include<boost/assign.hpp>

using namespace ramen::nurbs;

/*
void eval_and_print_basis( int degree,
                            float u,
                            int num_cvs,
                            int span,
                            const const_array_ref_t<float>& knots)
{
    boost::array<float,4> N;
    basis_functions( degree, u, span, knots, array_view_t<float>( N.begin(), N.size()));

    for( int i = 0; i < 4; ++i)
        std::cout << N[i] << " ";

    std::cout << std::endl;
}

void eval_and_print_basis( int degree, float u, int num_cvs, const const_array_ref_t<float>& knots)
{
    int span = find_span( 3, num_cvs, knots, u);
    eval_and_print_basis( degree, u, num_cvs, span, knots);
}
*/

TEST( Basis, All)
{
    std::vector<float> knots;

    // bezier
    knots.clear();
    boost::assign::push_back( knots) =  0,
                                        0,
                                        0,
                                        0,
                                        1,
                                        1,
                                        1,
                                        1;

    //for( int i = 0; i <= 10; ++i)
    //    eval_and_print_basis( 3, i * 0.1f, 4, knots);

    //std::cout << std::endl;

    // clampled
    int num_cvs = 7;
    knots.clear();
    boost::assign::push_back( knots) =  1,
                                        1,
                                        1,
                                        1,
                                        2.45904,
                                        3.60621,
                                        4.03928,
                                        5.5,
                                        5.5,
                                        5.5,
                                        5.5;

    //eval_and_print_basis( 3, 1.0f, num_cvs, knots);
    //eval_and_print_basis( 3, 3.8f, num_cvs, knots);
    //eval_and_print_basis( 3, 5.4f, num_cvs, knots);
    //eval_and_print_basis( 3, 5.5f, num_cvs, knots);

    //std::cout << std::endl;

    // unclampled
    knots.clear();
    boost::assign::push_back( knots) =  0.48698,
                                        0.69014,
                                        1.09027,

                                        1.3,
                                        2.45904,
                                        3.60621,
                                        4.03928,
                                        5.3,

                                        6.48367,
                                        8.19503,
                                        9.04089;

    //eval_and_print_basis( 3, 1.3f, num_cvs, knots);
    //eval_and_print_basis( 3, 3.77f, num_cvs, knots);
    //eval_and_print_basis( 3, 4.17f, num_cvs, knots);
    //eval_and_print_basis( 3, 1.35f, num_cvs, knots);
    //eval_and_print_basis( 3, 1.10f, num_cvs, knots);
    //eval_and_print_basis( 3, 5.3f, num_cvs, knots);
    //std::cout << std::endl;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}

