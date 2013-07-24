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

#ifndef RAMEN_CORE_TEST_MOVABLE_HPP
#define RAMEN_CORE_TEST_MOVABLE_HPP

#include<iostream>
#include<cassert>

#include<boost/move/move.hpp>

// A movable type, used in tests.
class movable_t
{
    BOOST_COPYABLE_AND_MOVABLE( movable_t)

public:

    movable_t()
    {
        std::cout << "default ctor" << std::endl;
        ptr_ = new int( 0);
    }

    explicit movable_t( int x)
    {
        std::cout << "ctor" << std::endl;
        ptr_ = new int( x);
    }

    movable_t( const movable_t& other)
    {
        std::cout << "copy" << std::endl;
        ptr_ = new int( other.value());
    }

   movable_t( BOOST_RV_REF( movable_t) other) : ptr_( other.ptr_)
   {
        assert( ptr_);
        assert( other.ptr_);

        std::cout << "move" << std::endl;
        other.ptr_ = 0;
   }

    ~movable_t()
    {
        std::cout << "dtor" << std::endl;
        delete ptr_;
    }

    movable_t& operator=( BOOST_COPY_ASSIGN_REF( movable_t) other) // Copy assignment
    {
        assert( ptr_);
        assert( other.ptr_);

        if( this != &other)
        {
            std::cout << "assign" << std::endl;
            int *tmp_p = other.ptr_ ? new int( other.value()) : 0;
            delete ptr_;
            ptr_ = tmp_p;
        }

        return *this;
    }

   movable_t& operator=( BOOST_RV_REF( movable_t) other) //Move assignment
   {
       assert( ptr_);
       assert( other.ptr_);

       if (this != &other)
       {
           std::cout << "move assign" << std::endl;
           delete ptr_;
           ptr_ = other.ptr_;
           other.ptr_ = 0;
       }

       return *this;
   }

    int value() const
    {
        assert( ptr_);
        return *ptr_;
    }

    bool operator==( const movable_t& other)
    {
        if( ptr_ == 0 || other.ptr_ == 0)
        {
            return ptr_ == other.ptr_;
        }

        return *ptr_ == *other.ptr_;
    }

    bool operator!=( const movable_t& other)
    {
        return !( *this == other);
    }

    bool was_moved() const
    {
        return ptr_ == 0;
    }

private:

    int *ptr_;
};

movable_t make_movable( int x) { return movable_t( x);}

// traits
namespace boost
{

template<>
struct has_nothrow_move<movable_t>
{
   static const bool value = true;
};

} // boost

#endif
