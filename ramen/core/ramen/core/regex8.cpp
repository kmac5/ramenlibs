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

#include<ramen/core/regex8.hpp>

#include<pcre.h>
#include<pcrecpp.h>

#include<ramen/core/memory.hpp>
#include<ramen/core/exceptions.hpp>

namespace ramen
{
namespace core
{

struct regex8_t::impl
{
    impl( const string8_t& pattern) : regex_( pattern.c_str(), pcrecpp::UTF8())
    {
    }

    pcrecpp::RE regex_;
};

regex8_t::regex8_t( const string8_t& pattern) : pimpl_( 0)
{
    pimpl_ = new impl( pattern);
}

regex8_t::~regex8_t()
{
    delete pimpl_;
}

// Copy constructor
regex8_t::regex8_t( const regex8_t& other) : pimpl_( 0)
{
    assert( other.pimpl_);

    pimpl_ = new impl( *other.pimpl_);
}

bool regex8_t::full_match( const string8_t& str) const
{
    assert( pimpl_);

    return pimpl_->regex_.FullMatch( pcrecpp::StringPiece( str.c_str(), str.size()));
}

bool regex8_t::full_match( const char *str) const
{
    assert( pimpl_);

    return pimpl_->regex_.FullMatch( pcrecpp::StringPiece( str));
}

void regex8_t::swap( regex8_t& other)
{
    std::swap( pimpl_, other.pimpl_);
}

} // core
} // ramen
