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

#ifndef RAMEN_CORE_EXCEPTIONS_HPP
#define RAMEN_CORE_EXCEPTIONS_HPP

#include<ramen/core/config.hpp>

#include<typeinfo>

#include<ramen/core/name.hpp>
#include<ramen/core/string8.hpp>
#include<ramen/core/types.hpp>

namespace ramen
{
namespace core
{

/*!
\ingroup core
\brief Exception class
*/
class RAMEN_CORE_API exception
{
public:

    exception() {}
    virtual ~exception() {}

    virtual const char *what() const = 0;
};

/*!
\ingroup core
\brief Exception class
*/
class RAMEN_CORE_API runtime_error : public exception
{
public:

    explicit runtime_error( string8_t message);

    virtual const char *what() const;

private:

    string8_t message_;
};

/*!
\ingroup core
\brief Exception class
*/
class RAMEN_CORE_API bad_cast : public exception
{
public:

    bad_cast( const std::type_info& src_type, const std::type_info& dst_type);

    virtual const char *what() const;

private:

    string8_t message_;
};

/*!
\ingroup core
\brief Exception class
*/
class RAMEN_CORE_API bad_type_cast : public exception
{
public:

    bad_type_cast( const type_t& src_type, const type_t& dst_type);

    virtual const char *what() const;

private:

    core::string8_t message_;
};

/*!
\ingroup core
\brief Exception class
*/
class RAMEN_CORE_API key_not_found : public exception
{
public:

    explicit key_not_found( const core::name_t& name);
    explicit key_not_found( core::string8_t message);
    
    virtual const char *what() const;
    
private:
    
    core::string8_t message_;
};

/*!
\ingroup core
\brief Exception class
*/
class RAMEN_CORE_API not_implemented : public exception
{
public:

    not_implemented();

    virtual const char *what() const;
};

} // core
} // ramen

#endif
