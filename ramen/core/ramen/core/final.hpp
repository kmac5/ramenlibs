// From Adobe source libraries. Original license follows.
/*
    Copyright 2005-2007 Adobe Systems Incorporated
    Distributed under the MIT License (see accompanying file LICENSE_1_0_0.txt
    or a copy at http://stlab.adobe.com/licenses.html)
*/

#ifndef RAMEN_CORE_FINAL_HPP
#define RAMEN_CORE_FINAL_HPP

#include<ramen/core/config.hpp>

namespace ramen
{
namespace core
{
namespace implementation
{

template <typename T>
class final
{
protected:

    final() {}
};

} // implementation
} // core
} // ramen

#define RAMEN_CORE_FINAL( T) private virtual ramen::core::implementation::final<T>

#endif
