/**********************************************************************
 * Copyright (C) 2013 Esteban Tovagliari. All Rights Reserved.        *
 **********************************************************************/

#ifndef RAMEN_CORE_CONFIG_HPP
#define RAMEN_CORE_CONFIG_HPP

#include<ramen/config/config.hpp>

#ifdef ramen_core_EXPORTS // <-- #defined by CMake automagically
    #define RAMEN_CORE_API RAMEN_CONFIG_EXPORT
#else
    #define RAMEN_CORE_API RAMEN_CONFIG_IMPORT
#endif

#endif
