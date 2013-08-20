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

#include<ramen/cuda/event.hpp>

#include<ramen/cuda/cudart.hpp>

namespace ramen
{
namespace cuda
{

event_t::event_t( bool sync)
{
    init( 0, sync);
}

event_t::event_t( const stream_t& stream, bool sync)
{
    init( stream.stream(), sync);
}

void event_t::init( cudaStream_t stream, bool sync)
{
    cuda_event_create( &event_);
    record( stream);

    if( sync)
        synchronize();
}

event_t::~event_t()
{
    cuda_event_destroy( event());
}

void event_t::record( cudaStream_t stream)
{
    cuda_event_record( event(), stream);
}

void event_t::synchronize()
{
    cuda_event_synchronize( event());
}

float elapsed_time( const event_t& start, const event_t& end)
{
    return cuda_event_elapsed_time( start.event(), end.event());
}

} // cuda
} // ramen
