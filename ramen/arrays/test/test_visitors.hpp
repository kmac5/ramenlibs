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

struct max_visitor
{
    max_visitor()
    {
        max_value = -1677.0;
    }

    void operator()( const const_array_ref_t<float>& array)
    {
        for( int i = 0, e = array.size(); i < e; ++i)
        {
            if( array[i] > max_value)
                max_value = array[i];
        }
    }

    void operator()( const const_array_ref_t<boost::uint32_t>& array)
    {
        for( int i = 0, e = array.size(); i < e; ++i)
        {
            if( array[i] > max_value)
                max_value = array[i];
        }
    }

    template<class T>
    void operator()( const const_array_ref_t<T>& array)
    {
        assert( false);
    }    

    double max_value;
};

class double_visitor
{
public:

    void operator()( array_ref_t<float>& array)
    {
        for( int i = 0, e = array.size(); i < e; ++i)
            array[i] *= 2.0f;
    }

    void operator()( array_ref_t<boost::uint32_t>& array)
    {
        for( int i = 0, e = array.size(); i < e; ++i)
            array[i] *= 2;
    }

    template<class T>
    void operator()( array_ref_t<T>& array)
    {
        assert( false);
    }
};
