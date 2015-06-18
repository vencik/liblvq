/**
 *  LVQ model
 *
 *  \date    2015/06/18
 *  \author  Vaclav Krpec  <vencik@razdva.cz>
 *
 *
 *  LEGAL NOTICE
 *
 *  Copyright (c) 2015, Vaclav Krpec
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of
 *     its contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
 *  OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "config.hxx"
#include "model.hxx"

#include <stdexcept>


std::ostream & operator << (std::ostream & out, const base_t & x) {
    if (x.is_defined())
        out << (double)x;
    else
        out << "<undef>";

    return out;
}


std::istream & operator >> (std::istream & in, base_t & x) {
    base_impl_t x_impl;

    if ((in >> x_impl).fail())
        x = base_t::undef;
    else
        x = x_impl;

    return in;
}


std::ostream & operator << (std::ostream & out, const lvq_t::input_t & input) {
    out << '[';

    for (size_t i = 0; i < input.rank() - 1; ++i)
        out << input[i] << ' ';

    out << input[input.rank() - 1] << ']';

    return out;
}


std::istream & operator >> (std::istream & in, lvq_t::input_t & input) {
    char c;

    in >> c;
    if ('[' != c)
        throw std::runtime_error(
            "lvq_t::input_t parse error: '[' expected");

    for (size_t i = 0; i < input.rank(); ++i)
        in >> input[i];

    in >> c;
    if (']' != c)
        throw std::runtime_error(
            "lvq_t::input_t parse error: ']' expected");

    return in;
}
