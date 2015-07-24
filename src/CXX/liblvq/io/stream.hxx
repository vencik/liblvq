#ifndef liblvq__io__stream_hxx
#define liblvq__io__stream_hxx

/**
 *  LVQ model stream I/O
 *
 *  \date    2015/07/23
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


#include <liblvq/math/R_undef.hxx>
#include <liblvq/math/lingebra.hxx>

#include <iostream>
#include <string>
#include <stdexcept>
#include <locale>


/** Undefined value string representation */
#define LIBLVQ__IO__UNDEF "<undef>"


/** Skip whitespace */
#define skip_whitespace() \
    do { while (std::isspace(in.peek())) in.get(); } while (0)


/**
 *  \brief  Serialise extended real number
 */
template <typename base_t>
std::ostream & operator << (
    std::ostream & out,
    const math::realx<base_t> & x)
{
    if (x.is_defined())
        out << (base_t)x;
    else
        out << LIBLVQ__IO__UNDEF;

    return out;
}


/**
 *  \brief  Deserialise extended real number
 */
template <typename base_t>
std::istream & operator >> (
    std::istream & in,
    math::realx<base_t> & x)
{
    base_t x_impl;

    if ((in >> x_impl).fail()) {
        std::string undef;
        in >> undef;

        if (LIBLVQ__IO__UNDEF != undef)
            throw std::runtime_error(
                "math::realx: syntax error");

        x = math::realx<base_t>::undef;
    }
    else
        x = x_impl;

    return in;
}


/**
 *  \brief  Serialise vector
 */
template <typename base_t>
std::ostream & operator << (
    std::ostream & out,
    const math::vector<base_t> & vec)
{
    out << '[';

    for (size_t i = 0; i < vec.rank() - 1; ++i)
        out << vec[i] << ' ';

    out << vec[vec.rank() - 1] << ']';

    return out;
}


/**
 *  \brief  Deserialise vector
 */
template <typename base_t>
std::istream & operator >> (
    std::istream & in,
    math::vector<base_t> & vec)
{
    char c;

    in >> c;
    if ('[' != c)
        throw std::runtime_error(
            "ml::lvq::input_t parse error: '[' expected");

    for (size_t i = 0; i < vec.rank(); ++i)
        in >> vec[i];

    in >> c;
    if (']' != c)
        throw std::runtime_error(
            "ml::lvq::input_t parse error: ']' expected");

    return in;
}


/**
 *  \brief  Serialise matrix
 */
template <typename base_t>
std::ostream & operator << (
    std::ostream & out,
    const math::matrix<base_t> & mx)
{
    size_t row_cnt = mx.row_cnt();

    for (size_t i = 0; i < row_cnt; ++i)
        out << mx[i] << std::endl;

    return out;
}


/**
 *  \brief  Deserialise matrix
 */
template <typename base_t>
std::istream & operator >> (
    std::istream & in,
    math::matrix<base_t> & mx)
{
    size_t row_cnt = mx.row_cnt();

    for (size_t i = 0; i < row_cnt; ++i) {
        skip_whitespace();

        in >> mx[i];
    }

    return in;
}

#endif  // end of #ifndef liblvq__io__stream_hxx
