#ifndef math__R_undef_hxx
#define math__R_undef_hxx

/**
 *  Real numbers with undefined value.
 *
 *  Augmentation of R set with an undefned value.
 *  Algebraic operations with the undefined value produce 0.
 *
 *  \date    2015/06/03
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

#include <cmath>

#ifndef NAN
#error "NAN is not defined"
#endif


namespace math {

/** R u {undef} */
template <typename base_t>
class realx {
    private:

    base_t m_impl;  /**< Implementation */

    /**
     *  \brief  Binary operator template implementation
     *
     *  \param  bin_op    Operation implementation for defined values
     *  \param  larg      Operation left argument
     *  \param  rarg      Operation right argument
     *  \param  on_undef  Operation result if an operand is undefined
     *
     *  \return Operation result (or \c on_undef)
     */
    template <class BinOp>
    static base_t undef_case(
        BinOp  bin_op,
        base_t larg,
        base_t rarg,
        base_t on_undef)
    {
        return isnan(larg) || isnan(rarg) ? on_undef : bin_op(larg, rarg);
    }

    public:

    /** Default constructor (undef) */
    realx(): m_impl(NAN) {}

    /** Constructor */
    realx(base_t val): m_impl(val) {}

    /** Comparison */
    bool operator == (const realx & rarg) const {
        if (isnan(m_impl)) return isnan(rarg.m_impl);

        return isnan(rarg.m_impl) ? false : m_impl == rarg.m_impl;
    }

    /** Comparison (negated) */
    bool operator != (const realx & rarg) const { return !(*this == rarg); }

    /** Addition */
    realx operator + (const realx & rarg) const {
        return undef_case(
            [](base_t r, base_t l) { return r + l; },
            m_impl, rarg.m_impl, 0);
    }

    /** Subtraction */
    realx operator - (const realx & rarg) const {
        return undef_case(
            [](base_t r, base_t l) { return r - l; },
            m_impl, rarg.m_impl, 0);
    }

    /** Multiplication */
    realx operator * (const realx & rarg) const {
        return undef_case(
            [](base_t r, base_t l) { return r * l; },
            m_impl, rarg.m_impl, 0);
    }

    /** Division */
    realx operator / (const realx & rarg) const {
        return undef_case(
            [](base_t r, base_t l) { return r / l; },
            m_impl, rarg.m_impl, 0);
    }

};  // end of template class realx

}  // end of namespace math

#endif  // end of #ifndef math__R_undef_hxx
