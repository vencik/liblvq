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
     *  \brief  Binary assignment operator template implementation
     *
     *  \param  bin_op    Operation implementation for defined values
     *  \param  larg      Operation left argument
     *  \param  rarg      Operation right argument
     *  \param  on_undef  Operation result if an operand is undefined
     */
    template <class BinOp>
    static void undef_case_binop_assign(
        BinOp          bin_op,
        base_t       & larg,
        const base_t & rarg,
        const base_t & on_undef)
    {
        larg = isnan(larg) || isnan(rarg)
             ? on_undef
             : bin_op(larg, rarg);
    }

    public:

    /** Default constructor (undef) */
    realx(): m_impl(NAN) {}

    /** Constructor */
    realx(base_t val): m_impl(val) {}

    /** Defined check */
    bool is_defined() const { return !isnan(m_impl); }

    /** Base type value getter */
    operator base_t () const { return m_impl; }

    /** Comparison */
    bool operator == (const realx & rarg) const {
        if (isnan(m_impl)) return isnan(rarg.m_impl);

        return isnan(rarg.m_impl) ? false : m_impl == rarg.m_impl;
    }

    /** Comparison with base type */
    bool operator == (const base_t & rarg) const {
        return *this == realx(rarg);
    }

    /** Comparison (negated) */
    bool operator != (const realx & rarg) const { return !(*this == rarg); }

    /** Comparison with base type (negated) */
    bool operator != (const base_t & rarg) const { return !(*this == rarg); }

    /** Addition (in place) */
    realx & operator += (const realx & rarg) {
        undef_case_binop_assign(
            [](const base_t & r, const base_t & l) { return r + l; },
            m_impl, rarg.m_impl, 0);

        return *this;
    }

    /** Addition */
    realx operator + (const realx & rarg) const {
        realx result(*this);

        return result += rarg;
    }

    /** Subtraction (in place) */
    realx & operator -= (const realx & rarg) {
        undef_case_binop_assign(
            [](const base_t & r, const base_t & l) { return r - l; },
            m_impl, rarg.m_impl, 0);

        return *this;
    }

    /** Subtraction */
    realx operator - (const realx & rarg) const {
        realx result(*this);

        return result -= rarg;
    }

    /** Multiplication (in place) */
    realx & operator *= (const realx & rarg) {
        undef_case_binop_assign(
            [](const base_t & r, const base_t & l) { return r * l; },
            m_impl, rarg.m_impl, 0);

        return *this;
    }

    /** Multiplication */
    realx operator * (const realx & rarg) const {
        realx result(*this);

        return result *= rarg;
    }

    /** Division (in place) */
    realx & operator /= (const realx & rarg) {
        undef_case_binop_assign(
            [](const base_t & r, const base_t & l) { return r / l; },
            m_impl, rarg.m_impl, 0);

        return *this;
    }

    /** Division */
    realx operator / (const realx & rarg) const {
        realx result(*this);

        return result /= rarg;
    }

};  // end of template class realx


/** \c realx comparison with base type */
template <typename base_t>
bool operator == (const base_t & larg, const realx<base_t> & rarg) {
    return rarg == larg;  // equality is symmetric
}


/** \c realx comparison with base type (negated) */
template <typename base_t>
bool operator != (const base_t & larg, const realx<base_t> & rarg) {
    return rarg != larg;  // inequality is symmetric
}

}  // end of namespace math

#endif  // end of #ifndef math__R_undef_hxx
