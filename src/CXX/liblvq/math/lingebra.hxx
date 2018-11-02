#ifndef liblvq__math__lingebra_hxx
#define liblvq__math__lingebra_hxx

/**
 *  Linear algebra module
 *
 *  The module contains what's necessary for vector/matrix computation.
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


#include <stdexcept>
#include <vector>
#include <algorithm>
#include <initializer_list>
#include <cstdarg>


namespace math {

/**
 *  \brief  Vector
 *
 *  \tparam M Base numeric type
 */
template <typename M>
class vector {
    public:

    typedef M base_t;  /**< Base numeric type */

    typedef std::vector<base_t>             impl_t;          /**< Impl. type */
    typedef typename impl_t::iterator       iterator;        /**< Iterator   */
    typedef typename impl_t::const_iterator const_iterator;  /**< Const iter */

    private:

    impl_t m_impl;  /**< Implementation */

    public:

    /**
     *  \brief  Constructor
     *
     *  \param  size  Vector size
     *  \param  init  Initialiser
     */
    vector(size_t size, const base_t & init = 0):
        m_impl(size, init)
    {}

    /**
     *  \brief  Constructor
     *
     *  \param  initl  Initialiser
     */
    vector(const std::initializer_list<base_t> & initl):
        m_impl(initl)
    {}

    /** Vector rank */
    size_t rank() const { return m_impl.size(); }

    /** Access operator */
    base_t & operator [] (size_t i) { return m_impl[i]; }

    /** Access operator (const) */
    const base_t & operator [] (size_t i) const { return m_impl[i]; }

    /** Begin iterator */
    iterator begin() { return m_impl.begin(); }

    /** End iterator */
    iterator end() { return m_impl.end(); }

    /** Const. begin iterator */
    const_iterator begin() const { return m_impl.begin(); }

    /** Const. end iterator */
    const_iterator end() const { return m_impl.end(); }

    /**
     *  \brief  Multiplication by scalar (in place)
     *
     *  \param  coef  Scalar coefficient
     */
    vector & operator *= (const base_t & coef) {
        std::for_each(m_impl.begin(), m_impl.end(),
        [coef](base_t & item) {
            item *= coef;
        });

        return *this;
    }

    /**
     *  \brief  Multiplication by scalar
     *
     *  \param  coef  Scalar coefficient
     */
    vector operator * (const base_t & coef) const {
        vector result(*this);
        result *= coef;
        return result;
    }

    /**
     *  \brief  Division by scalar (in place)
     *
     *  \param  denom  Scalar denominator
     */
    vector & operator /= (const base_t & denom) { return *this *= 1/denom; }

    /**
     *  \brief  Division by scalar
     *
     *  \param  denom  Scalar denominator
     */
    vector operator / (const base_t & denom) const { return *this * 1/denom; }

    /**
     *  \brief  Scalar multiplication
     *
     *  \param  rarg  Right-hand argument
     */
    base_t operator * (const vector & rarg) const {
        if (m_impl.size() != rarg.m_impl.size())
            throw std::logic_error(
                "math::vector: incompatible scalar mul. args");

        base_t sum = 0;
        for (size_t i = 0; i < m_impl.size(); ++i)
            sum += m_impl[i] * rarg.m_impl[i];

        return sum;
    }

    private:

    /**
     *  \brief  Per-item operation (in place)
     *
     *  \param  rarg  Right-hand argument
     *  \param  fn    Item computation
     */
    template <class Fn>
    vector & per_item_in_place(const vector & rarg, Fn fn) {
        if (m_impl.size() != rarg.m_impl.size())
            throw std::logic_error(
                "math::vector: incompatible args");

        for (size_t i = 0; i < m_impl.size(); ++i)
            m_impl[i] = fn(m_impl[i], rarg[i]);

        return *this;
    }

    public:

    /** Vector addition (in place) */
    vector & operator += (const vector & rarg) {
        return per_item_in_place(rarg,
        [](const base_t & la, const base_t & ra) {
            return la + ra;
        });
    }

    /** Vector addition */
    vector operator + (const vector & rarg) const {
        vector result(*this);
        result += rarg;
        return result;
    }

    /** Vector subtraction (in place) */
    vector & operator -= (const vector & rarg) {
        return per_item_in_place(rarg,
        [](const base_t & la, const base_t & ra) {
            return la - ra;
        });
    }

    /** Vector subtraction */
    vector operator - (const vector & rarg) const {
        vector result(*this);
        result -= rarg;
        return result;
    }

    /** Vectors equal */
    bool operator == (const vector & rarg) const {
        if (m_impl.size() != rarg.m_impl.size())
            throw std::logic_error(
                "math::vector: comparing incompatible args");

        for (size_t i = 0; i < m_impl.size(); ++i)
            if (m_impl[i] != rarg.m_impl[i]) return false;

        return true;
    }

    /** Vectors not equal */
    bool operator != (const vector & rarg) const {
        return !(*this == rarg);
    }

};  // end of template class vector

/** Product of scalar and vector */
template <typename M>
vector<M> operator * (const M & larg, const vector<M> & rarg) {
    return rarg * larg;  // c * v is comutative
}


/**
 *  \brief  Matrix
 *
 *  \tparam M Base numeric type
 */
template <typename M>
class matrix {
    public:

    typedef M base_t;  /**< Base numeric type */

    typedef vector<base_t> vector_t;  /**< Vector of the base type */

    private:

    std::vector<vector_t> m_impl;  /**< Implementation */

    public:

    /**
     *  \brief  Constructor
     *
     *  \param  row_cnt  Number of rows
     *  \param  col_cnt  Number of columns
     *  \param  init     Initialiser
     */
    matrix(size_t row_cnt, size_t col_cnt, const base_t & init = 0) {
        m_impl.reserve(row_cnt);

        for (size_t i = 0; i < row_cnt; ++i)
            m_impl.emplace_back(col_cnt, init);
    }

    /**
     *  \brief  Constructor (rows initialisation)
     *
     *  \param  inits  Row initialiser lists
     */
    matrix(
        const std::initializer_list<std::initializer_list<base_t> > & inits)
    {
        m_impl.reserve(inits.size());

        std::for_each(inits.begin(), inits.end(),
        [this](const std::initializer_list<base_t> & init) {
            m_impl.emplace_back(init);
        });
    }

    /** Number of rows */
    size_t row_cnt() const { return m_impl.size(); }

    /**
     *  Multiplication by a vector
     *
     *  \param  rarg  Vector
     */
    vector_t operator * (const vector_t & rarg) const {
        vector_t result(m_impl.size(), 0);

        for (size_t i = 0; i < m_impl.size(); ++i)
            result[i] = m_impl[i] * rarg;

        return result;
    }

    /** Access operator */
    vector_t & operator [] (size_t i) { return m_impl[i]; }

    /** Access operator (const) */
    const vector_t & operator [] (size_t i) const { return m_impl[i]; }

};  // end of template class matrix

}  // end of namespace math

#endif  // end of #ifndef liblvq__math__lingebra_hxx
