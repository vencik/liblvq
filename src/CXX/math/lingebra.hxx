#ifndef math__lingebra_hxx
#define math__lingebra_hxx


#include <stdexcept>
#include <vector>
#include <algorithm>
#include <stdarg>


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

    private:

    std::vector<base_t> m_impl;  /**< Implementation */

    public:

    /**
     *  \brief  Constructor
     *
     *  \param  size  Vector size
     *  \param  init  Initialiser
     */
    vector(size_t size, const base_t & init):
        m_impl(size, init)
    {}

    /**
     *  \brief  Constructor
     *
     *  @param  initl  Initialiser
     */
    vector(const std::initialiser_list<base_t> & initl):
        m_impl(initl)
    {}

    /** Access operator */
    base_t & operator [] (size_t i) { return m_impl[i]; }

    /** Access operator (const) */
    const base_t & operator [] (size_t i) const { return m_impl[i]; }

    /**
     *  \brief  Multiplication by scalar (in place)
     *
     *  \param  coef  Scalar coefficient
     */
    vector & operator *= (const base_t & coef) {
        std::for_each(m_impl.begin(), m_impl.end(),
        [coef](base_t & item) {
            item *= coef;
        }

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

    typedef vector<base_t> vector;  /**< Vector of the base type */

    private:

    typedef std::vector<vector> m_impl;  /**< Implementation */

    public:

    /**
     *  \brief  Constructor
     *
     *  TODO
     *
     *  @param  initl  Initialiser
     */
    matrix()
    {}

    /**
     *  Multiplication by a vector
     *
     *  \param  rarg  Vector
     */
    vector operator * (const vector & rarg) const {
        vector result(m_impl.size(), 0);

        for (size_t i = 0; i < m_impl.size(); ++i)
            result[i] = m_impl[i] * rarg;

        return result;
    }

    /** Access operator */
    vector & operator [] (size_t i) { return m_impl[i]; }

    /** Access operator (const) */
    const vector & operator [] (size_t i) const { return m_impl[i]; }

};  // end of template class matrix

}  // end of namespace math

#endif  // end of #ifndef math__lingebra_hxx
