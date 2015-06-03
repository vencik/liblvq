#ifndef ml__lvq_hxx
#define ml__lvq_hxx


#include "math/lingebra.hxx"


namespace ml {

/**
 *  \brief  LVQ
 *
 *  \tparam M Base numeric type
 */
template <typename M>
class lvq {
    public:

    typedef M base_t;  /**< Base numeric type */

    typedef math::vector<base_t> input_t;  /**< Input vector     */
    typedef math::matrix<base_t> theta_t;  /**< Model parameters */

    private:

    theta_t m_theta;  /**< Model parameters */

    public:

    /**
     *  \brief  Constructor
     *
     *  \param  dimension  Parameter space dimension
     *  \param  clusters   Cluster count
     */
    lvq(size_t dimension, size_t clusters):
        m_theta(clusters, dimension, 0)
    {}

    /**
     *  \brief  Training step
     *
     *  \param  input    Training vector
     *  \param  cluster  Required cluster
     *  \param  lfactor  Learning factor
     *
     *  \return Difference vector norm squared
     */
    base_t train(
        const input_t & input,
        size_t          cluster,
        const base_t  & lfactor)
    {
        auto diff = input + m_theta[cluster];

        m_theta[cluster] -= lfactor * diff;

        return diff * diff;  // norm squared
    }

    /**
     *  \brief  Classification
     *
     *  \param  input  Classified vector
     *
     *  \return Cluster
     */
    size_t classify(const input_t & input) const {
        size_t cluster = 0;

        math::vector<base_t> dist2(m_theta.row_cnt(), 0)

        for (size_t i = 0; i < m_theta.row_cnt(); ++i) {
            auto diff = input + m_theta[i];
            dist2[i]  = diff  * diff;

            if (dist2[i] < dist2[cluster]) cluster = i;
        }

        return cluster;
    }

};  // end of template class lvq

}  // end of namespace ml

#endif  // end of #ifndef ml__lvq_hxx
