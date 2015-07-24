#ifndef liblvq__ml__lvq_hxx
#define liblvq__ml__lvq_hxx

/**
 *  LVQ
 *
 *  Implementation of the Learning Vector Quantisation algorithm.
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


#include <liblvq/math/lingebra.hxx>
#include <liblvq/io/stream.hxx>
#include <liblvq/io/debug.hxx>

#include <cassert>
#include <string>
#include <fstream>


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

    typedef std::pair<size_t, double> cw_t;  /**< Cluster & its weight */

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
        m_theta(clusters, dimension)
    {}

    /**
     *  \brief  Cluster representant setter
     *
     *  \param  input    Training vector
     *  \param  cluster  Required cluster
     */
    void set(const input_t & input, size_t cluster) {
        m_theta[cluster] = input;
    }

    /**
     *  \brief  Cluster representant getter
     *
     *  \param  cluster  Cluster
     *
     *  \return Current cluster representant
     */
    const input_t & get(size_t cluster) const { return m_theta[cluster]; }

    /**
     *  \brief  Training step
     *
     *  \param  input    Training vector
     *  \param  cluster  Required cluster
     *  \param  lfactor  Learning factor
     *
     *  \return Difference vector norm squared
     */
    base_t train1(
        const input_t & input,
        size_t          cluster,
        const base_t  & lfactor)
    {
        auto & theta = m_theta[cluster];
        auto   diff  = input - theta;
        auto   fdiff = lfactor * diff;

        // Fix undefined values
        for (size_t i = 0; i < diff.rank(); ++i) {
            // diff is undefined if at least one of input or theta are undefined
            if (isnan(diff[i])) {
                // theta is undefined
                if (isnan(theta[i])) {
                    // input is defined => set theta accordingly (avoid factor)
                    if (!isnan(input[i])) {
                        theta[i] = 0;
                        fdiff[i] = input[i];
                    }
                }

                // theta is defined => input is undefined
                else
                    fdiff[i] = 0;  // don't change theta

                diff[i] = 0;  // make sure norm is defined
            }
        }

        m_theta[cluster] += fdiff;

        diff *= (1 - lfactor);
        return diff * diff;  // result diff. norm squared
    }

    /**
     *  \brief  Train set
     *
     *  Train set of classified samples.
     *
     *  The function uses a heuristic auto-adaptation of learning factor.
     *  Initially, the factor is set to 1 (i.e. the 1st ever training sample
     *  is used directly as the cluster representant.
     *  The lower the sample/cluster representant difference norm squared
     *  gets, the lower the learning factor is set.
     *
     *  Training stops if an acceptable average difference norm is achieved
     *  or if max. training loop count is reached.
     *
     *  \param  set          Training set
     *  \param  conv_win     Convergency window size
     *  \param  max_div_cnt  Max. number of diverging windows
     *  \param  max_tlc      Max. number of training loops in total
     */
    template <class Set>
    void train(
        const Set    & set,
        const unsigned conv_win    = 5,
        unsigned       max_div_cnt = 3,
        const unsigned max_tlc     = 1000)
    {
        const math::vector<base_t> l(set.size(), 1);  // unit vector

        math::vector<base_t> norm2(set.size());  // diff norm squared

        unsigned conv_win_i = 0;      // convergency window index
        unsigned div_cnt    = 0;      // continuous divergency counter
        base_t andn2        = 0;      // avg normalised diff norm squared
        base_t lf           = 0.999;  // learning factor

        for (unsigned loop = 1; loop <= max_tlc; ++loop) {
            DEBUG_MSG("ml::lvq::train: loop " << loop);

            size_t i = 0;

            std::for_each(set.begin(), set.end(),
            [&, this](const std::pair<input_t, size_t> & item) {
                auto & n2      = norm2[i];
                auto & sample  = item.first;
                auto   cluster = item.second;

                n2 = train1(sample, cluster, lf);

                DEBUG_MSG(sample <<
                    ": f == " << lf <<
                    ", |delta|^2 == " << n2);

                ++i;
            });

            // Convergency window is full
            if (conv_win == ++conv_win_i) {
                // Compute normalised average norm^2 difference
                i = 0;
                std::for_each(set.begin(), set.end(),
                [&, this](const std::pair<input_t, size_t> & item) {
                    auto & n2      = norm2[i];
                    auto & sample  = item.first;
                    auto   cluster = item.second;

                    const auto diff = sample - m_theta[cluster];
                    n2 = diff * diff;

                    ++i;
                });

                base_t new_andn2 = norm2 * l;
                norm2 /= new_andn2;
                new_andn2 = (norm2 * l) / (base_t)set.size();

                DEBUG_MSG("Normalised delta norm^2 == " << norm2);
                DEBUG_MSG("Avg normalised |delta norm^2| == " << new_andn2);

                // Divergency
                if (new_andn2 >= andn2) {
                    DEBUG_MSG("Divergency");

                    if (++div_cnt >= max_div_cnt)
                        break;  // diverged for too long

                    lf /= 2;
                }

                // Convergency
                else {
                    DEBUG_MSG("CONVERGENCY");

                    div_cnt = 0;
                }

                // Store normalised average norm^2 difference
                andn2 = new_andn2;

                conv_win_i = 0;
            }
        }
    }

    private:

    /**
     *  \brief  Classification (implementation)
     *
     *  \param[in ]  input  Classified vector
     *  \param[out]  dist2  Distances squared vector
     *  \param[out]  sumd2  Distances squared sum
     *
     *  \return Cluster
     */
    size_t classify_impl(
        const input_t        & input,
        math::vector<base_t> & dist2,
        base_t               & sumd2) const
    {
        assert(m_theta.row_cnt() == dist2.rank());

        size_t cluster = 0;

        sumd2 = 0;
        for (size_t i = 0; i < m_theta.row_cnt(); ++i) {
            auto diff = input - m_theta[i];

            // Fill missing differences in
            for (size_t j = 0; j < diff.rank(); ++j)
                if (isnan(diff[j]))
                    diff[j] = 0;  // optimistic approach

            dist2[i]  = diff * diff;
            sumd2    += dist2[i];

            if (dist2[i] < dist2[cluster]) cluster = i;
        }

        return cluster;
    }

    public:

    /**
     *  \brief  Classification (n-ary)
     *
     *  The function provides the winning cluster.
     *
     *  \param  input  Classified vector
     *
     *  \return Cluster
     */
    size_t classify(const input_t & input) const {
        math::vector<base_t> dist2(m_theta.row_cnt());
        base_t sumd2;

        return classify_impl(input, dist2, sumd2);
    }

    /**
     *  \brief  Classification (weighed)
     *
     *  The function provides vector of squared distance weights
     *  per each cluster.
     *  The weight is computed as follows:
     *    v_c = sum_C dist_c^2 / dist_c^2
     *    w_c = v_c / sum_C v_c
     *
     *  I.e. the weight represents normalised (sum eq. 1) measure of
     *  relative distance ratio between the \c input and each cluster
     *  representant.
     *
     *  \param  input  Classified vector
     *
     *  \return Vector of squared distance weights per cluster
     */
    std::vector<double> classify_weight(const input_t & input) const {
        math::vector<base_t> dist2(m_theta.row_cnt());
        base_t sumd2;

        classify_impl(input, dist2, sumd2);

        std::vector<double> weight(dist2.rank());
        float fnorm = 0.0;
        for (size_t i = 0; i < dist2.rank(); ++i)
            fnorm += weight[i] = (float)(sumd2 / dist2[i]);
        for (size_t i = 0; i < weight.size(); ++i)
            weight[i] /= fnorm;

        return weight;
    }

    private:

    /**
     *  \brief  Sort cluster weights
     *
     *  Provides vector of [cluster, weight] pairs, sorted by weights
     *  in descending order.
     *
     *  \param  weight  Cluster weights
     *
     *  \return Vector of [cluster, weight] pairs
     */
    static std::vector<cw_t> sort_weight(const std::vector<double> & weight) {
        std::vector<cw_t> sorted(weight.size());

        for (size_t c = 0; c < weight.size(); ++c)
            sorted[c] = cw_t(c, weight[c]);

        std::sort(sorted.begin(), sorted.end(),
        [](const cw_t & cw_l, const cw_t & cw_r) -> bool {
            return cw_r.second < cw_l.second;
        });

        return sorted;
    }

    /**
     *  \brief  Renormalise cluster weights
     *
     *  Renormalises [cluster, weight] pairs.
     *
     *  \param[in,out]  cw  [cluster, weighs] pairs
     *
     *  \return \c cw
     */
    static std::vector<cw_t> & renormalise(std::vector<cw_t> & cw) {
        double fnorm = 0.0;

        std::for_each(cw.begin(), cw.end(),
        [&fnorm](const cw_t & cw) { fnorm += cw.second; });

        std::for_each(cw.begin(), cw.end(),
        [fnorm](cw_t & cw) { cw.second /= fnorm; });

        return cw;
    }

    public:

    /**
     *  \brief  Best matching clusters (renormalised)
     *
     *  Provides \c n best matching clusters with renormalised weights.
     *  Renormalisation means that only the weights of the \c n best
     *  clusters are taken, as if the rest got weight of 0.
     *
     *  \param  weight  Cluster weights (as returned by \ref classify_weight)
     *  \param  n       Number of output clusters
     *
     *  \return \c n best matching clusters with their renormalised weights
     */
    static std::vector<cw_t> best(
        const std::vector<double> & weight,
        size_t                      n)
    {
        std::vector<cw_t> best = sort_weight(weight);

        auto pos = best.begin();
        for (size_t i = 0; i < n && pos != best.end(); ++i, ++pos);

        best.erase(pos, best.end());

        return renormalise(best);
    }

    /**
     *  \brief  Classify to best matching clusters
     *
     *  Same as if \ref best was applied to \ref classify_weight.
     *
     *  \param  input  Classified vector
     *  \param  n      Number of output clusters
     *
     *  \return \c n best matching clusters with their renormalised weights
     */
    std::vector<cw_t> classify_best(const input_t & input, size_t n) const {
        return best(classify_weight(input), n);
    }

    /**
     *  \brief  Weight threshold reaching clusters
     *
     *  Provides best matching clusters which combined weight reaches
     *  required threshold; i.e. sum_Cbest w_c >= wthres.
     *  Returns Cbest with renormalised weights (see \ref best).
     *
     *  \param  weight  Cluster weights (as returned by \ref classify_weight)
     *  \param  wthres  Weight threshold
     *
     *  \return Best matching clusters with their renormalised weights
     */
    static std::vector<cw_t> weight_threshold(
        const std::vector<double> & weight,
        double                      wthres)
    {
        std::vector<cw_t> best = sort_weight(weight);

        auto pos = best.begin();
        for (; pos != best.end() && 0.0 < wthres; ++pos)
            wthres -= pos->second;

        best.erase(pos, best.end());

        return renormalise(best);
    }

    /**
     *  \brief  Classify to weight threshold
     *
     *  Same as if \ref weight_threshold was applied to \ref classify_weight.
     *
     *  \param  input   Classified vector
     *  \param  wthres  Weight threshold
     *
     *  \return Best matching clusters with their renormalised weights
     */
    std::vector<cw_t> classify_weight_threshold(
        const input_t & input, double wthres) const
    {
        return weight_threshold(classify_weight(input), wthres);
    }

    /**
     *  \brief  Compute learn rate of a training set
     *
     *  \param  set  Training set
     *
     *  \return Learn rate
     */
    template <class Set>
    float learn_rate(const Set & set) {
        float learned_cnt = 0;

        std::for_each(set.begin(), set.end(),
        [this, &learned_cnt](const std::pair<input_t, size_t> & item) {
            const auto & sample  = item.first;
            size_t       cluster = item.second;

            size_t lvq_cluster = classify(sample);

            DEBUG_MSG(sample <<
                ": class " << cluster <<
                ", got class " << lvq_cluster);

            if (lvq_cluster == cluster) ++learned_cnt;
        });

        return learned_cnt / set.size();
    }

    /**
     *  \brief  Store LVQ instance
     *
     *  \param  file  Output file
     */
    void store(const std::string & file) const {
        std::ofstream fs;
        fs.open(file, std::ios::out);

        fs
            << m_theta[0].rank() << ' '
            << m_theta.row_cnt() << std::endl
            << m_theta;

        fs.close();
    }

    /**
     *  \brief  Load LVQ instance
     *
     *  \param  file  Input file
     *
     *  \return LVQ instance
     */
    static lvq load(const std::string & file) {
        std::ifstream fs;
        fs.open(file, std::ios::in);

        size_t dimension;
        size_t clusters;
        fs >> dimension >> clusters;

        lvq inst(dimension, clusters);

        fs >> inst.m_theta;

        fs.close();

        return inst;
    }

};  // end of template class lvq

}  // end of namespace ml

#endif  // end of #ifndef liblvq__ml__lvq_hxx
