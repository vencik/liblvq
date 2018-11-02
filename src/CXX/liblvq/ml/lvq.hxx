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
#include <algorithm>


/** Default convergency window for \c lvq::train */
#define LIBLVQ__ML__LVQ__TRAIN__CONV_WIN 5

/** Default max. number of diverging windows in a row for \c lvq::train */
#define LIBLVQ__ML__LVQ__TRAIN__MAX_DIV_CNT 3

/** Default max. number of training loops in total for \c lvq::train */
#define LIBLVQ__ML__LVQ__TRAIN__MAX_TLC 1000


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

    /** Training/testing sample (together with its annotation) */
    typedef std::pair<input_t, size_t> sample_t;

    private:

    input_t m_normc;  /**< Input normalisation coefficients */
    theta_t m_theta;  /**< Model parameters                 */

    public:

    /**
     *  \brief  Constructor
     *
     *  \param  dimension  Parameter space dimension
     *  \param  clusters   Cluster count
     */
    lvq(size_t dimension, size_t clusters):
        m_normc(dimension, 1),
        m_theta(clusters, dimension)
    {}

    /**
     *  \brief  Input normalisation coefficients setter
     *
     *  BEWARE: Changing input normalisation coefficients will break the model.
     *
     *  \param  normc  Normalisation coefficients
     */
    void normc_set(const input_t & normc) { m_normc = normc; }

    /**
     *  \brief  Normalisation coefficients getter
     *
     *  \return Coefficients
     */
    const input_t & normc_get() const { return m_normc; }

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
     *  \brief  Normalise input vector (in place)
     *
     *  \param  input  Input vector
     */
    void normalise(input_t & input) const {
        size_t ix = 0;
        std::for_each(input.begin(), input.end(),
        [&ix, this](base_t & item) {
            item *= m_normc[ix++];
        });
    }

    /**
     *  \brief  Normalise input vector
     *
     *  \param  input  Input vector
     *
     *  \return Normalised input vector
     */
    input_t normalise(const input_t & input) const {
        input_t ninput = input;
        normalise(ninput);
        return ninput;
    }

    /**
     *  \brief  Find normalisation coefficients for an input vector training set
     *
     *  \param  set  Input vector set (must not be empty)
     *
     *  \return Normalisation coefficients vector
     */
    template <class Set>
    static input_t norm_coefficients(const Set & set) {
        input_t normc(set.begin()->first.rank(), 0);

        base_t n = 0;
        std::for_each(set.begin(), set.end(),
        [&normc, &n](const sample_t & item) {
            const input_t & input = item.first;

            base_t n_div_n_plus_1 = n;
            n += 1;
            n_div_n_plus_1 /= n;

            normc *= n_div_n_plus_1;
            normc += input / n;
        });

        // We need inversions
        std::for_each(normc.begin(), normc.end(),
        [](base_t & item) {
            item = item != 0 ? 1 / item : 1;
        });

        return normc;
    }

    /**
     *  \brief  Normalise set of input vectors (in place)
     *
     *  \param  set    Input vector set
     */
    template <class Set>
    void normalise(Set & set) const {
        std::for_each(set.begin(), set.end(),
        [this](input_t & input) {
            normalise(input);
        });
    }

    /**
     *  \brief  Normalise set of input vectors
     *
     *  \param  set  Input vector set
     *
     *  \return Normalised set of input vectors
     */
    template <class Set>
    Set normalise(const Set & set) const {
        Set nset = set;
        normalise(nset);
        return nset;
    }

    /**
     *  \brief  Training step
     *
     *  NOTE: The training step should work with input with normalised items.
     *  Not normalising items typically means poorer results as noticeable
     *  differences in value magnitudes in different dimensions mean different
     *  contribution rates to vector distances.
     *  Thus, dimensions in which the values are comparatively much smaller
     *  (in absolute value) will have much smaller distinguishing effect.
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
     *  \brief  Training
     *
     *  Train on set of samples.
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
     *  \param  train_set    Training set
     *  \param  conv_win     Convergency window size
     *  \param  max_div_cnt  Max. number of diverging windows
     *  \param  max_tlc      Max. number of training loops in total
     */
    template <class Set>
    void train(
        const Set    & train_set,
        const unsigned conv_win    = LIBLVQ__ML__LVQ__TRAIN__CONV_WIN,
        unsigned       max_div_cnt = LIBLVQ__ML__LVQ__TRAIN__MAX_DIV_CNT,
        const unsigned max_tlc     = LIBLVQ__ML__LVQ__TRAIN__MAX_TLC)
    {
        m_normc = norm_coefficients(train_set);
        const Set set = normalise(train_set);

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
            [&, this](const sample_t & item) {
                auto & n2      = norm2[i];
                auto & vector  = item.first;
                auto   cluster = item.second;

                n2 = train1(vector, cluster, lf);

                DEBUG_MSG(vector <<
                    ": f == " << lf <<
                    ", |delta|^2 == " << n2);

                ++i;
            });

            // Convergency window is full
            if (conv_win == ++conv_win_i) {
                // Compute normalised average norm^2 difference
                i = 0;
                std::for_each(set.begin(), set.end(),
                [&, this](const sample_t & item) {
                    auto & n2      = norm2[i];
                    auto & vector  = item.first;
                    auto   cluster = item.second;

                    const auto diff = vector - m_theta[cluster];
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

        const auto ninput = normalise(input);
        sumd2 = 0;
        for (size_t i = 0; i < m_theta.row_cnt(); ++i) {
            auto diff = ninput - m_theta[i];

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
     *  \brief  Test statistics
     */
    class statistics {
        private:

        /** Statistics counters */
        struct counters {
            size_t tp;   /**< True Positive  */
            size_t fp;   /**< False Positive */
            size_t cnt;  /**< Total count    */

            /** Constructor */
            counters(): tp(0), fp(0), cnt(0) {}

            /** Precision */
            double precision() const {
                return (double)tp / ((double)tp + (double)fp);
            }

            /** Recall */
            double recall() const {
                return (double)tp / (double)cnt;
            }

            /**
             *  \brief  F_beta score
             *
             *  \param  bb  Beta^2
             */
            double F(double bb) const {
                const double p = precision();
                const double r = recall();

                return (1 + bb) * (p * r) / (bb * p + r);
            }

        };  // end of struct counters

        std::vector<counters> m_cnts;     /**< Counters per class      */
        size_t                m_correct;  /**< Correct classifications */
        size_t                m_total;    /**< Total count             */

        public:

        /**
         *  \brief  Constructor
         *
         *  \param  ccnt  Number of classes
         */
        statistics(size_t ccnt):
            m_cnts(ccnt),
            m_correct(0),
            m_total(0)
        {}

        /**
         *  \brief  Record one classification result
         *
         *  \param  cclass  The correct class
         *  \param  dclass  The detected class
         */
        void record(size_t cclass, size_t dclass) {
            // Correct classification
            if (cclass == dclass) {
                ++m_correct;
                ++m_cnts[cclass].tp;
            }

            // Incorrect classification
            else {
                ++m_cnts[dclass].fp;
            }

            ++m_cnts[cclass].cnt;
            ++m_total;
        }

        /** Accuracy */
        inline double accuracy() const {
            return (double)m_correct / (double)m_total;
        }

        /**
         *  \brief  F_beta score
         *
         *  \param  beta  Beta parameter
         *
         *  \return Weighed average of per-class F_beta scores
         */
        double F(double beta) const {
            const double bb  = beta * beta;
            double       sum = 0.0;

            std::for_each(m_cnts.begin(), m_cnts.end(),
            [bb, &sum](const counters & cnts) {
                sum += cnts.cnt * cnts.F(bb);
            });

            return sum / (double)m_total;
        }

        /** F-score (aka F_1 score) */
        inline double F() const { return F(1); }

    };  // end of class statistics

    /**
     *  \brief  Test model
     *
     *  \param  set  Testing set
     *
     *  \return Test statistics
     */
    template <class Set>
    statistics test(const Set & set) {
        statistics stats(m_theta.row_cnt());

        std::for_each(set.begin(), set.end(),
        [this, &stats](const sample_t & item) {
            const auto & vector  = item.first;
            size_t       cluster = item.second;

            size_t lvq_cluster = classify(vector);

            DEBUG_MSG(vector <<
                ": class " << cluster <<
                ", got class " << lvq_cluster);

            stats.record(cluster, lvq_cluster);
        });

        return stats;
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
