/**
 *  LVQ unit test
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


#include "config.hxx"

#include <liblvq/ml/lvq.hxx>
#include <liblvq/math/R_undef.hxx>

#include <list>
#include <initializer_list>
#include <algorithm>
#include <iostream>
#include <exception>
#include <stdexcept>


/** Base numeric type */
typedef math::realx<double> base_t;

/** LVQ */
typedef ml::lvq<base_t> lvq_t;


/**
 *  \brief  Serialise base numeric type
 */
std::ostream & operator << (std::ostream & out, const base_t & x) {
    if (x.is_defined())
        out << (double)x;
    else
        out << "<undef>";

    return out;
}

/**
 *  \brief  Serialise input vector
 */
std::ostream & operator << (std::ostream & out, const lvq_t::input_t & input) {
    out << '[';

    for (size_t i = 0; i < input.rank() - 1; ++i)
        out << input[i] << ' ';

    out << input[input.rank() - 1] << ']';

    return out;
}


/**
 *  \brief  Create input vector
 *
 *  \param  init  Initialiser list
 *
 *  \return Input vector
 */
static lvq_t::input_t input(const std::initializer_list<base_t> & init) {
    return lvq_t::input_t(init);
}


/** LVQ low-level test */
static int test_lvq_ll() {
    std::cout << "Low-level test BEGIN" << std::endl;

    int error_cnt = 0;

    std::list<std::pair<lvq_t::input_t, unsigned> > inputs;
    inputs.emplace_back(input({1, 0, 0}), 0);
    inputs.emplace_back(input({0, 1, 0}), 1);
    inputs.emplace_back(input({0, 0, 1}), 2);
    inputs.emplace_back(input({1, 1, 0}), 3);
    inputs.emplace_back(input({1, 0, 1}), 4);
    inputs.emplace_back(input({1, 1, 1}), 5);

    lvq_t lvq(3, 6);

    // Training phase
    std::cout << "Training phase" << std::endl;

    for (unsigned tloop = 1; tloop <= 100; ++tloop) {
        std::cout << "Training loop " << tloop << std::endl;

        std::for_each(inputs.begin(), inputs.end(),
        [&lvq](const std::pair<lvq_t::input_t, unsigned> & input) {
            const auto & vector = input.first;
            unsigned     clas5  = input.second;

            auto dnorm2 = lvq.train1(vector, clas5, 0.01);

            std::cout << vector << " "
                << "|delta|^2 == " << dnorm2
                << std::endl;
        });
    }

    // Testing phase
    std::cout << "Testing phase" << std::endl;

    std::for_each(inputs.begin(), inputs.end(),
    [&error_cnt, &lvq](const std::pair<lvq_t::input_t, unsigned> & input) {
        const auto & vector  = input.first;
        unsigned     clas5   = input.second;
        unsigned     cluster = lvq.classify(vector);
        bool         right   = clas5 == cluster;

        std::cout
            << vector << " (class " << clas5 << ") "
            << "classified as " << cluster << " "
            << "(" << (right ? "right" : "WRONG") << ")"
            << std::endl;

        if (!right) ++error_cnt;
    });

    // Some vectors to test further
    std::cout << "Experimentation phase" << std::endl;

    std::list<lvq_t::input_t> einputs;
    einputs.emplace_back(input({ 0.8, 0.1,-0.2}));
    einputs.emplace_back(input({ 0.2, 1.1,-0.3}));
    einputs.emplace_back(input({-0.3, 0.1, 0.9}));
    einputs.emplace_back(input({ 0.9, 1.2, 0.1}));
    einputs.emplace_back(input({ 0.9, 0.2, 1.1}));
    einputs.emplace_back(input({ 1.3, 0.8, 1.1}));

    std::for_each(einputs.begin(), einputs.end(),
    [&lvq](const lvq_t::input_t & input) {
        // Plain classification
        const size_t cluster = lvq.classify(input);
        std::cout
            << input << " classified as " << cluster
            << std::endl;

        // Weighed classification
        const auto weight = lvq.classify_weight(input);
        for (size_t c = 0; c < weight.size(); ++c)
            std::cout
                << (cluster == c ? "  * " : "    ")
                << "Cluster " << c << ": " << weight[c] * 100.0 << " %"
                << std::endl;

        // Best n classification
        static const size_t best_cnt = 3;

        std::cout << "  Best " << best_cnt << ":" << std::endl;

        const auto best = lvq.classify_best(input, best_cnt);
        std::for_each(best.begin(), best.end(),
        [](const lvq_t::cw_t & cw) {
            std::cout
                << "    Cluster " << cw.first << ": "
                << cw.second * 100.0 << " %"
                << std::endl;
        });

        // Weight threshold classification
        static const double wthres = 0.60;  // 60% weight threshold

        std::cout
            << "  Weight threshold of "
            << wthres * 100.0 << " %:" << std::endl;

        const auto top = lvq.classify_weight_threshold(input, wthres);
        std::for_each(top.begin(), top.end(),
        [](const lvq_t::cw_t & cw) {
            std::cout
                << "    Cluster " << cw.first << ": "
                << cw.second * 100.0 << " %"
                << std::endl;
        });
    });

    std::cout << "Low-level test END" << std::endl;

    return error_cnt;
}


/** LVQ test */
static int test_lvq() {
    std::cout << "Test BEGIN" << std::endl;

    std::list<std::pair<lvq_t::input_t, size_t> > inputs;
    inputs.emplace_back(input({1, 0, 0}), 0);
    inputs.emplace_back(input({0, 1, 0}), 1);
    inputs.emplace_back(input({0, 0, 1}), 2);
    inputs.emplace_back(input({1, 1, 0}), 3);
    inputs.emplace_back(input({1, 0, 1}), 4);
    inputs.emplace_back(input({1, 1, 1}), 5);

    inputs.emplace_back(input({ 0.8, 0.1,-0.2}), 0);
    inputs.emplace_back(input({ 0.2, 1.1,-0.3}), 1);
    inputs.emplace_back(input({-0.3, 0.1, 0.9}), 2);
    inputs.emplace_back(input({ 0.9, 1.2, 0.1}), 3);
    inputs.emplace_back(input({ 0.9, 0.2, 1.1}), 4);
    inputs.emplace_back(input({ 1.3, 0.8, 1.1}), 5);

    inputs.emplace_back(input({ 1.1,-0.1,-0.1}), 0);
    inputs.emplace_back(input({ 0.0, 1.1,-0.1}), 1);
    inputs.emplace_back(input({-0.1, 0.2, 0.8}), 2);
    inputs.emplace_back(input({ 0.9, 1.1, 0.0}), 3);
    inputs.emplace_back(input({ 0.8,-0.1, 1.0}), 4);
    inputs.emplace_back(input({ 1.2, 0.9, 1.0}), 5);

    lvq_t lvq(3, 6);

    // Training
    std::cout << "Training phase" << std::endl;

    lvq.train(inputs);

    float lrate = lvq.learn_rate(inputs);

    std::cout << "Learn rate: " << lrate << std::endl;

    std::cout << "Test END" << std::endl;

    return lrate != 1.0 ? 1 : 0;
}


/** Unit test */
static int main_impl(int argc, char * const argv[]) {
    int exit_code = 64;  // pessimistic assumption

    do {  // pragmatic do ... while (0) loop allowing for breaks
        if (0 != (exit_code = test_lvq_ll())) break;

        if (0 != (exit_code = test_lvq())) break;

    } while (0);  // end of pragmatic loop

    std::cerr
        << "Exit code: " << exit_code
        << std::endl;

    return exit_code;
}

/** Unit test exception-safe wrapper */
int main(int argc, char * const argv[]) {
    int exit_code = 128;

    try {
        exit_code = main_impl(argc, argv);
    }
    catch (const std::exception & x) {
        std::cerr
            << "Standard exception caught: "
            << x.what()
            << std::endl;
    }
    catch (...) {
        std::cerr
            << "Unhandled non-standard exception caught"
            << std::endl;
    }

    return exit_code;
}
