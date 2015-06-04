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

#include "ml/lvq.hxx"
#include "math/R_undef.hxx"

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

    for (size_t i = 0; i < input.size() - 1; ++i)
        out << input[i] << ' ';

    out << input[input.size() - 1] << ']';

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


/** LVQ test */
static int test_lvq() {
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

            auto dnorm2 = lvq.train(vector, clas5, 0.01);

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
        std::cout
            << input << " classified as " << lvq.classify(input)
            << std::endl;
    });

    return error_cnt;  // all OK
}


/** Unit test */
static int main_impl(int argc, char * const argv[]) {
    int exit_code = 64;  // pessimistic assumption

    do {  // pragmatic do ... while (0) loop allowing for breaks
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