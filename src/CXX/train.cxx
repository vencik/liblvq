/**
 *  LVQ: learn training data
 *
 *  \date    2015/06/18
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
#include "model.hxx"

#include <liblvq/io/stream.hxx>

#include <list>
#include <iostream>
#include <limits>
#include <exception>
#include <stdexcept>


/**
 *  \brief  Read training vector
 *
 *  \param  in    Input
 *  \param  rank  Vector rank
 *
 *  \return Training vector & cluster number
 */
static std::pair<lvq_t::input_t, size_t> read_train_set_item(
    std::istream & in,
    size_t         rank)
{
    size_t cluster;
    if ((in >> cluster).fail())
        throw std::runtime_error(
            "parse error: cluster number expected");

    lvq_t::input_t vector(rank);
    in >> vector;

    return std::pair<lvq_t::input_t, size_t>(vector, cluster);
}


/**
 *  \brief  Train LVQ model
 *
 *  \param  input_rank  Input vector rank
 *
 *  \return Exit code
 */
static int train(
    size_t input_rank)
{
    size_t clusters = 0;  // cluster count

    // Read raining set
    std::list<std::pair<lvq_t::input_t, size_t> > train_set;

    while (!std::cin.eof() && EOF != std::cin.peek()) {
        auto pair = read_train_set_item(std::cin, input_rank);

        // Ignore everything till EOL
        std::cin.ignore(
            std::numeric_limits<std::streamsize>::max(),
            '\n');

        std::cerr
            << pair.second << ": " << pair.first << std::endl;

        if (clusters < pair.second + 1)
            clusters = pair.second + 1;

        train_set.push_back(pair);
    }

    lvq_t lvq(input_rank, clusters);

    // Train training set
    lvq.train(train_set);

    // Check learning rate
    float lrate = lvq.learn_rate(train_set);

    std::cout << "Learn rate: " << lrate << std::endl;

    // Print cluster representants
    for (size_t i = 0; i < clusters; ++i)
        std::cout << lvq.get(i) << std::endl;

    return 0;
}


/** Main routine */
static int main_impl(int argc, char * const argv[]) {
    int exit_code = 1;  // faulty execution assumption

    do {  // pragmatic do ... while (0) loop allowing for breaks
        // Read command line arguments
        if (argc < 2) {
            std::cerr
                << "Usage: " << argv[0] << " <input_rank>" << std::endl;

            break;
        }

        exit_code = 64;  // pessimistic assumption

        size_t input_rank = ::atoi(argv[1]);

        exit_code = train(input_rank);

    } while (0);  // end of pragmatic loop

    return exit_code;
}

/** Main routine exception-safe wrapper */
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
