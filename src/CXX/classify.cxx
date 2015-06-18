/**
 *  LVQ: classify data
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

#include <list>
#include <iostream>
#include <limits>
#include <exception>
#include <stdexcept>


/**
 *  \brief  Read vector
 *
 *  \param  in    Input
 *  \param  rank  Vector rank
 *
 *  \return Training vector
 */
static lvq_t::input_t read_vector(
    std::istream & in,
    size_t         rank)
{
    lvq_t::input_t vector(rank);
    in >> vector;

    return vector;
}


/**
 *  \brief  Ignore everything till EOL
 *
 *  \param  in  Input stream
 */
static void ignore_till_eol(std::istream & in) {
    in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}


/**
 *  \brief  Read LVQ model from std. input
 *
 *  \param  input_rank  Input vector rank
 *  \param  clusters    Number of clusters
 *
 *  \return LVQ model
 */
static lvq_t read_model(
    size_t input_rank,
    size_t clusters)
{
    lvq_t lvq(input_rank, clusters);

    size_t cluster = 0;
    while (!std::cin.eof() && '\n' != std::cin.peek()) {
        auto vector = read_vector(std::cin, input_rank);

        ignore_till_eol(std::cin);
        std::cerr
            << "Cluster " << cluster << ": " << vector << std::endl;

        lvq.set(vector, cluster);

        ++cluster;
    }

    ignore_till_eol(std::cin);  // read-out empty line

    return lvq;
}


/**
 *  \brief  Classify vectors from std. input
 *
 *  \param  input_rank  Input vector rank
 *  \param  clusters    Number of clusters
 *
 *  \return Exit code
 */
static int classify(
    size_t input_rank,
    size_t clusters)
{
    lvq_t lvq = read_model(input_rank, clusters);

    while (!std::cin.eof() && EOF != std::cin.peek()) {
        auto vector = read_vector(std::cin, input_rank);

        ignore_till_eol(std::cin);
        std::cerr
            << "Vector " << vector << std::endl;

        size_t cluster = lvq.classify(vector);

        std::cout << cluster << std::endl;
    }

    return 0;
}


/** Main routine */
static int main_impl(int argc, char * const argv[]) {
    int exit_code = 1;  // faulty execution assumption

    do {  // pragmatic do ... while (0) loop allowing for breaks
        // Read command line arguments
        if (argc < 3) {
            std::cerr
                << "Usage: " << argv[0] << " <input_rank> <clusters>"
                << std::endl;

            break;
        }

        exit_code = 64;  // pessimistic assumption

        size_t input_rank = ::atoi(argv[1]);
        size_t clusters   = ::atoi(argv[2]);

        exit_code = classify(input_rank, clusters);

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
