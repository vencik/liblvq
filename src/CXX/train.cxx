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
#include <fstream>
#include <limits>
#include <exception>
#include <stdexcept>
#include <cstdlib>


/** Classifier training/test set */
typedef lvq_t::tset_classifier tset_classifier_t;


/** Clustering training/test set */
typedef lvq_t::tset_clustering  tset_clustering_t;


/**
 *  \brief  Read vector
 *
 *  \param  in    Input
 *  \param  rank  Vector rank
 *
 *  \return Vector
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
 *  \brief  Read training/test sample
 *
 *  \param  in    Input
 *  \param  rank  Vector rank
 *
 *  \return Training/test sample
 */
static lvq_t::sample_t read_sample(
    std::istream & in,
    size_t         rank)
{
    size_t cluster;
    if ((in >> cluster).fail())
        throw std::runtime_error(
            "parse error: cluster number expected");

    return lvq_t::sample_t(read_vector(in, rank), cluster);
}


/**
 *  \brief  Read classifier training/test set
 *
 *  Reads sample set; if \c cluster_cnt is nt \c NULL, it stores
 *  the number of clusters in it.
 *
 *  \param  in           Input
 *  \param  rank         Vector rank
 *  \param  cluster_cnt  Collected cluster count (optional)
 *
 *  \return Training/test set
 */
static tset_classifier_t read_classifier_tset(
    std::istream & in,
    size_t         rank,
    size_t       * cluster_cnt = NULL)
{
    if (NULL != cluster_cnt) *cluster_cnt = 0;

    tset_classifier_t tset;

    while (!in.eof() && EOF != in.peek()) {
        auto sample = read_sample(in, rank);

        // Ignore everything till EOL
        in.ignore(
            std::numeric_limits<std::streamsize>::max(),
            '\n');

        //std::cerr << sample.second << ": " << sample.first << std::endl;

        if (NULL != cluster_cnt && *cluster_cnt < sample.second + 1)
            *cluster_cnt = sample.second + 1;

        tset.push_back(sample);
    }

    return tset;
}


/**
 *  \brief  Read clustering training/test set
 *
 *  Reads sample set; if \c cluster_cnt is nt \c NULL, it stores
 *  the number of clusters in it.
 *
 *  \param  in           Input
 *  \param  rank         Vector rank
 *  \param  cluster_cnt  Cluster count (optional)
 *
 *  \return Training/test set
 */
static tset_clustering_t read_clustering_tset(
    std::istream & in,
    size_t         rank,
    size_t       * cluster_cnt = NULL)
{
    if (NULL != cluster_cnt) {
        if ((in >> *cluster_cnt).fail())
            throw std::runtime_error(
                "parse error: cluster number expected");

        // Ignore everything till EOL
        in.ignore(
            std::numeric_limits<std::streamsize>::max(),
            '\n');
    }

    tset_clustering_t tset;

    while (!in.eof() && EOF != in.peek()) {
        auto vector = read_vector(in, rank);

        // Ignore everything till EOL
        in.ignore(
            std::numeric_limits<std::streamsize>::max(),
            '\n');

        //std::cerr << vector << std::endl;

        tset.push_back(vector);
    }

    return tset;
}


/**
 *  \brief  Train LVQ classifier
 *
 *  \param  train_src  Training set source stream
 *  \param  test_src   Testing set source stream
 *  \param  rank       Vector rank
 *
 *  \return Exit code
 */
static int train_classifier(
    std::istream & train_src,
    std::istream & test_src,
    size_t         rank)
{
    size_t cluster_cnt;  // cluster count

    // Read training set
    auto train_set = read_classifier_tset(train_src, rank, &cluster_cnt);

    lvq_t lvq(rank, cluster_cnt);

    // Train the model
    lvq.set_random();
    lvq.train_supervised(train_set);

    // Read test set
    auto test_set = read_classifier_tset(test_src, rank);

    // Test the model
    lvq_t::statistics stats = lvq.test_classifier(test_set);

    // Print statistics
    std::cerr << "F_1      : " << stats.F(1.0)     << std::endl;
    std::cerr << "F_0.5    : " << stats.F(0.5)     << std::endl;
    std::cerr << "F_2      : " << stats.F(2.0)     << std::endl;
    std::cerr << "Accuracy : " << stats.accuracy() << std::endl;

    // Print cluster representants
    for (size_t i = 0; i < cluster_cnt; ++i)
        std::cout << lvq.get(i) << std::endl;

    return 0;
}


/**
 *  \brief  Train LVQ clustering
 *
 *  \param  train_src  Training set source stream
 *  \param  test_src   Testing set source stream
 *  \param  rank       Vector rank
 *
 *  \return Exit code
 */
static int train_clustering(
    std::istream & train_src,
    std::istream & test_src,
    size_t         rank)
{
    // Read training set
    size_t cluster_cnt;
    auto train_set = read_clustering_tset(train_src, rank, &cluster_cnt);

    lvq_t lvq(rank, cluster_cnt);

    // Train the model
    lvq.set_random();
    lvq.train_unsupervised(train_set);

    // Read test set
    auto test_set = read_clustering_tset(test_src, rank);

    // Test the model
    lvq_t::statistics stats = lvq.test_clustering(test_set);

    // Print statistics
    std::cerr << "F_1      : " << stats.F(1.0)     << std::endl;
    std::cerr << "F_0.5    : " << stats.F(0.5)     << std::endl;
    std::cerr << "F_2      : " << stats.F(2.0)     << std::endl;
    std::cerr << "Accuracy : " << stats.accuracy() << std::endl;

    // Print cluster representants
    for (size_t i = 0; i < cluster_cnt; ++i)
        std::cout << lvq.get(i) << std::endl;

    return 0;
}


/** Main routine */
static int main_impl(int argc, char * const argv[]) {
    std::srand(0);

    int exit_code = 1;  // faulty execution assumption

    do {  // pragmatic do ... while (0) loop allowing for breaks
        // Read command line arguments
        if (argc < 3) {
            std::cerr
                << "Usage: " << argv[0]
                << " {supervised|unsupervised} <rank> [<training_set>] [<test_set>]"
                << std::endl;
            break;
        }

        bool   supervised = "supervised" == std::string(argv[1]);
        size_t rank       = ::atoi(argv[2]);

        std::istream  * train_src = &std::cin;
        std::ifstream   train_set;
        if (argc > 2) {
            train_set.open(argv[2]);
            train_src = &train_set;
        }

        std::istream  * test_src = &std::cin;
        std::ifstream   test_set;
        if (argc > 3) {
            test_set.open(argv[3]);
            test_src = &test_set;
        }

        exit_code = 64;  // pessimistic assumption

        if (supervised)
            exit_code = train_classifier(*train_src, *test_src, rank);
        else
            exit_code = train_clustering(*train_src, *test_src, rank);

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
