# Compiler & linker flags
AM_CXXFLAGS = -g -Wall -Werror
AM_LDFLAGS  =

SUBDIRS = \
    math \
    ml \
    unit_test


# Internal headers
noinst_HEADERS = \
    config.hxx \
    model.hxx

lib_LTLIBRARIES = \
    liblvq.la

liblvq_la_SOURCES =

liblvq_la_LIBADD =


# Executables
bin_PROGRAMS = \
    lvq_train \
    lvq_classify

lvq_train_SOURCES = \
    model.cxx \
    train.cxx

lvq_classify_SOURCES = \
    model.cxx \
    classify.cxx

LDADD = liblvq.la
