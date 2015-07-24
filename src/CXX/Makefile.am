# Compiler & linker flags
AM_CXXFLAGS = -g -Wall -Werror
AM_LDFLAGS  =

# Enable debug messages
if ENABLE_DEBUG
AM_CXXFLAGS += -DENABLE_DEBUG
endif


SUBDIRS = \
    liblvq \
    unit_test


# Internal headers
noinst_HEADERS = \
    config.hxx \
    model.hxx

#
# No shared object is necessary so far
# Uncomment in case it becomes so
#lib_LTLIBRARIES = \
#    liblvq.la
#
#liblvq_la_SOURCES =
#
#liblvq_la_LIBADD =
#
#LDADD = liblvq.la


# Executables
bin_PROGRAMS = \
    lvq_train \
    lvq_classify

lvq_train_SOURCES = \
    train.cxx

lvq_classify_SOURCES = \
    classify.cxx
