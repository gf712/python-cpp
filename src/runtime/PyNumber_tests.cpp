#include "PyNumber.hpp"

#include "gtest/gtest.h"


TEST(PyNumber, SatisfiesRelevantConcepts)
{
    ASSERT_TRUE(HasAdd<PyNumber>);
    ASSERT_TRUE(HasRepr<PyNumber>);
}