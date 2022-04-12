#include <gtest/gtest.h>
#include "nvbox/Stopwatch.hpp"

TEST(HelloTest1, BasicAssertions)
{
    EXPECT_STRNE("hello", "world");

    EXPECT_EQ(7 * 6, 42);
}