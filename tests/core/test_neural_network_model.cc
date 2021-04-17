#include <catch2/catch.hpp>

TEST_CASE("Sanity Test") {
  
  SECTION("Sanity Section 1") {
    REQUIRE(2 < 1);
  }
  
  SECTION("Sanity Section 2") {
    REQUIRE(2 > 1);
  }
}
