#include <catch/catch.hpp>
#include <cpplsq/SingleDiff.hpp>
#include <spline/SimplePolynomial.hpp>
#include <random>
#include <iostream>

TEST_CASE( "Univariate differentiation works correctly", "[cpplsq]" )
{

   cpplsq::SingleDiff<double> x;
   SECTION( "Initialization with comma operator works correctly" )
   {
      x = 5, 1;

      REQUIRE( x.getValue() == 5 );
      REQUIRE( x.getDiffValue() == 1 );
   }

   SECTION( "SingleDiff of polynomial evaluation and exp function deliver correct results" )
   {
      spline::SimplePolynomial<5, double> poly;

      std::mt19937 e1( 1422822953 );  //seed chosen randomly :)

      std::uniform_real_distribution<double> uniform_dist( -1, 1 );

      for( int i = 0; i < 100; ++i )
      {
         x = uniform_dist( e1 ), 1;
         poly.setCoeff( 0,  uniform_dist( e1 ) );
         poly.setCoeff( 1,  uniform_dist( e1 ) );
         poly.setCoeff( 2,  uniform_dist( e1 ) );
         poly.setCoeff( 3,  uniform_dist( e1 ) );
         poly.setCoeff( 4,  uniform_dist( e1 ) );
         poly.setCoeff( 5,  uniform_dist( e1 ) );
         cpplsq::SingleDiff<double> y = poly( x );
         double yval =  poly( x.getValue() );
         double ydval = poly.derivative<1> ( x.getValue() );
         REQUIRE( y.getValue() == Approx( yval ) );
         REQUIRE( y.getDiffValue() == Approx( ydval ) );
         y = exp( y );
         REQUIRE( y.getValue() == Approx( exp( yval ) ) );
         REQUIRE( y.getDiffValue() == Approx( exp( yval ) *ydval ) );
      }
   }





}
