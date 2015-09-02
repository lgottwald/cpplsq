#include <catch/catch.hpp>
#include <cpplsq/MultiDiff.hpp>
#include <random>
#include "Rosenbrock.hpp"

TEST_CASE( "Multivariate differentiation works correctly", "[cpplsq]" )
{
   std::vector<double> x( 10 );
   cpplsq::MultiDiff<double>::Context ctx( x.size() );


   std::mt19937 e1( 1422822953 );  //seed chosen randomly :)

   std::uniform_real_distribution<double> uniform_dist( -10, 10 );

   for( int i = 0; i < 100; ++i )
   {
      for( std::size_t i = 0; i < x.size(); ++i )
         x[i] = uniform_dist( e1 );

      std::vector<cpplsq::MultiDiff<double>> xad = cpplsq::Independent( x.begin(), x.end() );

      double y = rosen_brock( x.data(), x.size() );
      std::vector<double> yd = rosen_brock_deriv( x );

      cpplsq::MultiDiff<double> ady = rosen_brock( xad.data() , xad.size() );

      REQUIRE( y == Approx( ady.getValue() ) );

      for( std::size_t i = 0; i < yd.size(); ++i )
      {
         REQUIRE( yd[i] == Approx( ady.getDiffValues() [i] ) );
      }

   }
}
