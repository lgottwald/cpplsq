#include <catch/catch.hpp>
#include <cpplsq/gn_sbfgs_min.hpp>
#include "Rosenbrock.hpp"

struct RosenbrockResidual
{
   RosenbrockResidual( int N ) : N( N ) {}

   template<typename REAL>
   REAL operator()( const REAL *params )
   {
      return rosen_brock( params, N );
   }
private:
   int N;

};


struct Residual
{
   Residual( double x, double y ) : x( x ), y( y ) {}

   template<typename REAL >
   REAL operator()( const REAL *params )
   {
      return y - ( params[0] * exp( -params[1] * x ) + params[2] );
   }
private:
   double x;
   double y;
};


TEST_CASE( "Test of least squares routine with rosenbrock function", "[cpplsq]" )
{
   std::vector<RosenbrockResidual> r {RosenbrockResidual( 3 )};
   simd::aligned_vector<double> x( 3 );
   auto seed = 2724251330;

   std::mt19937 e1( seed );
   std::uniform_real_distribution<double> uniform_dist( -20, 20 );

   x[0] = uniform_dist( e1 );
   x[1] = uniform_dist( e1 );
   x[2] = uniform_dist( e1 );

   //not starting at optimum
   REQUIRE( std::abs( x[0] - 1 ) > 2 );
   REQUIRE( std::abs( x[1] - 1 ) > 2 );
   REQUIRE( std::abs( x[2] - 1 ) > 2 );

   cpplsq::gn_sbfgs_min( 1e-9, x, r );

   //approximately found optimum
   REQUIRE( x[0] == Approx( 1 ).epsilon( 1e-3 ) );
   REQUIRE( x[1] == Approx( 1 ).epsilon( 1e-3 ) );
   REQUIRE( x[2] == Approx( 1 ).epsilon( 1e-3 ) );

}


TEST_CASE( "Test of least squares routine with exponential decay", "[cpplsq]" )
{
   simd::aligned_vector<double> x( 3 );
   auto seed = 3256271490;


   std::mt19937 e1( seed );  //seed chosen randomly :)
   std::uniform_real_distribution<double> uniform_dist( 0.1, 10 );


   double p0 = uniform_dist( e1 );
   double p1 = uniform_dist( e1 );
   double p2 = uniform_dist( e1 );

   std::cout << "original params = ( ";
   std::cout << p0 << " ";
   std::cout << p1 << " ";
   std::cout << p2 << " ";
   std::cout << ")\n";

   for( std::size_t i = 0; i < x.size(); ++i )
      x[i] = uniform_dist( e1 );

   std::cout << "start x = ( ";

   for( std::size_t i = 0;  i < x.size(); ++i )
      std::cout << x[i] << " ";

   std::cout << ")\n";

   std::vector<Residual> r;

   int K = 10000;
   double a = 0.1;
   double b = 20;
   std::uniform_real_distribution<double> disturb( -0.1, 0.1 );

   for( int i = 0; i < K; ++i )
   {
      double x = a + ( i * ( b - a ) ) / K;
      double y = disturb( e1 ) + ( p0 * exp( -p1 * x ) + p2 );
      r.emplace_back( x, y );
   }

   cpplsq::gn_sbfgs_min( 1e-8, x, r );

   std::cout << "found x = ( ";

   for( std::size_t i = 0;  i < x.size(); ++i )
      std::cout << x[i] << " ";

   std::cout << ")\n";

   REQUIRE( x[0] == Approx( p0 ).epsilon( 0.1 ) );
   REQUIRE( x[1] == Approx( p1 ).epsilon( 0.1 ) );
   REQUIRE( x[2] == Approx( p2 ).epsilon( 0.1 ) );

}
