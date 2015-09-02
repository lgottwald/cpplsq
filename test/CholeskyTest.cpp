#include <catch/catch.hpp>
#include <cpplsq/cholesky_solve.hpp>
#include <iostream>

TEST_CASE( "cholesky decomposition works correctly", "[cpplsq]" )
{
   auto A = simd::alloc_aligned_array<double>( 3 * 3 );
   auto L = simd::alloc_aligned_array<double>( 3 * 3 );
   auto R = simd::alloc_aligned_array<double>( 3 * 3 );

   //store lower part of
   //   4  12 -16
   //  12  37 -43
   // -16 -43  98
   // in A
   A[0] = 4;
   //A[1] = 12;
   //A[2] = -16;
   A[3] = 12;
   A[4] = 37;
   //A[5] = -43;
   A[6] = -16;
   A[7] = -43;
   A[8] = 98;

   std::copy( A.get(), A.get() + 9, L.get() );


   auto x = simd::alloc_aligned_array<double>( 3 );
   auto b = simd::alloc_aligned_array<double>( 3 );
   x[0] = 4;
   x[1] = 5;
   x[2] = 6;

   // b = A * x
   cpplsq::blas::symv( 3, 1.0, A.get(), 3, x.get(), 1, 0.0, b.get(), 1 );
   bool OK = false;

   //now b differs from x
   for( int i = 0; i < 3; ++i )
      OK |= ( x[i] != Approx( b[i] ) );

   REQUIRE( OK );
   //solve A * u = b and store u into b and the lower part of A's cholesky decomp into L
   cpplsq::cholesky_solve( L, 3, b, 3 );

   OK = false;

   //now L should differ from A
   for( int i = 0; i < 3; ++i )
   {
      for( int j = 0; j <= i; ++j )
      {
         OK |= ( A[i * 3 + j] != Approx( L[i * 3 + j] ) );
      }
   }

   REQUIRE( OK );

   //now recompute A (if cholesky worked A*A^T is again the original A)
   //now recompute A (if cholesky worked A*A^T is again the original A)

   //copy L into R
   std::copy( L.get(), L.get() + 9, R.get() );

   //compute L=L*L^T
   cpplsq::blas::trmm( CblasRight, CblasTrans, CblasNonUnit, 3, 3, 1.0, R.get(), 3, L.get(), 3 );

   //now b should equal x and L should equal A
   for( int i = 0; i < 3; ++i )
   {
      REQUIRE( b[i] == Approx( x[i] ) );

      for( int j = 0; j <= i; ++j )
      {
         REQUIRE( A[i * 3 + j] == Approx( L[i * 3 + j] ) );
      }
   }

}
