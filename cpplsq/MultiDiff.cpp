#include "MultiDiff.hpp"
#include <simd/alloc.hpp>

#ifndef PARALLEL
#define PARALLEL 1
#endif

#if PARALLEL
#define THREAD_LOCAL thread_local
#include <atomic>
#else
#define THREAD_LOCAL
#endif


using std::size_t;
using std::ptrdiff_t;

constexpr static size_t block_size()
{
   return 4096;
}

struct Block
{
   Block *next;
   size_t countdown;

   static constexpr size_t offset()
   {
      return simd::next_size<char>( sizeof( Block ) );
   }

   char *as_char_ptr()
   {
      return reinterpret_cast<char *>( this );
   }
};


class FreeList
{
public:

   Block *head()
   {
      return list;
   }

   Block *pop()
   {
      Block *head = list;
#if PARALLEL

      while( head && !list.compare_exchange_weak( head, head->next ) );

      return head;
#else

      if( head )
         list = list->next;

      return head;
#endif
   }

   void push( Block *new_head )
   {
#if PARALLEL
      Block *head = list.load();

      do
      {
         new_head->next = head;
      }
      while( !list.compare_exchange_weak( head, new_head ) );

#else
      new_head->next = list;
      list = new_head;
#endif
   }

private:
#if PARALLEL
   using BlockPtr = std::atomic<Block *>;
#else
   using BlockPtr = Block *;
#endif

   BlockPtr list { nullptr };
};

class BlockList
{
public:

   Block *head()
   {
      return list;
   }

   void release_head()
   {
      Block *release = list;
      list = list->next;
      free_blocks.push( release );
   }

   void release_after( Block *blk )
   {
      if( blk )
      {
         Block *release = blk->next;
         blk->next = release->next;
         free_blocks.push( release );
      }
      else
      {
         release_head();
      }
   }

   Block *get_new_block()
   {
      Block *new_block = free_blocks.pop();

      if( !new_block )
      {
         new_block = ( Block * ) simd::cache_aligned_alloc( block_size() );
      }

      new_block->next = list;
      list = new_block;
      return new_block;
   }

   void release_all()
   {
      while( list )
         release_head();
   }

   void free_all()
   {
      Block *head;

      while( list )
      {
         head = list;
         list = head->next;
         simd::cache_aligned_free( head );
      }

      while( ( head = free_blocks.pop() ) )
         simd::cache_aligned_free( head );
   }

   ~BlockList()
   {
      release_all();
   }

private:
   static FreeList free_blocks;
   Block *list = nullptr;
};

FreeList BlockList::free_blocks;
THREAD_LOCAL static BlockList used_blocks;
THREAD_LOCAL static char *next_buffer = nullptr;
THREAD_LOCAL static char *block_end = nullptr;

namespace cpplsq
{
namespace internal
{

size_t buffer_size;
size_t num_directions;

void free_all()
{
   used_blocks.free_all();
   next_buffer = block_end = nullptr;
   buffer_size = 0;
   num_directions = 0;
}

void *new_buffer()
{
   if( next_buffer == block_end )
   {
      Block *new_block = used_blocks.get_new_block();
      next_buffer = new_block->as_char_ptr() + Block::offset();
      new_block->countdown = ( block_size() - Block::offset() ) / buffer_size;
      block_end = next_buffer + new_block->countdown * buffer_size;
   }

   char *buf = next_buffer;
   next_buffer += buffer_size;
   return buf;
}


void release_buffer( void *buf )
{
   if( buf )
   {
      Block *prev = nullptr;
      Block *blk = used_blocks.head();

      while( true )
      {
         assert( blk );
         ptrdiff_t distance = reinterpret_cast<char *>( buf ) - blk->as_char_ptr();

         //if the buffer belongs to the block
         if( distance >= 0 && size_t( distance ) < block_size() )
         {
            //decrement countdown and check if it is zero
            if( --( blk->countdown ) == 0 )
            {
               //block is not used anymore -> release it
               used_blocks.release_after( prev );
            }

            return;
         }

         prev = blk;
         blk = blk->next;
      }

   }
}

}//internal
}//cpplsq
