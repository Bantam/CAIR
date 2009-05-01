
//=========================================================================================================//
//CAIR - Content Aware Image Resizer
//Copyright (C) 2008 Joseph Auman (brain.recall@gmail.com)

//=========================================================================================================//
//This library is free software; you can redistribute it and/or
//modify it under the terms of the GNU Lesser General Public
//License as published by the Free Software Foundation; either
//version 2.1 of the License, or (at your option) any later version.
//This library is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//Lesser General Public License for more details.
//You should have received a copy of the GNU Lesser General Public
//License along with this library; if not, write to the Free Software
//Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

//=========================================================================================================//
//This thing should hopefully perform the image resize method developed by Shai Avidan and Ariel Shamir.

//=========================================================================================================//
//TODO (maybe):
//  - Add an Energy_Middle() function to allow more than two energy map threads.
//  - Try doing Poisson image reconstruction instead of the averaging technique in CAIR_HD() if I can figure it out (see the ReadMe).
//  - Abstract out pthreads into macros allowing for multiple thread types to be used (ugh, not for a while at least)
//  - Maybe someday push CAIR into OO land and create a class out of it (pff, OO is the devil!).

//=========================================================================================================//
//KNOWN BUGS:
//  - The threading changes in v2.16 lost the reentrant capability of CAIR. (If this hits someone hard, let me know.)
//  - The percent of completion for the CAIR_callback in CAIR_HD and CAIR_Removal are often wrong.

//=========================================================================================================//
//CHANGELOG:
//CAIR v2.17 Changelog:
//  - Ditched vectors for dynamic arrays, for about a 15% performance boost.
//  - Added some headers into CAIR_CML.h to fix some compilier errors with new versions of g++. (Special thanks to Alexandre Prokoudine)
//  - Added CAIR_Threads(), which allows the ability to dynamically change the number of threads CAIR will use.
//    NOTE: Don't ever call CAIR_Threads() while CAIR() is processing an image, unless you're a masochist.
//  - Added CAIR_callback parameters to CAIR(), CAIR_Removal(), and CAIR_HD(). This function pointer will be called every cycle,
//    passing the function the percent complete (0 to 1). If the function returns false, then the resize is canceled.
//    Then, CAIR(), CAIR_Removal(), and CAIR_HD() would also return a false, leaving the destination image/weights in an unknown state.
//    Set to NULL if this does not need to be used.
//CAIR v2.16 Changelog:
//  - A long overdue overhaul of the threading system yielded about a 40% performance boost. All threads, semaphores, and mutexes are kept around for
//    as long as possible, instead of destroying them every step.
//  - I stumbled across a bug in CAIR_Add() where some parts of the artifical weight matrix weren't being updated, creating slightly incorrect images.
//  - Comment overhauls to reflect the new threading.
//CAIR v2.15.1 Changelog:
//  - Mutexes and conditions used in Energy_Map() are now properly destroyed, fixing a serious memory leak. This was discovered when
//    processing huge images (7500x3800) that would cause the process to exceed the 32-bit address space.
//    Special thanks to Klaus Nordby for hitting this bug. CAIR has now been tested up to 9800x7800 without issue.
//  - A potential memory leak in Deallocate_Matrix() in the CML was squashed.
//  - By default CAIR now uses 4 threads (yay quad-core!).
//CAIR v2.15 Changelog:
//  - Added the new forward energy algorithm. A new CAIR_energy parameter determines the type of energy function to use. Forward energy
//    produces less artifacts in most images, but comes at a 5% performance cost. Thanks to Matt Newel for pointing it out.
//    Read the paper on it here: http://www.faculty.idc.ac.il/arik/papers/vidRet.pdf
//  - The number of threads CAIR uses can now be set by CAIR_NUM_THREADS. This currently does not apply to the Energy calculations.
//    On my dual-core system, I netted a 5% performance boost by reducing thread count from 4 to 2.
//  - Separate destination weights for the resize to standardize the interface.
//  - Removed "namespace std" from source headers. Special thanks to David Oster.
//  - Removed the clipping Get() method from the CML. This makes the CML generic again.
//  - Comment clean-ups.
//  - Comments have been spell-checked. Apparently, I don’t speel so good. (thanks again to David Oster)
//CAIR v2.14 Changelog:
//  - CAIR has been relicensed under the LGPLv2.1
//CAIR v2.13 Changelog:
//  - Added CAIR_Image_Map() and CAIR_Map_Resize() to allow for "content-aware multi-size images." Now it just needs to get put into a
//    file-format.
//  - CAIR() and CAIR_HD() now properly copy the Source to Dest when no resize is done.
//  - Fixed a bug in CAIR_HD(), Energy/TEnergy confusion.
//  - Fixed a compiler warning in main().
//  - Changed in Remove_Quadrant() "pixel remove" into "int remove"
//  - Comment updates and I decided to bring back the tabs (not sure why I got rid of them).
//CAIR v2.12 Changelog:
//  - About 20% faster across the board.
//  - Unchanged portions of the energy map are now retained. Special thanks to Jib for that (remind me to ask him how it works :-) ).
//  - Add_Edge() and Remove_Edge() now update the Edge in UNSAFE mode when able.
//  - The CML now has a CML_DEBUG mode to let the developers know when they screwed up.
//  - main() now displays the runtime with three decimal places for better accuracy. Special thanks to Jib.
//  - Various comment updates.
//CAIR v2.11 Changelog: (The Super-Speedy Jib version)
//  - 40% speed boost across the board with "high quality"
//  - Remove_Path() and Add_Path() directly recalculate only changed edge values. This gives the speed of low quality while
//      maintaining high quality output. Because of this, the quality factor is no longer used and has been removed. (Special thanks to Jib)
//  - Grayscale values during a resize are now properly recalculated for better accuracy.
//  - main() has undergone a major overhaul. Now most operations are accessible from the CLI. (Special thanks to Jib)
//  - Now uses multiple edge detectors, with V_SQUARE offering some of the best quality. (Special thanks to Jib)
//  - Minor change to Grayscale_Pixel() to increase speed. (Special thanks to Jib)
//CAIR v2.10 Changelog: (The great title of v3.0 is when I have CAIR_HD() using Poisson reconstruction, a ways away...)
//  - Removed multiple levels of derefrencing in all the thread functions for a 15% speed boost across the board.
//  - Changed the way CAIR_Removal() works for more flexibility and better operation.
//  - Fixed a bug in CAIR_Removal(): infinite loop problem that got eliminated with its new operation
//  - Some comment updates.
//CAIR v2.9 Changelog:
//  - Row-majorized and multi-threaded Add_Weights(), which gave a 40% speed boost while enlarging.
//  - Row-majorized Edge_Detect() (among many other functions) which gave about a 10% speed boost with quality == 1.
//  - Changed CML_Matrix::Resize_Width() so it gracefully handles enlarging beyond the Reserve()'ed max.
//  - Changed Energy_Path() to return a long instead of int, just in case.
//  - Fixed an enlarging bug in CAIR_Add() created in v2.8.5
//CAIR v2.8.5 Changelog:
//  - Added CAIR_HD() which, at each step, determines if the vertical path or the horizontal path has the least energy and then removes it.
//  - Changed Energy_Path() so it returns the total energy of the minimum path.
//  - Cleaned up unnecessary allocation of some CML objects.
//  - Fixed a bug in CML_Matrix:Shift_Row(): bounds checking could cause a shift when one wasn't desired
//  - Fixed a bug in Remove_Quadrant(): horrible bounds checking
//CAIR v2.8 Changelog:
//  - Now 30% faster across the board.
//  - Added CML_Matrix::Shift_Row() which uses memmove() to shift elements in a row of the matrix. Special thanks again to Brett Taylor
//      for helping me debug it.
//  - Add_Quadrant() and Remove_Quadrant() now use the CML_Matrix::Shift_Row() method instead of the old loops. They also specifically
//      handle their own bounds checking for assignments.
//  - Removed all bounds checking in CML_Matrix::operator() for a speed boost.
//  - Cleaned up some CML functions to directly use the private data instead of the class methods.
//  - CML_Matrix::operator=() now uses memcpy() for a speed boost, especially on those larger images.
//  - Fixed a bug in CAIR_Grayscale(), CAIR_Edge(), and the CAIR_V/H_Energy() functions: forgot to clear the alpha channel.
//  - De-tabbed a few more functions
//CAIR v2.7 Changelog:
//  - CML has gone row-major, which made the CPU cache nice and happy. Another 25% speed boost. Unfortunately, all the crazy resizing issues
//      from v2.5 came right back, so be careful when using CML_Matrix::Resize_Width() (enlarging requires a Reserve()).
//CAIR v2.6.2 Changelog:
//  - Made a ReadMe.txt and Makefile for the package
//  - De-tabbed the source files
//  - Comment updates
//  - Forgot a left-over Temp object in CAIR_Add()
//CAIR v2.6.1 Changelog:
//  - Fixed a memory leak in CML_Matrix::Resize_Width()
//CAIR v2.6 Changelog:
//  - Eliminated the copying into a temp matrix in CAIR_Remove() and CAIR_Add(). Another 15% speed boost.
//  - Fixed the CML resizing so its more logical. No more need for Reserve'ing memory.
//CAIR v2.5 Changelog:
//  - Now 35% faster across the board.
//  - CML has undergone a major rewrite. It no longer uses vectors as its internal storage. Because of this, its resizing functions
//      have severe limitations, so please read the CML comments if you plan to use them. This gave about a 30% performance boost.
//  - Optimized Energy_Map(). Now uses two specialized threading functions. About a 5% boost.
//  - Optimized Remove_Path() to give another boost.
//  - Energy is no longer created and destroyed in Energy_Path(). Gave another boost.
//  - Added CAIR_H_Energy(), which gives the horizontal energy of an image.
//  - Added CAIR_Removal(), which performs (experimental) automatic object removal. It counts the number of negative weight rows/columns,
//      then removes the least amount in that direction. It'll check to make sure it got rid of all negative areas, then it will expand
//      the result back out to its original dimensions.
//CAIR v2.1 Changelog:
//  - Unrolled the loops for Convolve_Pixel() and changed the way Edge_Detect() works. Optimizing that gave ANOTHER 25% performance boost
//      with quality == 1.
//  - inline'ed and const'ed a few accessor functions in the CML for a minor speed boost.
//  - Fixed a few cross-compiler issues; special thanks to Gabe Rudy.
//  - Fixed a few more comments.
//  - Forgot to mention, removal of all previous CAIR_DEBUG code. Most of it is in the new CAIR_Edge() and CAIR_Energy() anyways...
//CAIR v2.0 Changelog:
//  - Now 50% faster across the board.
//  - EasyBMP has been replaced with CML, the CAIR Matrix Library. This gave speed improvements and code standardization.
//      This is such a large change it has affected all functions in CAIR, all for the better. Reference objects have been
//      replaced with standard pointers.
//  - Added three new functions: CAIR_Grayscale(), CAIR_Edge(), and CAIR_Energy(), which give the grayscale, edge detection,
//      and energy maps of a source image.
//  - Add_Path() and Remove_Path() now maintain Grayscale during resizing. This gave a performance boost with no real 
//      quality reduction; special thanks to Brett Taylor.
//  - Edge_Detect() now handles the boundaries separately for a performance boost.
//  - Add_Path() and Remove_Path() no longer refill unchanged portions of an image since CML Resize's are no longer destructive.
//  - CAIR_Add() now Reserve's memory for the vectors in CML to prevent memory thrashing as they are enlarged.
//  - Fixed another adding bug; new paths have their own artificial weight
//CAIR v1.2 Changelog:
//  - Fixed ANOTHER adding bug; now works much better with < 1 quality
//  - a few more comment updates
//CAIR v1.1 Changelog:
//  - Fixed a bad bug in adding; averaging the wrong pixels
//  - Fixed a few incorrect/outdated comments
//CAIR v1.0 Changelog:
//  - Path adding now working with reasonable results; special thanks to Ramin Sabet
//  - Add_Path() has been multi-threaded
//CAIR v0.5 Changelog:
//  - Multi-threaded Energy_Map() and Remove_Path(), gave another 30% speed boost with quality = 0
//  - Fixed a few compiler warnings when at level 4 (just stuff in the CAIR_DEBUG code)
//=========================================================================================================//

#include "CAIR.h"
#include "CAIR_CML.h"
#include <cmath> //for abs(), floor()
#include <limits> //for max int
#include <pthread.h>
#include <semaphore.h>

using namespace std;

//=========================================================================================================//
//Thread parameters
struct Thread_Params
{
	//Image Parameters
	CML_color * Source;
	CML_int * D_Weights;
	CAIR_convolution conv;
	CAIR_energy ener;
	int add_weight;
	//Internal Stuff
	int * Path;
	pthread_mutex_t * Mine; //used only for energy threads
	pthread_mutex_t * Not_Mine;
	CML_int * Energy_Map;
	CML_int * Edge;
	CML_gray * Gray;
	CML_int * Add_Weight;
	CML_int * Sum_Weight;
	//Thread Parameters
	int top_y;
	int bot_y;
	int top_x;
	int bot_x;
	bool exit; //flag causing the thread to exit
};

//=========================================================================================================//
//Thread Info
Thread_Params * thread_info;

//Thread Handles
pthread_t * remove_threads;
pthread_t * edge_threads;
pthread_t * gray_threads;
pthread_t * add_threads;
pthread_t energy_threads[2]; //these are limited to only two
int num_threads = CAIR_NUM_THREADS;

//Thread Semaphores
sem_t remove_sem[3]; //start, edge_start, finish
sem_t add_sem[4]; //add_start, start, edge_start, finish
sem_t edge_sem[2]; //start, finish
sem_t gray_sem[2]; //start, finish
sem_t energy_sem[5]; //start_left, start_right, locks_done, good_to_go, finish

//early declarations on the threading functions
void Startup_Threads();
void Resize_Threads( int height );
void Shutdown_Threads();
//energy thread mutexes. these arrays will be created in Resize_Threads()
pthread_mutex_t * Left_Mutexes = NULL;
pthread_mutex_t * Right_Mutexes = NULL;
int mutex_height = 0;

//=========================================================================================================//
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

//=========================================================================================================//
//==                                          G R A Y S C A L E                                          ==//
//=========================================================================================================//

//=========================================================================================================//
//Performs a RGB->YUV type conversion (we only want Y', the luma)
inline CML_byte Grayscale_Pixel( CML_RGBA * pixel )
{
	return (CML_byte)floor( ( 299 * pixel->red +
							  587 * pixel->green +
							  114 * pixel->blue ) / 1000.0 );
}

//=========================================================================================================//
//Our thread function for the Grayscale
void * Gray_Quadrant( void * id )
{
	int num = (int)id;

	while( true )
	{
		//wait for the thread to get a signal to start
		sem_wait( &(gray_sem[0]) );

		//get updated parameters
		Thread_Params gray_area = thread_info[num];

		if( gray_area.exit == true )
		{
			//thread is exiting
			break;
		}

		CML_byte gray = 0;

		for( int y = gray_area.top_y; y < gray_area.bot_y; y++ )
		{
			for( int x = 0; x < (*(gray_area.Source)).Width(); x++ )
			{
				gray = Grayscale_Pixel( &(*(gray_area.Source))(x,y) );

				(*(gray_area.Gray))(x,y) = gray;
			}
		}

		//signal we're done
		sem_post( &(gray_sem[1]) );
	}

	return NULL;
} //end Gray_Quadrant()

//=========================================================================================================//
//Sort-of does a RGB->YUV conversion (actually, just RGB->Y)
//Multi-threaded with each thread getting a stirp across the image.
void Grayscale_Image( CML_color * Source, CML_gray * Dest )
{
	int thread_height = (*Source).Height() / num_threads;

	//setup parameters
	for( int i = 0; i < num_threads; i++ )
	{
		thread_info[i].Source = Source;
		thread_info[i].Gray = Dest;
		thread_info[i].top_y = i * thread_height;
		thread_info[i].bot_y = thread_info[i].top_y + thread_height;
	}

	//have the last thread pick up the slack
	thread_info[num_threads-1].bot_y = (*Source).Height();

	//startup the threads
	for( int i = 0; i < num_threads; i++ )
	{
		sem_post( &(gray_sem[0]) );
	}

	//now wait for them to come back to us
	for( int i = 0; i < num_threads; i++ )
	{
		sem_wait( &(gray_sem[1]) );
	}

} //end Grayscale_Image()

//=========================================================================================================//
//==                                                 E D G E                                             ==//
//=========================================================================================================//
enum edge_safe { SAFE, UNSAFE };

//=========================================================================================================//
//returns the convolution value of the pixel Source[x][y] with one of the kernels.
//Several kernels are avaialable, each with their strengths and weaknesses. The edge_safe
//param will use the slower, but safer Get() method of the CML.
int Convolve_Pixel( CML_gray * Source, int x, int y, edge_safe safety, CAIR_convolution convolution)
{
	int conv = 0;

	switch( convolution )
	{
	case PREWITT:
		if( safety == SAFE )
		{
			conv = abs( (*Source).Get(x+1,y+1) + (*Source).Get(x+1,y) + (*Source).Get(x+1,y-1) //x part of the prewitt
					   -(*Source).Get(x-1,y-1) - (*Source).Get(x-1,y) - (*Source).Get(x-1,y+1) ) +
				   abs( (*Source).Get(x+1,y+1) + (*Source).Get(x,y+1) + (*Source).Get(x-1,y+1) //y part of the prewitt
					   -(*Source).Get(x+1,y-1) - (*Source).Get(x,y-1) - (*Source).Get(x-1,y-1) );
		}
		else
		{
			conv = abs( (*Source)(x+1,y+1) + (*Source)(x+1,y) + (*Source)(x+1,y-1) //x part of the prewitt
					   -(*Source)(x-1,y-1) - (*Source)(x-1,y) - (*Source)(x-1,y+1) ) +
				   abs( (*Source)(x+1,y+1) + (*Source)(x,y+1) + (*Source)(x-1,y+1) //y part of the prewitt
					   -(*Source)(x+1,y-1) - (*Source)(x,y-1) - (*Source)(x-1,y-1) );
		}
		break;

	 case V_SQUARE:
		if( safety == SAFE )
		{
			conv = (*Source).Get(x+1,y+1) + (*Source).Get(x+1,y) + (*Source).Get(x+1,y-1) //x part of the prewitt
				  -(*Source).Get(x-1,y-1) - (*Source).Get(x-1,y) - (*Source).Get(x-1,y+1);
			conv *= conv;
		}
		else
		{
			conv = (*Source)(x+1,y+1) + (*Source)(x+1,y) + (*Source)(x+1,y-1) //x part of the prewitt
				  -(*Source)(x-1,y-1) - (*Source)(x-1,y) - (*Source)(x-1,y+1);
			conv *= conv;
		}
		break;

	 case V1:
		if( safety == SAFE )
		{
			conv =  abs( (*Source).Get(x+1,y+1) + (*Source).Get(x+1,y) + (*Source).Get(x+1,y-1) //x part of the prewitt
						-(*Source).Get(x-1,y-1) - (*Source).Get(x-1,y) - (*Source).Get(x-1,y+1) );
		}
		else
		{
			conv = abs( (*Source)(x+1,y+1) + (*Source)(x+1,y) + (*Source)(x+1,y-1) //x part of the prewitt
					   -(*Source)(x-1,y-1) - (*Source)(x-1,y) - (*Source)(x-1,y+1) ) ;
		}
		break;
	
	 case SOBEL:
		if( safety == SAFE )
		{
			conv = abs( (*Source).Get(x+1,y+1) + (2 * (*Source).Get(x+1,y)) + (*Source).Get(x+1,y-1) //x part of the sobel
					   -(*Source).Get(x-1,y-1) - (2 * (*Source).Get(x-1,y)) - (*Source).Get(x-1,y+1) ) +
				   abs( (*Source).Get(x+1,y+1) + (2 * (*Source).Get(x,y+1)) + (*Source).Get(x-1,y+1) //y part of the sobel
					   -(*Source).Get(x+1,y-1) - (2 * (*Source).Get(x,y-1)) - (*Source).Get(x-1,y-1) );
		}
		else
		{
			conv = abs( (*Source)(x+1,y+1) + (2 * (*Source)(x+1,y)) + (*Source)(x+1,y-1) //x part of the sobel
					   -(*Source)(x-1,y-1) - (2 * (*Source)(x-1,y)) - (*Source)(x-1,y+1) ) +
				   abs( (*Source)(x+1,y+1) + (2 * (*Source)(x,y+1)) + (*Source)(x-1,y+1) //y part of the sobel
					   -(*Source)(x+1,y-1) - (2 * (*Source)(x,y-1)) - (*Source)(x-1,y-1) );
		}
		break;

	case LAPLACIAN:
		if( safety == SAFE )
		{
			conv = abs( (*Source).Get(x+1,y) + (*Source).Get(x-1,y) + (*Source).Get(x,y+1) + (*Source).Get(x,y-1)
					   -(4 * (*Source).Get(x,y)) );
		}
		else
		{
			conv = abs( (*Source)(x+1,y) + (*Source)(x-1,y) + (*Source)(x,y+1) + (*Source)(x,y-1)
					   -(4 * (*Source)(x,y)) );
		}
		break;
	}
	return conv;
}

//=========================================================================================================//
//The thread function, splitting the image into strips
void * Edge_Quadrant( void * id )
{
	int num = (int)id;

	while( true )
	{
		sem_wait( &(edge_sem[0]) );

		//get updated parameters
		Thread_Params edge_area = thread_info[num];

		if( edge_area.exit == true )
		{
			//thread is exiting
			break;
		}

		for( int y = edge_area.top_y; y < edge_area.bot_y; y++ )
		{
			//left most edge
			(*(edge_area.Edge))(0,y) = Convolve_Pixel( edge_area.Gray, 0, y, SAFE, edge_area.conv );

			//fill in the middle
			for( int x = 1; x < (*(edge_area.Gray)).Width() - 1; x++ )
			{
				(*(edge_area.Edge))(x,y) = Convolve_Pixel( edge_area.Gray, x, y, UNSAFE, edge_area.conv );
			}

			//right most edge
			(*(edge_area.Edge))((*(edge_area.Gray)).Width()-1,y) = Convolve_Pixel( edge_area.Gray, (*(edge_area.Gray)).Width()-1, y, SAFE, edge_area.conv);
		}

		//signal we're done
		sem_post( &(edge_sem[1]) );
	}

	return NULL;
}

//=========================================================================================================//
//Performs full edge detection on Source with one of the kernels.
void Edge_Detect( CML_gray * Source, CML_int * Dest, CAIR_convolution conv )
{
	//There is no easy solution to the boundries. Calling the same boundry pixel to convolve itself against seems actually better
	//than padding the image with zeros or 255's.
	//Calling itself induces a "ringing" into the near edge of the image. Padding can lead to a darker or lighter edge.
	//The only "good" solution is to have the entire one-pixel wide edge not included in the edge detected image.
	//This would reduce the size of the image by 2 pixels in both directions, something that is unacceptable here.

	int thread_height = (*Source).Height() / num_threads;

	//setup parameters
	for( int i = 0; i < num_threads; i++ )
	{
		thread_info[i].Gray = Source;
		thread_info[i].Edge = Dest;
		thread_info[i].top_y = (i * thread_height) + 1; //handle very top row down below
		thread_info[i].bot_y = thread_info[i].top_y + thread_height;
		thread_info[i].conv = conv;
	}

	//have the last thread pick up the slack
	thread_info[num_threads-1].bot_y = (*Source).Height() - 1; //handle very bottom row down below

	//create the threads
	for( int i = 0; i < num_threads; i++ )
	{
		sem_post( &(edge_sem[0]) );
	}

	//while those are running we can go back and do the boundry pixels with the extra safety checks
	for( int x = 0; x < (*Source).Width(); x++ )
	{
		(*Dest)(x,0) = Convolve_Pixel( Source, x, 0, SAFE, conv );
		(*Dest)(x,(*Source).Height()-1) = Convolve_Pixel( Source, x, (*Source).Height()-1, SAFE, conv );
	}

	//now wait on them
	for( int i = 0; i < num_threads; i++ )
	{
		sem_wait( &(edge_sem[1]) );
	}

} //end Edge_Detect()

//=========================================================================================================//
//==                                           E N E R G Y                                               ==//
//=========================================================================================================//

//=========================================================================================================//
//Simple fuction returning the minimum of three values.
inline int min_of_three( int x, int y, int z )
{
	int min = y;

	if( x < min )
	{
		min = x;
	}
	if( z < min )
	{
		return z;
	}

	return min;
}

//=========================================================================================================//
//Get the value from the integer matrix, return a large value if out-of-bounds in the x-direction.
inline int Get_Max( CML_int * Energy, int x, int y )
{
	if( ( x < 0 ) || ( x >= (*Energy).Width() ) )
	{
		return std::numeric_limits<int>::max();
	}
	else
	{
		return (*Energy).Get( x, y );
	}
}

//=========================================================================================================//
//This calculates a minimum energy path from the given start point (min_x) and the energy map.
//Note: Path better be of proper size.
void Generate_Path( CML_int * Energy, int min_x, int * Path )
{
	int min;
	int x = min_x;
	for( int y = (*Energy).Height() - 1; y >= 0; y-- ) //builds from bottom up
	{
		min = x; //assume the minimum is straight up

		if( Get_Max( Energy, x-1, y ) < Get_Max( Energy, min, y ) ) //check to see if min is up-left
		{
			min = x - 1;
		}
		if( Get_Max( Energy, x+1, y ) < Get_Max( Energy, min, y) ) //up-right
		{
			min = x + 1;
		}
		
		Path[y] = min;
		x = min;
	}
}

//=========================================================================================================//
//Forward energy cost functions. These are additional energy values for the left, up, and right seam paths.
//See the paper "Improved Seam Carving for Video Retargeting" by Michael Rubinstein, Ariel Shamir, and Shai  Avidan.
inline int Forward_CostL( CML_int * Edge, int x, int y )
{
	return (abs((*Edge)(x+1,y) - (*Edge)(x-1,y)) + abs((*Edge)(x,y-1) - (*Edge)(x-1,y)));
}

inline int Forward_CostU( CML_int * Edge, int x, int y )
{
	return (abs((*Edge)(x+1,y) - (*Edge)(x-1,y)));
}

inline int Forward_CostR( CML_int * Edge, int x, int y )
{
	return (abs((*Edge)(x+1,y) - (*Edge)(x-1,y)) + abs((*Edge)(x,y-1) - (*Edge)(x+1,y)));
}

//=========================================================================================================//
//threading procedure for Energy Map
//-main signals to start left
//-main waits for left to signal locks_done
//+left starts and locks mutexes
//+left signals main locks_done
//+left waits for good_to_go
//-main starts and signals start right
//-main waits for right to signal locks_done
//#right starts and lock mutexes
//#right signals main locks_done
//#right waits for good_to_go
//-main starts and signals good_to_go twice
//-main waits for finish twice
//+#left and right start and do there thing
//#+left and right signal finish when done
//+#left and right wait for thier start
//-main starts and continues on

//The reason for the crazy mutex locking is because I need to evenly split the matrix in half for each thread.
//So, for the boundry between the two threads, they will try to access a value that the other thread is
//managing. For example, the left thread needs the value of an element on the boundry between the left and right threads.
//It needs the value of the three pixels above it, one of them, the top-right, is managed by the right thread.
//The right thread might not have gotten around to filling that value in, so the Left thread must check.
//It does that by trying to lock the mutex on that value. If the right thread already filled that value in,
//it'll get it immediately and continue. If not, the left thread will block until the right thread gets around
//to filling it in and unlocking the mutex. That means each value along the border has its own mutex.
//The thread responsible for those values must lock those mutexes first before the other thread can try.
//This limits one thread only getting about 2 rows ahead of the other thread before it finds itself blocked.
//=========================================================================================================//
void * Energy_Left( void * id )
{
	int num = (int)id;
	int energy = 0;// current calculated enery
	int min_x = 0, max_x = 0;

	while( true )
	{
		sem_wait( &(energy_sem[0]) );

		//get the update parameters
		Thread_Params energy_area = thread_info[num];

		if( energy_area.exit == true )
		{
			//thread is exiting
			break;
		}

		int * Path = energy_area.Path;

		if( Path == NULL )
		{
			//calculate full region
			min_x = energy_area.top_x;
			max_x = energy_area.bot_x;
		}
		else
		{
			//restrict calculation tree based on path location
			min_x = MAX( Path[0]-3, energy_area.top_x );
			max_x = MIN( Path[0]+2, energy_area.bot_x );	
		}

		//lock our mutexes
		for( int i = 0; i < mutex_height; i++ )
		{
			pthread_mutex_lock( &(energy_area.Mine)[i] );
		}

		//signal we are done
		sem_post( &(energy_sem[2]) );

		//wait until we are good to go
		sem_wait( &(energy_sem[3]) );

		//set the first row with the correct energy
		for( int x = min_x; x <= max_x; x++ )
		{
			(*(energy_area.Energy_Map))(x,0) = (*(energy_area.Edge))(x,0) + (*(energy_area.D_Weights))(x,0);
		}

		//now signal that one is done
		pthread_mutex_unlock( &(energy_area.Mine)[0] );

		for( int y = 1; y < (*(energy_area.Edge)).Height(); y++ )
		{
			min_x=MAX( min_x-1, energy_area.top_x );
			max_x=MIN( max_x+1, energy_area.bot_x );

			for( int x = min_x; x <= max_x; x++ ) 
			{
				if( x == energy_area.top_x )
				{
					//being the edge value, forward energy would have no benefit here, and hence is not checked
					energy = MIN( (*(energy_area.Energy_Map))(energy_area.top_x,y-1), (*(energy_area.Energy_Map))(energy_area.top_x+1,y-1) )
							  + (*(energy_area.Edge))(energy_area.top_x,y) + (*(energy_area.D_Weights))(energy_area.top_x,y);
				}
				else
				{	    
					if( x == energy_area.bot_x )//get access to the bad pixel (the one not maintained by us)
					{
						pthread_mutex_lock( &(energy_area.Not_Mine)[y-1] );
						pthread_mutex_unlock( &(energy_area.Not_Mine)[y-1] );
					}

					if( energy_area.ener == BACKWARD )
					{
						//grab the minimum of straight up, up left, or up right
						energy = min_of_three( (*(energy_area.Energy_Map))(x-1,y-1),
											   (*(energy_area.Energy_Map))(x,y-1),
											   (*(energy_area.Energy_Map))(x+1,y-1) )
								 + (*(energy_area.Edge))(x,y) + (*(energy_area.D_Weights))(x,y);
					}
					else
					{
						energy = min_of_three( (*(energy_area.Energy_Map))(x-1,y-1) + Forward_CostL(energy_area.Edge,x,y),
											   (*(energy_area.Energy_Map))(x,y-1) + Forward_CostU(energy_area.Edge,x,y),
											   (*(energy_area.Energy_Map))(x+1,y-1) + Forward_CostR(energy_area.Edge,x,y) )
								 + (*(energy_area.D_Weights))(x,y);
					}
				}

				//now we have the energy
				if((*(energy_area.Energy_Map))(x,y) == energy && Path != NULL )
				{
					if(x == min_x && Path[y]>min_x+3 )min_x++;
					if(x == max_x && Path[y]<max_x-2 )max_x--;
				}
				else
				{ //set the energy of the pixel
					 (*(energy_area.Energy_Map))(x,y) = energy;
				} 			
			}
			pthread_mutex_unlock( &(energy_area.Mine)[y] );
		}

		//signal we're done
		sem_post( &(energy_sem[4]) );
	} //end while(true)

	return NULL;
} //end Energy_Left()

//=========================================================================================================//
void * Energy_Right( void * id )
{
	int num = (int)id;
	int energy = 0;// current calculated enery
	int min_x = 0, max_x = 0;

	while( true )
	{
		sem_wait( &(energy_sem[1]) );

		//get the update parameters
		Thread_Params energy_area = thread_info[num];
		int * Path = energy_area.Path;

		if( energy_area.exit == true )
		{
			//thread is exiting
			break;
		}

		if( Path == NULL )
		{
			min_x = energy_area.top_x;
			max_x = energy_area.bot_x;
		}
		else
		{
			min_x = MAX( Path[0]-3, energy_area.top_x );
			max_x = MIN( Path[0]+2, energy_area.bot_x );	
		}

		//lock our mutexes
		for( int i = 0; i < mutex_height; i++ )
		{
			pthread_mutex_lock( &(energy_area.Mine)[i] );
		}

		//signal we are done
		sem_post( &(energy_sem[2]) );

		//wait until we are good to go
		sem_wait( &(energy_sem[3]) );

		//set the first row with the correct energy
		for( int x = min_x; x <= max_x; x++ )
		{
			(*(energy_area.Energy_Map))(x,0) = (*(energy_area.Edge))(x,0) + (*(energy_area.D_Weights))(x,0);
		}

		//now signal that one is done
		pthread_mutex_unlock( &(energy_area.Mine)[0] );

		for( int y = 1; y < (*(energy_area.Edge)).Height(); y++ )
		{
			min_x = MAX( min_x-1, energy_area.top_x );
			max_x = MIN( max_x+1, energy_area.bot_x );

			for( int x = min_x ; x <= max_x; x++ ) //+1 because we handle that seperately
			{
				if( x == energy_area.bot_x )
				{
					//being the edge value, forward energy would have no benefit here, and hence is not checked
					energy = MIN( (*(energy_area.Energy_Map))(energy_area.bot_x,y-1), (*(energy_area.Energy_Map))(energy_area.bot_x-1,y-1) )
							  + (*(energy_area.Edge))(energy_area.bot_x,y) + (*(energy_area.D_Weights))(energy_area.bot_x,y);

				}
				else
				{	    
					if(x == energy_area.top_x )//get access to the bad pixel (the one not maintained by us)
					{
						pthread_mutex_lock( &(energy_area.Not_Mine)[y-1] );
						pthread_mutex_unlock( &(energy_area.Not_Mine)[y-1] );
					}

					if( energy_area.ener == BACKWARD )
					{
						//grab the minimum of straight up, up left, or up right
						energy = min_of_three( (*(energy_area.Energy_Map))(x-1,y-1),
											   (*(energy_area.Energy_Map))(x,y-1),
											   (*(energy_area.Energy_Map))(x+1,y-1) )
								 + (*(energy_area.Edge))(x,y) + (*(energy_area.D_Weights))(x,y);
					}
					else
					{
						energy = min_of_three( (*(energy_area.Energy_Map))(x-1,y-1) + Forward_CostL(energy_area.Edge,x,y),
											   (*(energy_area.Energy_Map))(x,y-1) + Forward_CostU(energy_area.Edge,x,y),
											   (*(energy_area.Energy_Map))(x+1,y-1) + Forward_CostR(energy_area.Edge,x,y) )
								 + (*(energy_area.D_Weights))(x,y);
					}
				}

				//now we have the energy
				if( (*(energy_area.Energy_Map))(x,y) == energy && Path != NULL )
				{
					if( x == min_x && Path[y] > x+3 ) min_x++;
					if( x == max_x && Path[y] < x-2 ) max_x--;
				}
				else
				{//set the energy of the pixel
					 (*(energy_area.Energy_Map))(x,y) = energy;
				}
			} 
			pthread_mutex_unlock( &(energy_area.Mine)[y] );// could be put in the loop for faster a releasing of the mutex, but to be VERY carefull (use a boolean on the previous lock) 
		}

		//signal we're done
		sem_post( &(energy_sem[4]) );
	} //end while(true)

	return NULL;
} //end Energy_Right()

//=========================================================================================================//
//Calculates the energy map from Edge, adding in Weights where needed. The Path is used to determine how much of the
//given Map is to remain unchanged. A Path of NULL will cause the Map to be fully recalculated.
void Energy_Map( CML_int * Edge, CML_int * Weights, CML_int * Map, CAIR_energy ener, int * Path )
{
	//set the paramaters
	//left side
	thread_info[0].Edge = Edge;
	thread_info[0].D_Weights = Weights;
	thread_info[0].Energy_Map = Map;
	thread_info[0].Path = Path;
	thread_info[0].top_x = 0;
	thread_info[0].bot_x = (*Edge).Width() / 2;
	thread_info[0].ener = ener;
	thread_info[0].Mine = Left_Mutexes;
	thread_info[0].Not_Mine = Right_Mutexes;

	//the right side
	thread_info[1] = thread_info[0];
	thread_info[1].top_x = thread_info[0].bot_x + 1;
	thread_info[1].bot_x = (*Edge).Width() - 1;
	thread_info[0].Mine = Right_Mutexes;
	thread_info[0].Not_Mine = Left_Mutexes;

	//startup the left
	sem_post( &(energy_sem[0]) );

	//wait for it to lock
	sem_wait( &(energy_sem[2]) );

	//startup the right
	sem_post( &(energy_sem[1]) );

	//wait for it to lock
	sem_wait( &(energy_sem[2]) );

	//fire them up
	sem_post( &(energy_sem[3]) );
	sem_post( &(energy_sem[3]) );

	//now wait on them
	sem_wait( &(energy_sem[4]) );
	sem_wait( &(energy_sem[4]) );

} //end Energy_Map()

//=========================================================================================================//
//Energy_Path() generates the least energy Path of the Edge and Weights and returns the total energy of that path.
//This uses a dynamic programming method to easily calculate the path and energy map (see wikipedia for a good example).
//Weights should be of the same size as Edge, Path should be of proper length (the height of Edge).
int Energy_Path( CML_int * Edge, CML_int * Weights, CML_int * Energy, int * Path, CAIR_energy ener, bool first_time )
{
	(*Energy).Resize_Width( (*Edge).Width() );

	//calculate the energy map
	if( first_time == true )
	{
		Energy_Map( Edge, Weights, Energy, ener, NULL );
	}
	else
	{
		Energy_Map( Edge, Weights, Energy, ener, Path );
	}

	//find minimum path start
	int min_x = 0;
	for( int x = 0; x < (*Energy).Width(); x++ )
	{
		if( (*Energy)(x, (*Energy).Height() - 1 ) < (*Energy)(min_x, (*Energy).Height() - 1 ) )
		{
			min_x = x;
		}
	}

	//generate the path back from the energy map
	Generate_Path( Energy, min_x, Path );
	return (*Energy)( min_x, (*Energy).Height() - 1 );
}

//=========================================================================================================//
//==                                                 A D D                                               ==//
//=========================================================================================================//

//=========================================================================================================//
//averages two pixels and returns the values
CML_RGBA Average_Pixels( CML_RGBA Pixel1, CML_RGBA Pixel2 )
{
	CML_RGBA average;

	average.alpha = ( Pixel1.alpha + Pixel2.alpha ) / 2;
	average.blue = ( Pixel1.blue + Pixel2.blue ) / 2;
	average.green = ( Pixel1.green + Pixel2.green ) / 2;
	average.red = ( Pixel1.red + Pixel2.red ) / 2;

	return average;
}

//=========================================================================================================//
//This works like Remove_Quadrant, stripes across the image.
void * Add_Quadrant( void * id )
{
	int num = (int)id;
	Thread_Params add_area;

	while( true )
	{
		sem_wait( &(add_sem[0]) );

		//get updated_parameters
		add_area = thread_info[num];

		if( add_area.exit == true )
		{
			//thread is exiting
			break;
		}

		//first add the weights with the artificial weights
		//Adds the two weight matrircies, Weights and the artifical weight, into Sum.
		//This is so the new-path artificial weight doesn't poullute our input Weight matrix.
		for( int y = add_area.top_y; y < add_area.bot_y; y++ )
		{
			for( int x = 0; x < (*(add_area.D_Weights)).Width(); x++ )
			{
				(*(add_area.Sum_Weight))(x,y) = (*(add_area.Add_Weight))(x,y) + (*(add_area.D_Weights))(x,y);
			}
		}

		//signal that part is done
		sem_post( &(add_sem[3]) );

		//wait to begin the next part
		sem_wait( &(add_sem[1]) );

		//get updated_parameters
		add_area = thread_info[num];

		for( int y = add_area.top_y; y < add_area.bot_y; y++ )
		{
			int add = (add_area.Path)[y];

			//shift over everyone to the right
			(*(add_area.Source)).Shift_Row( add, y, 1 );
			(*(add_area.Add_Weight)).Shift_Row( add, y, 1 );
			(*(add_area.D_Weights)).Shift_Row( add, y, 1 );
			(*(add_area.Gray)).Shift_Row( add, y, 1 );
			(*(add_area.Energy_Map)).Shift_Row( add, y, 1 );
			
			//go back and set the added pixel
			(*(add_area.Source))(add,y) = Average_Pixels( (*(add_area.Source))(add,y), (*(add_area.Source)).Get(add-1,y));
			(*(add_area.D_Weights))(add,y) = ( (*(add_area.D_Weights))(add,y) + (*(add_area.D_Weights)).Get(add-1,y) ) / 2;
			(*(add_area.Gray))(add,y) = Grayscale_Pixel( &(*(add_area.Source))(add,y) );

			(*(add_area.Add_Weight))(add,y) = add_area.add_weight; //the new path
			if( add < (*(add_area.Add_Weight)).Width() )
			{
				(*(add_area.Add_Weight))(add+1,y) += add_area.add_weight; //the previous least-energy path
			}
		}

		//signal that part is done
		sem_post( &(add_sem[3]) );

		//wait to begin the edges
		sem_wait( &(add_sem[2]) );

		//get updated_parameters
		add_area = thread_info[num];

		for( int y = add_area.top_y; y < add_area.bot_y; y++ )
		{
			int add = (add_area.Path)[y];
			edge_safe safety = UNSAFE;
			if( (y <= 3) || (y >= (*(add_area.Edge)).Height() - 4) || (add <= 3) || (add >= (*(add_area.Edge)).Width() - 4) )
			{
				safety = SAFE;
			}

			(*(add_area.Edge)).Shift_Row( add, y, 1 );

			//these checks assume a convolution kernel no larger than 3x3
			if( (add - 1) >= 0 )
			{
				(*(add_area.Edge))(add-1,y) = Convolve_Pixel( add_area.Gray, add-1, y, safety, add_area.conv );

				if( (add - 2) >= 0 )
				{
					(*(add_area.Edge))(add-2,y) = Convolve_Pixel( add_area.Gray, add-2, y, safety, add_area.conv );

					if( (add - 3) >= 0 )
					{
						(*(add_area.Edge))(add-3,y) = Convolve_Pixel( add_area.Gray, add-3, y, safety, add_area.conv );
					}
				}
			}

			//no checks on these since they will always be there
			(*(add_area.Edge))(add,y) = Convolve_Pixel( add_area.Gray, add, y, safety, add_area.conv );
			(*(add_area.Edge))(add+1,y) = Convolve_Pixel( add_area.Gray, add+1, y, safety, add_area.conv );

			if( (add + 2) < (*(add_area.Edge)).Width() )
			{
				(*(add_area.Edge))(add+2,y) = Convolve_Pixel( add_area.Gray, add+2, y, safety, add_area.conv );

				if( (add + 3) < (*(add_area.Edge)).Width() )
				{
					(*(add_area.Edge))(add+3,y) = Convolve_Pixel( add_area.Gray, add+3, y, safety, add_area.conv );
				}
			}
		} //end edge loop

		//signal the add thread is done
		sem_post( &(add_sem[3]) );

	} //end while(true)
	return NULL;
}

//=========================================================================================================//
//Adds Path into Source, storing the result in Dest.
//AWeights is used to store the enlarging artifical weights.
void Add_Path( CML_color * Source, int * Path, CML_int * Weights, CML_int * Edge, CML_gray * Grayscale, CML_int * AWeights, CML_int * Energy, int add_weight, CAIR_convolution conv )
{
	(*Source).Resize_Width( (*Source).Width() + 1 );
	(*AWeights).Resize_Width( (*Source).Width() );
	(*Weights).Resize_Width( (*Source).Width() );
	(*Edge).Resize_Width( (*Source).Width() );
	(*Grayscale).Resize_Width( (*Source).Width() );
	(*Energy).Resize_Width( (*Source).Width() );

	int thread_height = (*Source).Height() / num_threads;

	//setup parameters
	for( int i = 0; i < num_threads; i++ )
	{
		thread_info[i].Source = Source;
		thread_info[i].Path = Path;
		thread_info[i].D_Weights = Weights;
		thread_info[i].Add_Weight = AWeights;
		thread_info[i].Edge = Edge;
		thread_info[i].conv = conv;
		thread_info[i].Gray = Grayscale;
		thread_info[i].Energy_Map = Energy;
		thread_info[i].add_weight = add_weight;
		thread_info[i].top_y = i * thread_height;
		thread_info[i].bot_y = thread_info[i].top_y + thread_height;
	}

	//have the last thread pick up the slack
	thread_info[num_threads-1].bot_y = (*Source).Height();

	//startup the threads
	for( int i = 0; i < num_threads; i++ )
	{
		sem_post( &(add_sem[1]) );
	}

	//now wait for them to come back to us
	for( int i = 0; i < num_threads; i++ )
	{
		sem_wait( &(add_sem[3]) );
	}

	//We have to wait until the grayscale image is correctly shifted to avoid bad things from happening when we edge detect.
	//We may try to get a value on the bounderies of the threads before the row is shifted.
	for( int i = 0; i < num_threads; i++ )
	{
		sem_post( &(add_sem[2]) );
	}

	//now wait on them again
	for( int i = 0; i < num_threads; i++ )
	{
		sem_wait( &(add_sem[3]) );
	}

} //end Add_Path()

//=========================================================================================================//
//Performs non-destructive Reserving for weights.
void Reserve_Weights( CML_int * Weights, int goal_x )
{
	CML_int temp( (*Weights).Width(), (*Weights).Height() );

	temp = (*Weights);

	(*Weights).Reserve( goal_x, (*Weights).Height() );

	for( int y = 0; y < temp.Height(); y++ )
	{
		for( int x = 0; x < temp.Width(); x++ )
		{
			(*Weights)(x,y) = temp(x,y);
		}
	}
}

//=========================================================================================================//
//Performs a simple copy, but preserves Reserve'ed memory.
//For copying Source back into some other image.
void Copy_Reserved( CML_color * Source, CML_color * Dest )
{
	for( int y = 0; y < (*Source).Height(); y++ )
	{
		for( int x = 0; x < (*Source).Width(); x++ )
		{
			(*Dest)(x,y) = (*Source)(x,y);
		}
	}
}

//=========================================================================================================//
//start the add threads to add the user-given weights with the artifical path weights to a sum matrix
void Start_Weight_Add( CML_int * Weights, CML_int * art_weights, CML_int * sum_weights )
{
	//setup the thread info for the sum-weights part
	int thread_height = (*Weights).Height() / num_threads;
	for( int i = 0; i < num_threads; i++ )
	{
		thread_info[i].Sum_Weight = sum_weights;
		thread_info[i].Add_Weight = art_weights;
		thread_info[i].D_Weights = Weights;
		thread_info[i].top_y = i * thread_height;
		thread_info[i].bot_y = thread_info[i].top_y + thread_height;
	}
	thread_info[num_threads-1].bot_y = (*Weights).Height();

	//fire up the threads to do the artifical weight sum
	for( int j = 0; j < num_threads; j++ )
	{
		sem_post( &(add_sem[0]) );
	}
	//wait on them
	for( int j = 0; j < num_threads; j++ )
	{
		sem_wait( &(add_sem[3]) );
	}
}

//=========================================================================================================//
//Adds some paths to Source and outputs into Dest.
//This doesn't work anything like the paper describes, mostly because they probably forgot to mention a key point:
//Added paths need a high weight so they are not chosen again, or so their neighbors are less likely to be chosen.
//For each new path, the energy of the image and its least-energy path is calculated. 
//To prevent the same path from being chosen and to prevent merging paths from being chosen, 
//additonal weight is placed to the old least-energy path and the new inserted path. Having 
//a very large add_weight will cause the algorithm to work more like a linear algorithm, evenly distributing new paths.
//Having a very small weight will cause stretching. I provide this as a paramater mainly because I don't know if someone
//will see a need for it, so I might of well leave it in.
bool CAIR_Add( CML_color * Source, CML_int * Weights, int goal_x, int add_weight, CAIR_convolution conv, CAIR_energy ener, CML_color * Dest, bool (*CAIR_callback)(float), int total_seams, int seams_done )
{
	//adjust energy thread mutexes
	Resize_Threads( (*Source).Height() );

	CML_gray Grayscale( (*Source).Width(), (*Source).Height() );

	int adds = goal_x - (*Source).Width();
	CML_int art_weight( (*Source).Width(), (*Source).Height() ); //artifical path weight
	CML_int sum_weight( (*Source).Width(), (*Source).Height() ); //the sum of Weights and the artifical weight
	CML_int Edge( (*Source).Width(), (*Source).Height() );
	CML_int Energy( (*Source).Width(), (*Source).Height() );
	int * Min_Path = new int[(*Source).Height()];

	//increase thier reserved size as we enlarge. non-destructive resizes would be too slow
	(*Dest).D_Resize( (*Source).Width(), (*Source).Height() );
	(*Dest).Reserve( goal_x, (*Source).Height() );
	Grayscale.Reserve( goal_x, (*Source).Height() );
	Edge.Reserve( goal_x, (*Source).Height() );
	Energy.Reserve( goal_x, (*Source).Height() );
	Reserve_Weights( Weights, goal_x );
	art_weight.Reserve( goal_x, (*Source).Height() );
	sum_weight.Reserve( goal_x, (*Source).Height() );

	//clear the new weight
	art_weight.Fill( 0 );

	//have to do this first to get it started
	Copy_Reserved( Source, Dest );
	Grayscale_Image( Source, &Grayscale );
	Edge_Detect( &Grayscale, &Edge, conv );

	for( int i = 0; i < adds; i++ )
	{
		//If you're going to maintain some sort of progress counter/bar, here's where you would do it!
		if( (CAIR_callback != NULL) && (CAIR_callback( (float)(i+seams_done)/total_seams ) == false) )
		{
			delete[] Min_Path;
			return false;
		}

		Start_Weight_Add( Weights, &art_weight, &sum_weight );
		if( i == 0 )
		{
			Energy_Path( &Edge, &sum_weight, &Energy, Min_Path, ener, true );
		}
		else
		{
			Energy_Path( &Edge, &sum_weight, &Energy, Min_Path, ener, false );
		}
		Add_Path( Dest, Min_Path, Weights, &Edge, &Grayscale, &art_weight, &Energy, add_weight, conv );

	}

	delete[] Min_Path;
	return true;
} //end CAIR_Add()

//=========================================================================================================//
//==                                             R E M O V E                                             ==//
//=========================================================================================================//

//=========================================================================================================//
//more multi-threaded goodness
//the areas are not quadrants, rather, more like strips, but I keep the name convention
void * Remove_Quadrant( void * id )
{
	int num = (int)id;
	Thread_Params remove_area;

	while( true )
	{
		sem_wait( &(remove_sem[0]) );

		//get updated parameters
		remove_area = thread_info[num];

		if( remove_area.exit == true )
		{
			//thread is exiting
			break;
		}

		//remove
		for( int y = remove_area.top_y; y < remove_area.bot_y; y++ )
		{
			//reduce each row by one, the removed pixel
			int remove = (remove_area.Path)[y];

			//now, bounds check the assignments
			if( (remove - 1) > 0 )
			{
				if( (*(remove_area.D_Weights))(remove,y) >= 0 ) //otherwise area marked for removal, don't blend
				{
					//average removed pixel back in
					(*(remove_area.Source))(remove-1,y) = Average_Pixels( (*(remove_area.Source))(remove,y), (*(remove_area.Source)).Get(remove-1,y) );
				}
				(*(remove_area.Gray))(remove-1,y) = Grayscale_Pixel( &(*(remove_area.Source))(remove-1,y) );
			}

			if( (remove + 1) < (*(remove_area.Source)).Width() )
			{
				if( (*(remove_area.D_Weights))(remove,y) >= 0 ) //otherwise area marked for removal, don't blend
				{
					//average removed pixel back in
					(*(remove_area.Source))(remove+1,y) = Average_Pixels( (*(remove_area.Source))(remove,y), (*(remove_area.Source)).Get(remove+1,y) );
				}
				(*(remove_area.Gray))(remove+1,y) = Grayscale_Pixel( &(*(remove_area.Source))(remove+1,y) );
			}

			//shift everyone over
			(*(remove_area.Source)).Shift_Row( remove + 1, y, -1 );
			(*(remove_area.Gray)).Shift_Row( remove + 1, y, -1 );
			(*(remove_area.D_Weights)).Shift_Row( remove + 1, y, -1 );
			(*(remove_area.Energy_Map)).Shift_Row( remove + 1, y, -1 );//to be recalculated ...
		}

		//signal that part is done
		sem_post( &(remove_sem[2]) );

		//wait to begin edge removal
		sem_wait( &(remove_sem[1]) );

		//get updated parameters
		remove_area = thread_info[num];

		//correct the edge values that have changed around the removed path
		for( int y = remove_area.top_y; y < remove_area.bot_y; y++ )
		{
			int remove = (remove_area.Path)[y];
			edge_safe safety = UNSAFE;
			if( (y <= 3) || (y >= (*(remove_area.Edge)).Height() - 4) || (remove <= 3) || (remove >= (*(remove_area.Edge)).Width() - 4) )
			{
				safety = SAFE;
			}

			//these checks assume a convolution kernel no larger than 3x3
			//check we don't blow past the left of the map
			if( (remove - 3) >= 0 )
			{
				(*(remove_area.Edge))(remove-3,y) = Convolve_Pixel( remove_area.Gray, remove-3, y, safety, remove_area.conv );

				if( (remove - 2) >= 0 )
				{
					(*(remove_area.Edge))(remove-2,y) = Convolve_Pixel( remove_area.Gray, remove-2, y, safety, remove_area.conv );

					if( (remove - 1) >= 0 )
					{
						(*(remove_area.Edge))(remove-1,y) = Convolve_Pixel( remove_area.Gray, remove-1, y, safety, remove_area.conv );
					}
				}
			}
			
			//check we don't blow past the right of the map
			if( (remove + 1) < (*(remove_area.Source)).Width() )
			{
				(*(remove_area.Edge))(remove+1,y) = Convolve_Pixel( remove_area.Gray, remove, y, safety, remove_area.conv );

				if( (remove + 2) < (*(remove_area.Source)).Width() )
				{
					(*(remove_area.Edge))(remove+2,y) = Convolve_Pixel( remove_area.Gray, remove+1, y, safety, remove_area.conv );

					if( (remove + 3) < (*(remove_area.Source)).Width() )
					{
						(*(remove_area.Edge))(remove+3,y) = Convolve_Pixel( remove_area.Gray, remove+2, y, safety, remove_area.conv );
					}
				}
			}

			//now we can safely shift
			(*(remove_area.Edge)).Shift_Row( remove + 1, y, -1 );
		}

		//signal we're now done
		sem_post( &(remove_sem[2]) );
	} //end while( true )

	return NULL;
} //end Remove_Quadrant()

//=========================================================================================================//
//Removes the requested path from the Edge, Weights, and the image itself.
//Edge and the image have the path blended back into the them.
//Weights and Edge better match the dimentions of Source! Path needs to be the same length as the height of the image!
void Remove_Path( CML_color * Source, int * Path, CML_int * Weights, CML_int * Edge, CML_gray * Grayscale, CML_int * Energy, CAIR_convolution conv )
{
	int thread_height = (*Source).Height() / num_threads;

	//setup parameters
	for( int i = 0; i < num_threads; i++ )
	{
		thread_info[i].Source = Source;
		thread_info[i].Path = Path;
		thread_info[i].D_Weights = Weights;
		thread_info[i].Edge = Edge;
		thread_info[i].conv = conv;
		thread_info[i].Gray = Grayscale;
		thread_info[i].Energy_Map = Energy;
		thread_info[i].top_y = i * thread_height;
		thread_info[i].bot_y = thread_info[i].top_y + thread_height;
	}

	//have the last thread pick up the slack
	thread_info[num_threads-1].bot_y = (*Source).Height();

	//start the four threads
	for( int i = 0; i < num_threads; i++ )
	{
		sem_post( &(remove_sem[0]) );
	}
	//now wait on them
	for( int i = 0; i < num_threads; i++ )
	{
		sem_wait( &(remove_sem[2]) );
	}

	//now we can safely resize everyone down
	(*Source).Resize_Width( (*Source).Width() - 1 );
	(*Weights).Resize_Width( (*Source).Width() );
	(*Grayscale).Resize_Width( (*Source).Width() );
	//Energy_Path() will resize Energy

	//now get the threads to handle the edge
	//we must wait for the grayscale to be complete before we can recalculate changed edge values
	for( int i = 0; i < num_threads; i++ )
	{
		sem_post( &(remove_sem[1]) );
	}

	//now wait on them, ... again
	for( int i = 0; i < num_threads; i++ )
	{
		sem_wait( &(remove_sem[2]) );
	}

	(*Edge).Resize_Width( (*Source).Width() );
} //end Remove_Path()

//=========================================================================================================//
//Removes all requested vertical paths form the image.
bool CAIR_Remove( CML_color * Source, CML_int * Weights, int goal_x, CAIR_convolution conv, CAIR_energy ener, CML_color * Dest, bool (*CAIR_callback)(float), int total_seams, int seams_done )
{
	//readjust energy thread mutexes
	Resize_Threads( (*Source).Height() );

	CML_gray Grayscale( (*Source).Width(), (*Source).Height() );

	int removes = (*Source).Width() - goal_x;
	CML_int Edge( (*Source).Width(), (*Source).Height() );
	CML_int Energy( (*Source).Width(), (*Source).Height() );
	int * Min_Path = new int[(*Source).Height()];

	//setup the images
	(*Dest) = (*Source);
	Grayscale_Image( Source, &Grayscale );
	Edge_Detect( &Grayscale, &Edge, conv );

	for( int i = 0; i < removes; i++ )
	{
		//If you're going to maintain some sort of progress counter/bar, here's where you would do it!
		if( (CAIR_callback != NULL) && (CAIR_callback( (float)(i+seams_done)/total_seams ) == false) )
		{
			delete[] Min_Path;
			return false;
		}

		if( i == 0 )
		{
			Energy_Path( &Edge, Weights, &Energy, Min_Path, ener, true );
		}
		else
		{
			Energy_Path( &Edge, Weights, &Energy, Min_Path, ener, false );
		}
		Remove_Path( Dest, Min_Path, Weights, &Edge, &Grayscale, &Energy, conv );
	}

	delete[] Min_Path;
	return true;
} //end CAIR_Remove()

//=========================================================================================================//
//Startup all threads, create all needed semaphores.
//NOTE: This does NOT create the mutexes for the energy threads! Use Resize_Threads() after this, to do that.
void Startup_Threads()
{
	//create semaphores
	sem_init( &(remove_sem[0]), 0, 0 ); //start
	sem_init( &(remove_sem[1]), 0, 0 ); //edge_start
	sem_init( &(remove_sem[2]), 0, 0 ); //finish
	sem_init( &(add_sem[0]), 0, 0 ); //add_start
	sem_init( &(add_sem[1]), 0, 0 ); //start
	sem_init( &(add_sem[2]), 0, 0 ); //edge_start
	sem_init( &(add_sem[3]), 0, 0 ); //finish
	sem_init( &(edge_sem[0]), 0, 0 ); //start
	sem_init( &(edge_sem[1]), 0, 0 ); //finish
	sem_init( &(gray_sem[0]), 0, 0 ); //start
	sem_init( &(gray_sem[1]), 0, 0 ); //finish
	sem_init( &(energy_sem[0]), 0, 0 ); //start_left
	sem_init( &(energy_sem[1]), 0, 0 ); //start_right
	sem_init( &(energy_sem[2]), 0, 0 ); //locks_done
	sem_init( &(energy_sem[3]), 0, 0 ); //good_to_go
	sem_init( &(energy_sem[4]), 0, 0 ); //finish

	//create the thread handles
	remove_threads = new pthread_t[num_threads];
	edge_threads   = new pthread_t[num_threads];
	gray_threads   = new pthread_t[num_threads];
	add_threads    = new pthread_t[num_threads];

	thread_info = new Thread_Params[num_threads];

	//startup the threads
	for( int i = 0; i < num_threads; i++ )
	{
		thread_info[i].exit = false;

		pthread_create( &(remove_threads[i]), NULL, Remove_Quadrant, (void *)i );
		pthread_create( &(edge_threads[i]), NULL, Edge_Quadrant, (void *)i );
		pthread_create( &(gray_threads[i]), NULL, Gray_Quadrant, (void *)i );
		pthread_create( &(add_threads[i]), NULL, Add_Quadrant, (void *)i );
	}

	//startup energy
	pthread_create( &(energy_threads[0]), NULL, Energy_Left, (void *)0 );
	pthread_create( &(energy_threads[1]), NULL, Energy_Right, (void *)1 );
}

//=========================================================================================================//
//Creates or resizes the arrays of mutexes for the two energy threads, depending on the height of the image.
void Resize_Threads( int height )
{
	if( Left_Mutexes != NULL )
	{
		//clear out and delete the left
		for( int i = 0; i < mutex_height; i++ )
		{
			pthread_mutex_destroy( &(Left_Mutexes[i]) );
		}

		delete[] Left_Mutexes;
	}
	if( Right_Mutexes != NULL )
	{
		//clear out and delete the right
		for( int i = 0; i < mutex_height; i++ )
		{
			pthread_mutex_destroy( &(Right_Mutexes[i]) );
		}

		delete[] Right_Mutexes;
	}

	//creat the new objects
	if( height > 0 )
	{
		Left_Mutexes = new pthread_mutex_t[height];
		Right_Mutexes = new pthread_mutex_t[height];
	}
	else
	{
		Left_Mutexes = NULL;
		Right_Mutexes = NULL;
	}

	//init the mutexes
	for( int i = 0; i < height; i++ )
	{
		pthread_mutex_init( &(Left_Mutexes[i]), NULL );
		pthread_mutex_init( &(Right_Mutexes[i]), NULL );
	}

	mutex_height = height;
}

//=========================================================================================================//
//Stops all threads. Deletes all semaphores and mutexes.
void Shutdown_Threads()
{
	//notify the threads
	for( int i = 0; i < num_threads; i++ )
	{
		thread_info[i].exit = true;
	}

	//start them up
	for( int i = 0; i < num_threads; i++ )
	{
		sem_post( &(remove_sem[0]) );
		sem_post( &(add_sem[0]) );
		sem_post( &(edge_sem[0]) );
		sem_post( &(gray_sem[0]) );
	}
	sem_post( &(energy_sem[0]) );
	sem_post( &(energy_sem[1]) );

	//wait for the joins
	for( int i = 0; i < num_threads; i++ )
	{
		pthread_join( remove_threads[i], NULL );
		pthread_join( edge_threads[i], NULL );
		pthread_join( gray_threads[i], NULL );
		pthread_join( add_threads[i], NULL );
	}
	pthread_join( energy_threads[0], NULL );
	pthread_join( energy_threads[1], NULL );

	//remove the thread handles
	delete[] remove_threads;
	delete[] edge_threads;
	delete[] gray_threads;
	delete[] add_threads;

	delete[] thread_info;

	//delete the semaphores
	sem_destroy( &(remove_sem[0]) ); //start
	sem_destroy( &(remove_sem[1]) ); //edge_start
	sem_destroy( &(remove_sem[2]) ); //finish
	sem_destroy( &(add_sem[0]) ); //add_start
	sem_destroy( &(add_sem[1]) ); //start
	sem_destroy( &(add_sem[2]) ); //edge_start
	sem_destroy( &(add_sem[3]) ); //finish
	sem_destroy( &(edge_sem[0]) ); //start
	sem_destroy( &(edge_sem[1]) ); //finish
	sem_destroy( &(gray_sem[0]) ); //start
	sem_destroy( &(gray_sem[1]) ); //finish
	sem_destroy( &(energy_sem[0]) ); //start_left
	sem_destroy( &(energy_sem[1]) ); //start_right
	sem_destroy( &(energy_sem[2]) ); //locks_done
	sem_destroy( &(energy_sem[3]) ); //good_to_go
	sem_destroy( &(energy_sem[4]) ); //finish

	//let the mutexes begone!
	Resize_Threads( 0 );
}

//=========================================================================================================//
//Set the number of threads that CAIR should use. Minimum of 2 required.
//WARNING: Never call this function while CAIR() is processing an image, otherwise bad things will happen!
void CAIR_Threads( int thread_count )
{
	//minimum of two because I need two thread_info[] structs for the energy threads
	if( thread_count < 2 )
	{
		num_threads = 2;
	}
	else
	{
		num_threads = thread_count;
	}
}

//=========================================================================================================//
//==                                          F R O N T E N D                                            ==//
//=========================================================================================================//
//The Great CAIR Frontend. This baby will retarget Source using S_Weights into the dimensions supplied by goal_x and goal_y into D_Weights and Dest.
//Weights allows for an area to be biased for removal/protection. A large positive value will protect a portion of the image,
//and a large negative value will remove it. Do not exceed the limits of int's, as this will cause an overflow. I would suggest
//a safe range of -2,000,000 to 2,000,000 (this is a maximum guideline, much smaller weights will work just as well for most images).
//Weights must be the same size as Source. D_Weights will contain the weights of Dest after the resize. Dest is the output,
//and as such has no constraints (its contents will be destroyed, just so you know). 
//To prevent the same path from being chosen during an add, and to prevent merging paths from being chosen during an add, 
//additional weight is placed to the old least-energy path and the new inserted path. Having  a very large add_weight 
//will cause the algorithm to work more like a linear algorithm. Having a very small add_weight will cause stretching. 
//A weight of greater than 25 should prevent stretching, but may not evenly distribute paths through an area. 
//Note: Weights does affect path adding, so a large negative weight will attract the most paths. Also, if add_weight is too large,
//it may eventually force new paths into areas marked for protection. I am unsure of an exact ratio on such things at this time.
//The internal order is this: remove horizontal, remove vertical, add horizontal, add vertical.
//CAIR can use multiple convolution methods to determine the image energy. 
//Prewitt and Sobel are close to each other in results and represent the "traditional" edge detection.
//V_SQUARE and V1 can produce some of the better quality results, but may remove from large objects to do so. Do note that V_SQUARE
//produces much larger edge values, any may require larger weight values (by about an order of magnitude) for effective operation.
//Laplacian is a second-derivative operator, and can limit some artifacts while generating others.
//CAIR now can use the new improved energy algorithm called "forward energy." Removing seams can sometimes add energy back to the image
//by placing nearby edges directly next to each other. Forward energy can get around this by determining the future cost of a seam.
//Forward energy removes most serious artifacts from a retarget, but is slightly more costly in terms of performance.
bool CAIR( CML_color * Source, CML_int * S_Weights, int goal_x, int goal_y, int add_weight, CAIR_convolution conv, CAIR_energy ener, CML_int * D_Weights, CML_color * Dest, bool (*CAIR_callback)(float) )
{
	//if no change, then just copy to the source to the destination
	if( (goal_x == (*Source).Width()) && (goal_y == (*Source).Height() ) )
	{
		(*Dest) = (*Source);
		(*D_Weights) = (*S_Weights);
		return true;
	}

	//calculate the total number of operations needed
	int total_seams = abs((*Source).Width()-goal_x) + abs((*Source).Height()-goal_y);
	int seams_done = 0;

	//create threads for the run
	Startup_Threads();

	CML_color Temp( 1, 1 );
	Temp = (*Source);
	(*D_Weights) = (*S_Weights);

	if( goal_x < (*Source).Width() )
	{
		if( CAIR_Remove( Source, D_Weights, goal_x, conv, ener, Dest, CAIR_callback, total_seams, seams_done ) == false )
		{
			Shutdown_Threads();
			return false;
		}
		Temp = (*Dest);
		seams_done += abs((*Source).Width()-goal_x);
	}

	if( goal_y < (*Source).Height() )
	{
		//remove horiztonal paths
		//works like above, except hand it a rotated image AND weights
		CML_color TSource( 1, 1 );
		CML_color TDest( 1, 1 );
		CML_int TWeights( 1, 1 );
		TSource.Transpose( &Temp );
		TWeights.Transpose( D_Weights );

		if( CAIR_Remove( &TSource, &TWeights, goal_y, conv, ener, &TDest, CAIR_callback, total_seams, seams_done ) == false )
		{
			Shutdown_Threads();
			return false;
		}
		
		//store back the transposed info
		(*Dest).Transpose( &TDest );
		(*D_Weights).Transpose( &TWeights );
		Temp = (*Dest);
		seams_done += abs((*Source).Height()-goal_y);
	}

	if( goal_x > (*Source).Width() )
	{
		if( CAIR_Add( &Temp, D_Weights, goal_x, add_weight, conv, ener, Dest, CAIR_callback, total_seams, seams_done ) == false )
		{
			Shutdown_Threads();
			return false;
		}
		Temp = (*Dest); //incase we resize in the y direction
		seams_done += abs((*Source).Width()-goal_x);
	}
	if( goal_y > (*Source).Height() )
	{
		//add horiztonal paths
		//works like above, except hand it a rotated image
		CML_color TSource( 1, 1 );
		CML_color TDest( 1, 1 );
		CML_int TWeights( 1, 1 );
		TSource.Transpose( &Temp );
		TWeights.Transpose( D_Weights );

		if( CAIR_Add( &TSource, &TWeights, goal_y, add_weight, conv, ener, &TDest, CAIR_callback, total_seams, seams_done ) == false )
		{
			Shutdown_Threads();
			return false;
		}
		
		//store back the transposed info
		(*Dest).Transpose( &TDest );
		(*D_Weights).Transpose( &TWeights );
		seams_done += abs((*Source).Height()-goal_y);
	}

	//shutdown threads, remove semaphores and mutexes
	Shutdown_Threads();
	return true;
} //end CAIR()

//=========================================================================================================//
//==                                                E X T R A S                                          ==//
//=========================================================================================================//
//Simple function that generates the grayscale image of Source and places the result in Dest.
void CAIR_Grayscale( CML_color * Source, CML_color * Dest )
{
	Startup_Threads();

	CML_gray gray( (*Source).Width(), (*Source).Height() );
	Grayscale_Image( Source, &gray );

	(*Dest).D_Resize( (*Source).Width(), (*Source).Height() );

	for( int x = 0; x < (*Source).Width(); x++ )
	{
		for( int y = 0; y < (*Source).Height(); y++ )
		{
			(*Dest)(x,y).red = gray(x,y);
			(*Dest)(x,y).green = gray(x,y);
			(*Dest)(x,y).blue = gray(x,y);
			(*Dest)(x,y).alpha = (*Source)(x,y).alpha;
		}
	}

	Shutdown_Threads();
}

//=========================================================================================================//
//Simple function that generates the edge-detection image of Source and stores it in Dest.
void CAIR_Edge( CML_color * Source, CAIR_convolution conv, CML_color * Dest )
{
	Startup_Threads();

	CML_gray gray( (*Source).Width(), (*Source).Height() );
	Grayscale_Image( Source, &gray );

	CML_int edge( (*Source).Width(), (*Source).Height() );
	Edge_Detect( &gray, &edge, conv );

	(*Dest).D_Resize( (*Source).Width(), (*Source).Height() );

	for( int x = 0; x < (*Source).Width(); x++ )
	{
		for( int y = 0; y < (*Source).Height(); y++ )
		{
			int value = edge(x,y);

			if( value > 255 )
			{
				value = 255;
			}

			(*Dest)(x,y).red = (CML_byte)value;
			(*Dest)(x,y).green = (CML_byte)value;
			(*Dest)(x,y).blue = (CML_byte)value;
			(*Dest)(x,y).alpha = (*Source)(x,y).alpha;
		}
	}

	Shutdown_Threads();
}

//=========================================================================================================//
//Simple function that generates the vertical energy map of Source placing it into Dest.
//All values are scaled down to their relative gray value. Weights are assumed all zero.
void CAIR_V_Energy( CML_color * Source, CAIR_convolution conv, CAIR_energy ener, CML_color * Dest )
{
	Startup_Threads();
	Resize_Threads( (*Source).Height() );

	CML_gray gray( (*Source).Width(), (*Source).Height() );
	Grayscale_Image( Source, &gray );

	CML_int edge( (*Source).Width(), (*Source).Height() );
	Edge_Detect( &gray, &edge, conv );

	CML_int energy( edge.Width(), edge.Height() );
	CML_int weights( edge.Width(), edge.Height() );
	weights.Fill(0);

	//calculate the energy map
	Energy_Map( &edge, &weights, &energy, ener, NULL );

	int max_energy = 0; //find the maximum energy value
	for( int x = 0; x < energy.Width(); x++ )
	{
		for( int y = 0; y < energy.Height(); y++ )
		{
			if( energy(x,y) > max_energy )
			{
				max_energy = energy(x,y);
			}
		}
	}
	
	(*Dest).D_Resize( (*Source).Width(), (*Source).Height() );

	for( int x = 0; x < energy.Width(); x++ )
	{
		for( int y = 0; y < energy.Height(); y++ )
		{
			//scale the gray value down so we can get a realtive gray value for the energy level
			int value = (int)(((double)energy(x,y) / max_energy) * 255);
			if( value < 0 )
			{
				value = 0;
			}

			(*Dest)(x,y).red = (CML_byte)value;
			(*Dest)(x,y).green = (CML_byte)value;
			(*Dest)(x,y).blue = (CML_byte)value;
			(*Dest)(x,y).alpha = (*Source)(x,y).alpha;
		}
	}

	Shutdown_Threads();
} //end CAIR_V_Energy()

//=========================================================================================================//
//Simple function that generates the horizontal energy map of Source placing it into Dest.
//All values are scaled down to their relative gray value. Weights are assumed all zero.
void CAIR_H_Energy( CML_color * Source, CAIR_convolution conv, CAIR_energy ener, CML_color * Dest )
{
	CML_color Tsource( 1, 1 );
	CML_color Tdest( 1, 1 );

	Tsource.Transpose( Source );
	CAIR_V_Energy( &Tsource, conv, ener, &Tdest );

	(*Dest).Transpose( &Tdest );
}

//=========================================================================================================//
//Experimental automatic object removal.
//Any area with a negative weight will be removed. This function has three modes, determined by the choice paramater.
//AUTO will have the function count the veritcal and horizontal rows/columns and remove in the direction that has the least.
//VERTICAL will force the function to remove all negative weights in the veritcal direction; likewise for HORIZONTAL.
//Because some conditions may cause the function not to remove all negative weights in one pass, max_attempts lets the function
//go through the remoal process as many times as you're willing.
bool CAIR_Removal( CML_color * Source, CML_int * S_Weights, CAIR_direction choice, int max_attempts, int add_weight, CAIR_convolution conv, CAIR_energy ener, CML_int * D_Weights, CML_color * Dest, bool (*CAIR_callback)(float) )
{
	int negative_x = 0;
	int negative_y = 0;
	CML_color Temp( 1, 1 );
	Temp = (*Source);
	(*D_Weights) = (*S_Weights);

	for( int i = 0; i < max_attempts; i++ )
	{
		negative_x = 0;
		negative_y = 0;

		//count how many negative columns exist
		for( int x = 0; x < (*D_Weights).Width(); x++ )
		{
			for( int y = 0; y < (*D_Weights).Height(); y++ )
			{
				if( (*D_Weights)(x,y) < 0 )
				{
					negative_x++;
					break; //only breaks the inner loop
				}
			}
		}

		//count how many negative rows exist
		for( int y = 0; y < (*D_Weights).Height(); y++ )
		{
			for( int x = 0; x < (*D_Weights).Width(); x++ )
			{
				if( (*D_Weights)(x,y) < 0 )
				{
					negative_y++;
					break;
				}
			}
		}

		switch( choice )
		{
		case AUTO :
			//remove in the direction that has the least to remove
			if( negative_y < negative_x )
			{
				if( CAIR( &Temp, D_Weights, Temp.Width(), Temp.Height() - negative_y, add_weight, conv, ener, D_Weights, Dest, CAIR_callback ) == false )
				{
					return false;
				}
				Temp = (*Dest);
			}
			else
			{
				if( CAIR( &Temp, D_Weights, Temp.Width() - negative_x, Temp.Height(), add_weight, conv, ener, D_Weights, Dest, CAIR_callback ) == false )
				{
					return false;
				}
				Temp = (*Dest);
			}
			break;

		case HORIZONTAL :
			if( CAIR( &Temp, D_Weights, Temp.Width(), Temp.Height() - negative_y, add_weight, conv, ener, D_Weights, Dest, CAIR_callback ) == false )
			{
				return false;
			}
			Temp = (*Dest);
			break;

		case VERTICAL :
			if( CAIR( &Temp, D_Weights, Temp.Width() - negative_x, Temp.Height(), add_weight, conv, ener, D_Weights, Dest, CAIR_callback ) == false )
			{
				return false;
			}
			Temp = (*Dest);
			break;
		}
	}

	//now expand back out to the origional
	return CAIR( &Temp, D_Weights, (*Source).Width(), (*Source).Height(), add_weight, conv, ener, D_Weights, Dest, CAIR_callback );
} //end CAIR_Removal()

//=========================================================================================================//
//Precompute removals in the x direction. Map will hold the largest width the corisponding pixel is still visible.
//This will calculate all removals down to 3 pixels in width.
//Right now this only performs removals and only the x-direction. For the future enlarging is planned. Precomputing for both directions
//doesn't work all that well and generates significant artifacts. This function is intended for "content-aware multi-size images" as mentioned
//in the doctors' presentation. The next logical step would be to encode Map into an existing image format. Then, using a function like
//CAIR_Map_Resize() the image can be resized on a client machine with very little overhead.
void CAIR_Image_Map( CML_color * Source, CML_int * Weights, CAIR_convolution conv, CAIR_energy ener, CML_int * Map )
{
	Startup_Threads();
	Resize_Threads( (*Source).Height() );

	(*Map).D_Resize( (*Source).Width(), (*Source).Height() );
	(*Map).Fill( 0 );

	CML_color Temp( 1, 1 );
	Temp = (*Source);
	CML_int Temp_Weights( 1, 1 );
	Temp_Weights = (*Weights); //don't change Weights since there is no change to the image

	for( int i = Temp.Width(); i > 3; i-- ) //3 is the minimum safe amount with 3x3 convolution kernels without causing problems
	{
		//grayscale
		CML_gray Grayscale( Temp.Width(), Temp.Height() );
		Grayscale_Image( &Temp, &Grayscale );

		//edge detect
		CML_int Edge( Temp.Width(), Temp.Height() );
		Edge_Detect( &Grayscale, &Edge, conv );

		//find the energy values
		int * Path = new int[(*Source).Height()];
		CML_int Energy( Temp.Width(), Temp.Height() );
		Energy_Path( &Edge, &Temp_Weights, &Energy, Path, ener, true );

		Remove_Path( &Temp, Path, &Temp_Weights, &Edge, &Grayscale, &Energy, conv );

		//now set the corisponding map value with the resolution
		for( int y = 0; y < Temp.Height(); y++ )
		{
			int index = 0;
			int offset = Path[y];

			while( (*Map)(index,y) != 0 ) index++; //find the pixel that is in location zero (first unused)
			while( offset > 0 )
			{
				if( (*Map)(index,y) == 0 ) //find the correct x index
				{
					offset--;
				}
				index++;
			}
			while( (*Map)(index,y) != 0 ) index++; //if the current and subsequent pixels have been removed

			(*Map)(index,y) = i; //this is now the smallest resolution this pixel will be visible
		}

		delete[] Path;
	}

	Shutdown_Threads();
} //end CAIR_Image_Map()

//=========================================================================================================//
//An "example" function on how to decode the Map to quickly resize an image. This is only for the width, since multi-directional
//resizing produces significant artifacts. Do note this will produce different results than standard CAIR(), because this resize doesn't
//average pixels back into the image as does CAIR(). This function could be multi-threaded much like Remove_Path() for even faster performance.
void CAIR_Map_Resize( CML_color * Source, CML_int * Map, int goal_x, CML_color * Dest )
{
	(*Dest).D_Resize( goal_x, (*Source).Height() );

	for( int y = 0; y < (*Source).Height(); y++ )
	{
		int input_x = 0; //map the Source's pixels to the Dests smaller size
		for( int x = 0; x < goal_x; x++ )
		{
			while( (*Map)(input_x,y) > goal_x ) input_x++; //skip past pixels not in this resolution

			(*Dest)(x,y) = (*Source)(input_x,y);
			input_x++;
		}
	}
}

//=========================================================================================================//
//==                                             C A I R  H D                                            ==//
//=========================================================================================================//
//This works as CAIR, except here maximum quality is attempted. When removing in both directions some amount, CAIR_HD()
//will determine which direction has the least amount of energy and then removes in that direction. This is only done
//for removal, since enlarging will not benifit, although this function will perform addition just like CAIR().
//Inputs are the same as CAIR().
bool CAIR_HD( CML_color * Source, CML_int * S_Weights, int goal_x, int goal_y, int add_weight, CAIR_convolution conv, CAIR_energy ener, CML_int * D_Weights, CML_color * Dest, bool (*CAIR_callback)(float) )
{
	Startup_Threads();

	//if no change, then just copy to the source to the destination
	if( (goal_x == (*Source).Width()) && (goal_y == (*Source).Height() ) )
	{
		(*Dest) = (*Source);
		(*D_Weights) = (*S_Weights);
		return true;
	}

	int total_seams = abs((*Source).Width()-goal_x) + abs((*Source).Height()-goal_y);
	int seams_done = 0;

	CML_color Temp( 1, 1 );
	CML_color TTemp( 1, 1 );

	//to start the loop
	(*Dest) = (*Source);
	(*D_Weights) = (*S_Weights);

	//do this loop when we can remove in either direction
	while( ((*Dest).Width() > goal_x) && ((*Dest).Height() > goal_y) )
	{
		Temp = (*Dest);
		TTemp.Transpose( Dest );

		//grayscale the normal and transposed
		CML_gray Grayscale( Temp.Width(), Temp.Height() );
		CML_gray TGrayscale( TTemp.Width(), TTemp.Height() );
		Grayscale_Image( &Temp, &Grayscale );
		Grayscale_Image( &TTemp, &TGrayscale );

		//edge detect
		CML_int Edge( Temp.Width(), Temp.Height() );
		CML_int TEdge( TTemp.Width(), TTemp.Height() );
		Edge_Detect( &Grayscale, &Edge, conv );
		Edge_Detect( &TGrayscale, &TEdge, conv );

		//find the energy values
		CML_int TWeights( 1, 1 );
		TWeights.Transpose( D_Weights );
		int * Path = new int[Temp.Height()];
		int * TPath = new int[TTemp.Height()];
		CML_int Energy( Temp.Width(), Temp.Height() );
		CML_int TEnergy( TTemp.Width(), TTemp.Height() );
		Resize_Threads( Temp.Height() );
		int energy_x = Energy_Path( &Edge, D_Weights, &Energy, Path, ener, true );
		Resize_Threads( TTemp.Height() );
		int energy_y = Energy_Path( &TEdge, &TWeights, &TEnergy, TPath, ener, true );

		if( energy_y < energy_x )
		{
			Remove_Path( &TTemp, TPath, &TWeights, &TEdge, &TGrayscale, &TEnergy, conv );
			(*Dest).Transpose( &TTemp );
			(*D_Weights).Transpose( &TWeights );
		}
		else
		{
			Remove_Path( &Temp, Path, D_Weights, &Edge, &Grayscale, &Energy, conv );
			(*Dest) = Temp;
		}

		delete[] Path;
		delete[] TPath;

		if( (CAIR_callback != NULL) && (CAIR_callback( (float)(seams_done)/total_seams ) == false) )
		{
			Shutdown_Threads();
			return false;
		}
		seams_done++;
	}

	//one dimension is the now on the goal, so finish off the other direction
	Temp = (*Dest);
	Shutdown_Threads();
	return CAIR( &Temp, D_Weights, goal_x, goal_y, add_weight, conv, ener, D_Weights, Dest, CAIR_callback );
} //end CAIR_HD()
