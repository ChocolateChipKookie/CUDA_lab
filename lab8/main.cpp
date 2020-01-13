//------------------------------------------------------------------------------
//
// Name:       vadd_cpp.cpp
// 
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
//                   c = a + b
//
// HISTORY:    Written by Tim Mattson, June 2011
//             Ported to C++ Wrapper API by Benedict Gaster, September 2011
//             Updated to C++ Wrapper API v1.2 by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//             
//------------------------------------------------------------------------------

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

#include "util.hpp" // utility library

#include "err_code.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>
#include <random>

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (2<<26)    // length of vectors a, b, and c

int main(void)
{
   std::vector<float> h_a(LENGTH);                // a vector 
   std::vector<float> h_b(LENGTH);                // b vector 
   std::vector<float> h_e(LENGTH);                // e vector 	
   std::vector<float> h_g(LENGTH);                // g vector 	
   std::vector<float> h_c (LENGTH, 0xdeadbeef);    // c = a + b, from compute device
   std::vector<float> h_d (LENGTH, 0xdeadbeef);    // d = c + e, from compute device
   std::vector<float> h_f (LENGTH, 0xdeadbeef);    // f = d + g, from compute device

   cl::Buffer d_a, d_b, d_c, d_d, d_e, d_f, d_g; // device memory used for the input and output vectors

   // Fill vectors a and b with random float values
   int count = LENGTH;

   //Create random generator
   std::random_device rd;
   std::uniform_real_distribution<float> real_distrubution(0, 1)

   //Fill vectors
   for(int i = 0; i < count; i++)
   {
      h_a[i]  = real_distrubution(rd);
      h_b[i]  = real_distrubution(rd);
      h_e[i]  = real_distrubution(rd);
      h_g[i]  = real_distrubution(rd);
   }

   try 
   {
      // Create a context
      cl::Context context(DEVICE);

      // Load in kernel source, creating a program object for the context
      cl::Program program(context, util::loadProgram("vadd.cl"), true);

      // Get the command queue
      cl::CommandQueue queue(context);

      // Create the kernel functor
      auto vadd = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "vadd");

      //Transfer buffer information
      d_a   = cl::Buffer(context, begin(h_a), end(h_a), true);
      d_b   = cl::Buffer(context, begin(h_b), end(h_b), true);
      d_e   = cl::Buffer(context, begin(h_e), end(h_e), true);
      d_g   = cl::Buffer(context, begin(h_g), end(h_g), true);

      //Create result buffers
      d_c  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
      d_d  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * LENGTH);
      d_f  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);
        
      util::Timer timer;

      vadd(
         cl::EnqueueArgs(queue, cl::NDRange(count)), 
         d_a,
         d_b,
         d_c,
         count);

      vadd(
         cl::EnqueueArgs(queue, cl::NDRange(count)), 
         d_c,
         d_e,
         d_d,
         count);

      vadd(
         cl::EnqueueArgs(queue, cl::NDRange(count)), 
         d_d,
         d_g,
         d_f,
         count);

      queue.finish();

      double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
      printf("\nThe kernels ran in %lf seconds\n", rtime);

      //Returning the results 
      cl::copy(queue, d_c, begin(h_c), end(h_c));
      cl::copy(queue, d_d, begin(h_e), end(h_e));
      cl::copy(queue, d_f, begin(h_f), end(h_f));

      // Test the results A + B = C
      int correct_c = 0;
      for(int i = 0; i < count; i++) {
         float tmp = abs((h_a[i] + h_b[i]) - h_c[i]);
         if(tmp < TOL)
            correct_c++;
      }

      // Test the results C + E = D
      int correct_d = 0;
      for(int i = 0; i < count; i++) {
         float tmp = abs((h_c[i] + h_e[i]) - h_d[i]);
         if(tmp < TOL)
            correct_d++;
      }

      // Test the results D + G = F
      int correct_f = 0;
      for(int i = 0; i < count; i++) {
         float tmp = abs((h_d[i] + h_g[i]) - h_f[i]);
         if(tmp < TOL)
            correct_f++;
      }

      // Results
      std::cout <<
         "Results for: \n" << 
         "A + B = C -> Correct" << correct_c << "\n"
         "C + E = D -> Correct" << correct_d << "\n"
         "D + G = F -> Correct" << correct_f << "\n"
   }
   catch (cl::Error err) {
      std::cout << "Exception\n";
      std::cerr 
         << "ERROR: "
         << err.what()
         << "("
         << err_code(err.err())
         << ")"
         << std::endl;
   }
}
