#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <iomanip>

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#endif

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_ENABLE_EXCEPTIONS

#include "CL/opencl.hpp"

#ifndef ANALYZE
#define ANALYZE 1
#endif

#define dbgs                                                                   \
  if (!ANALYZE) {                                                              \
  } else                                                                       \
    std::cout

constexpr size_t ARR_SIZE = 1'000'000;

#define STRINGIFY(...) #__VA_ARGS__

// We can load kernels from common text
// ---------------------------------- OpenCL ---------------------------------
const char *vakernel = STRINGIFY(
__kernel void vectorAdd(__global int * A, __global int * B, __global int * C)
{
  int i = get_global_id(0);
  C[i] = A[i] + B[i];
}
);
// ---------------------------------- OpenCL ---------------------------------

class OpenCLApp
{
public: //

  OpenCLApp() :
    m_P(selectPlatform()),
    m_C(getGpuContext(m_P())),
    m_Q(m_C, cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder)
  {
    cl::string name = m_P.getInfo<CL_PLATFORM_NAME>();
    cl::string profile = m_P.getInfo<CL_PLATFORM_PROFILE>();
    dbgs << "Selected: " << name << ": " << profile << std::endl;
  }

  static cl::Platform selectPlatform()
  {
    cl::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (auto p : platforms) {

      cl::vector<cl::Device> devices;
      p.getDevices(CL_DEVICE_TYPE_GPU, &devices);

      if (!devices.empty())
        return p;
    }
    throw std::runtime_error("No platform selected");
  }

  static cl::Context getGpuContext(cl_platform_id _PlatformId)
  {
    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(_PlatformId),
        0
    };
    return cl::Context(CL_DEVICE_TYPE_GPU, properties);
  }

  cl_ulong vadd(const cl_int * _A, const cl_int * _B, cl_int * _C, size_t _Sz)
  {
    size_t BufSz = _Sz * sizeof(cl_int);

    cl::Buffer A(m_C, CL_MEM_READ_ONLY, BufSz);
    cl::Buffer B(m_C, CL_MEM_READ_ONLY, BufSz);
    cl::Buffer C(m_C, CL_MEM_WRITE_ONLY, BufSz);

    cl::copy(m_Q, _A, _A + _Sz, A);
    cl::copy(m_Q, _B, _B + _Sz, B);

    cl::Program program(m_C, vakernel, true);

    vadd_t add_vecs(program, "vectorAdd");

    cl::NDRange GlobalRange(_Sz);
    cl::EnqueueArgs Args(m_Q, GlobalRange);

    cl::Event evt = add_vecs(Args, A, B, C);
    evt.wait();

    const auto start = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    const auto end = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    cl::copy(m_Q, C, _C, _C + _Sz);

    return end - start;
  }

private:

  cl::Platform m_P;
  cl::Context m_C;
  cl::CommandQueue m_Q;

  using vadd_t = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>;
};

int main()
try
{

  OpenCLApp app;
  cl::vector<cl_int> src1(ARR_SIZE), src2(ARR_SIZE), dst(ARR_SIZE);

  std::iota(src1.begin(), src1.end(), 0);
  std::iota(src2.rbegin(), src2.rend(), 0);

  cl_ulong gpuTime = app.vadd(src1.data(), src2.data(), dst.data(), dst.size());

  cl::vector<cl_int> expected(ARR_SIZE);

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < ARR_SIZE; ++i)
  {
    expected[i] = src1[i] + src2[i];
  }
  const auto end = std::chrono::steady_clock::now();

  for (int i = 0; i < ARR_SIZE; ++i) {

    if (dst[i] != expected[i])
    {
      std::cerr << "Error at index " << i << ": " << dst[i] << " != " << expected[i] << std::endl;
      return -1;
    }
  }
  std::cout << "All checks passed" << std::endl;

  std::cout << "GPU time: " << std::setw(9) << gpuTime << " ns" << std::endl;
  std::cout << "CPU time: " << std::setw(9) << (end - start).count() << " ns" << std::endl;

  return 0;
}
catch (cl::Error & err)
{
  std::cerr << "OPEN_CL ERROR " << err.err() << ":" << err.what() << std::endl;
  return -1;
}
catch (std::runtime_error & err)
{
  std::cerr << "RUNTIME ERROR " << err.what() << std::endl;
  return -1;
}
catch (...)
{
  std::cerr << "UNKNOWN ERROR\n";
  return -1;
}