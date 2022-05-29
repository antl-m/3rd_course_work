#include <algorithm>
#include <cassert>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

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

#define STRINGIFY(...) #__VA_ARGS__

const char* hist_kernel = STRINGIFY(

  __kernel void histogram(__global uchar * data, int num_data, __global int* histogram, __local int* local_hist, int num_bins) {
  int i;
  int lid = get_local_id(0);
  int gid = get_global_id(0);
  int lsize = get_local_size(0);
  int gsize = get_global_size(0);

  for (i = lid; i < num_bins; i += lsize)
    local_hist[i] = 0;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (i = gid; i < num_data; i += gsize)
    atomic_add(&local_hist[data[i]], 1);

  barrier(CLK_LOCAL_MEM_FENCE);

  for (i = lid; i < num_bins; i += lsize)
    atomic_add(&histogram[i], local_hist[i]);
}
);

constexpr int DATABLOCK = 256;

struct Config {
  int DataSize = 1'000'000;   // number of 256-blocks
  int GlobalSize = 100; // number of 256-datagroups
  int HistSize = 1024;  // number of histogram bins
  int LSZ = 32;
  cl::QueueProperties QProps = cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder;

  void dump(std::ostream& _out) {
    _out << "data size = " << DataSize << "\n";
    _out << "hist size = " << HistSize << "\n";
    _out << "global size = " << GlobalSize << "\n";
    _out << "local size = " << LSZ << "\n";
  }
};

static std::ostream& operator<<(std::ostream& _out, Config _cfg) {
  _cfg.dump(_out);
  return _out;
}

template <typename It>
void randInit(It _start, It _end, int _low, int _up) {
  static std::mt19937_64 mt_source;
  std::uniform_int_distribution<int> dist(_low, _up);
  for (It cur = _start; cur != _end; ++cur)
    *cur = dist(mt_source);
}

void hist_ref(const unsigned char* _Data, int _DataSize, int* _Hist, int _HistSize) {
  for (int i = 0; i < _DataSize; ++i) {
    assert(_Data[i] < _HistSize && _Data[i] >= 0);
    _Hist[_Data[i]] += 1;
  }
}

class OclApp {

public:
  OclApp(Config _Cfg) :
    m_P(selectPlatform()),
    m_C(getGpuContext(m_P())),
    m_Q(m_C, _Cfg.QProps),
    m_Cfg(_Cfg)
  {
    cl::string name = m_P.getInfo<CL_PLATFORM_NAME>();
    cl::string profile = m_P.getInfo<CL_PLATFORM_PROFILE>();
    dbgs << "Selected: " << name << ": " << profile << std::endl;
  }

  int lsz() const { return m_Cfg.LSZ; }


  cl::Event OclApp::hist(const unsigned char* _Data, int _DataSize, int* _Hist, int _HistSize) {
    size_t DataBufSize = _DataSize * sizeof(unsigned char);
    size_t HistBufSize = _HistSize * sizeof(int);

    cl::Buffer D(m_C, CL_MEM_READ_ONLY, DataBufSize);
    cl::Buffer H(m_C, CL_MEM_WRITE_ONLY, HistBufSize);

    cl::copy(m_Q, _Data, _Data + _DataSize, D);

    cl::Program program(m_C, hist_kernel, true);

    hist_t hist(program, "histogram");

    cl::NDRange GlobalRange(m_Cfg.GlobalSize * DATABLOCK);
    cl::NDRange LocalRange(lsz());
    cl::EnqueueArgs Args(m_Q, GlobalRange, LocalRange);

    cl::Event Evt = hist(Args, D, _DataSize, H, cl::Local(_HistSize * sizeof(int)), _HistSize);
    Evt.wait();

    cl::copy(m_Q, H, _Hist, _Hist + _HistSize);

    return Evt;
  }

private:

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

private:

  cl::Platform m_P;
  cl::Context m_C;
  cl::CommandQueue m_Q;
  Config m_Cfg;

  using hist_t = cl::KernelFunctor<cl::Buffer, cl_int, cl::Buffer, cl::LocalSpaceArg, cl_int>;
};

int main(int argc, char** argv) try {
  std::chrono::high_resolution_clock::time_point TimeStart, TimeFin;
  cl_ulong GPUTimeStart, GPUTimeFin;
  long Dur, GDur;
  Config Cfg;
  dbgs << "Config:\n" << Cfg << std::endl;

  OclApp App(Cfg);
  cl::vector<unsigned char> Data(Cfg.DataSize * DATABLOCK);
  cl::vector<int> Hist(Cfg.HistSize);

  randInit(Data.begin(), Data.end(), 0, Cfg.HistSize - 1);

  TimeStart = std::chrono::high_resolution_clock::now();
  cl::Event evt = App.hist(Data.data(), Data.size(), Hist.data(), Cfg.HistSize);
  TimeFin = std::chrono::high_resolution_clock::now();
  Dur =
    std::chrono::duration_cast<std::chrono::milliseconds>(TimeFin - TimeStart)
    .count();
  std::cout << "GPU wall time measured: " << Dur << " ms" << std::endl;
  GPUTimeStart = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  GPUTimeFin = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  GDur = (GPUTimeFin - GPUTimeStart) / 1000000; // ns -> ms
  std::cout << "GPU pure time measured: " << GDur << " ms" << std::endl;

  cl::vector<int> HistCPU(Cfg.HistSize);
  TimeStart = std::chrono::high_resolution_clock::now();
  hist_ref(Data.data(), Data.size(), HistCPU.data(), Cfg.HistSize);
  TimeFin = std::chrono::high_resolution_clock::now();
  Dur = std::chrono::duration_cast<std::chrono::milliseconds>(TimeFin - TimeStart).count();
  std::cout << "CPU time measured: " << Dur << " ms" << std::endl;

  for (int i = 0; i < Cfg.HistSize; ++i) {
    auto lhs = Hist[i];
    auto rhs = HistCPU[i];
    if (lhs != rhs) {
      std::cerr << "Error at index " << i << ": " << lhs << " != " << rhs
        << std::endl;
      return -1;
    }
  }

  dbgs << "All checks passed" << std::endl;
}
catch (cl::BuildError& err) {
  std::cerr << "OCL BUILD ERROR: " << err.err() << ":" << err.what()
    << std::endl;
  std::cerr << "-- Log --\n";
  for (auto e : err.getBuildLog())
    std::cerr << e.second;
  std::cerr << "-- End log --\n";
  return -1;
}
catch (cl::Error& err) {
  std::cerr << "OCL ERROR: " << err.err() << ":" << err.what() << std::endl;
  return -1;
}
catch (std::runtime_error& err) {
  std::cerr << "RUNTIME ERROR: " << err.what() << std::endl;
  return -1;
}
catch (...) {
  std::cerr << "UNKNOWN ERROR\n";
  return -1;
}
