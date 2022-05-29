#include <cassert>
#include <iostream>
#include <random>
#include <sstream>

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

const char* mmult_kernel = STRINGIFY(

__kernel void matrixMultiply(__global int* A, __global int* B, __global int* C, int AX, int AY, int BY)
{
  int k, t;
  const int row = get_local_id(0);
  const int col = get_local_id(1);
  const int globalRow = LS * get_group_id(0) + row;
  const int globalCol = LS * get_group_id(1) + col;

  __local int Asub[LS][LS];
  __local int Bsub[LS][LS];

  int acc = 0;

  const int numTiles = AY / LS;

  for (t = 0; t < numTiles; t++) {
    const int tiledRow = LS * t + row;
    const int tiledCol = LS * t + col;
    Asub[col][row] = A[globalRow * AY + tiledCol];
    Bsub[col][row] = B[tiledRow * BY + globalCol];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (k = 0; k < LS; k++)
      acc += Asub[k][row] * Bsub[col][k];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  C[globalRow * BY + globalCol] = acc;
}
);

struct Config {

  int AX = 256 * 5;
  int AY = 256 * 5;
  int BY = 256 * 5;
  int LSZ = 16;
  cl::QueueProperties QProps = cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder;

  void dump(std::ostream& _out)
  {
    _out << "[" << AX << " x " << AY << "] * ";
    _out << "[" << AY << " x " << BY << "]\n";
    _out << "local size = [" << LSZ << " x " << LSZ << "]";
  }
};

static std::ostream& operator<<(std::ostream& _out, Config _cfg) {
  _cfg.dump(_out);
  return _out;
}

template <typename It>
void randInit(It _start, It _end, int _low, int _up)
{
  static std::mt19937_64 mt_source;
  std::uniform_int_distribution<int> dist(_low, _up);
  for (It cur = _start; cur != _end; ++cur)
    *cur = dist(mt_source);
}

void transposeMult(const int* A, const int* B, int* C, int AX, int AY, int BY)
{
  assert(A != NULL && B != NULL && C != NULL);
  assert(AX > 0 && AY > 0 && BY > 0);
  std::vector<int> tmp(BY * AY);
  int i, j, k;

  for (i = 0; i < AY; i++)
    for (j = 0; j < BY; j++)
      tmp[j * AY + i] = B[i * BY + j];

  for (i = 0; i < AX; i++) {
    for (j = 0; j < BY; j++) {
      int acc = 0;
      for (k = 0; k < AY; k++)
        acc += A[i * AY + k] * tmp[j * AY + k];
      C[i * BY + j] = acc;
    }
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

  int localx() const { return m_Cfg.LSZ; }
  int localy() const { return m_Cfg.LSZ; }

  cl::Event mmult(const int* _A, const int* _B, int* _C, int _AX, int _AY, int _BY)
  {
    size_t ASz = _AX * _AY, ABufSz = ASz * sizeof(int);
    size_t BSz = _AY * _BY, BBufSz = BSz * sizeof(int);
    size_t CSz = _AX * _BY, CBufSz = CSz * sizeof(int);

    cl::Buffer A(m_C, CL_MEM_READ_ONLY, ABufSz);
    cl::Buffer B(m_C, CL_MEM_READ_ONLY, BBufSz);
    cl::Buffer C(m_C, CL_MEM_WRITE_ONLY, CBufSz);

    cl::copy(m_Q, _A, _A + ASz, A);
    cl::copy(m_Q, _B, _B + ASz, B);

    std::ostringstream kernel;
    kernel << "#define LS " << m_Cfg.LSZ << "\n" << mmult_kernel;

    cl::Program program(m_C, kernel.str().data(), true);

    mmult_t gemm(program, "matrixMultiply");

    cl::NDRange GlobalRange(_AX, _BY);
    cl::NDRange LocalRange(localx(), localy());
    cl::EnqueueArgs Args(m_Q, GlobalRange, LocalRange);

    cl::Event Evt = gemm(Args, A, B, C, _AX, _AY, _BY);
    Evt.wait();

    cl::copy(m_Q, C, _C, _C + CSz);

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

  using mmult_t = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int>;
};

int main() try {

  std::chrono::high_resolution_clock::time_point TimeStart, TimeFin;
  cl_ulong GPUTimeStart, GPUTimeFin;
  long Dur, GDur;
  Config Cfg;
  dbgs << "Config:\n" << Cfg << std::endl;

  OclApp App(Cfg);
  cl::vector<int> A(Cfg.AX * Cfg.AY), B(Cfg.AY * Cfg.BY), C(Cfg.AX * Cfg.BY);

  randInit(A.begin(), A.end(), 0, 10);
  randInit(B.begin(), B.end(), 0, 10);

  TimeStart = std::chrono::high_resolution_clock::now();
  cl::Event Evt = App.mmult(A.data(), B.data(), C.data(), Cfg.AX, Cfg.AY, Cfg.BY);
  TimeFin = std::chrono::high_resolution_clock::now();
  Dur = std::chrono::duration_cast<std::chrono::milliseconds>(TimeFin - TimeStart).count();
  std::cout << "GPU wall time measured: " << Dur << " ms" << std::endl;
  GPUTimeStart = Evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  GPUTimeFin = Evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  GDur = (GPUTimeFin - GPUTimeStart) / 1000000; // ns -> ms
  std::cout << "GPU pure time measured: " << GDur << " ms" << std::endl;


  cl::vector<int> CCPU(Cfg.AX * Cfg.BY);
  TimeStart = std::chrono::high_resolution_clock::now();
  transposeMult(A.data(), B.data(), CCPU.data(), Cfg.AX, Cfg.AY, Cfg.BY);
  TimeFin = std::chrono::high_resolution_clock::now();
  Dur = std::chrono::duration_cast<std::chrono::milliseconds>(TimeFin - TimeStart).count();
  std::cout << "CPU time measured: " << Dur << " ms" << std::endl;

  for (int i = 0; i < Cfg.AX * Cfg.BY; ++i) {
    auto lhs = C[i];
    auto rhs = CCPU[i];
    if (lhs != rhs) {
      std::cerr << "Error at index " << i << ": " << lhs << " != " << rhs
        << std::endl;
      return -1;
    }
  }

  dbgs << "All checks passed" << std::endl;
}
catch (cl::BuildError& err) {
  std::cerr << "OCL BUILD ERROR: " << err.err() << ":" << err.what() << std::endl;
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

