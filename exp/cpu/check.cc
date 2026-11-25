#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <latch>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/thread_pool.hh"
#include "kernels/cpu/expert.hh"
#include "kernels/cpu/fp8_decode.hh"
#include "kernels/cpu/fp8_dequant.hh"
#include "kernels/cpu/fp8_dot.hh"
#include "kernels/cpu/fp8_matvec.hh"

#ifdef NANOVLLM_CPU_REFERENCE
namespace nanovllm::kernels::cpu::reference {
void expert_forward(ConstTensorView,const ExpertWeights&,TensorView,std::size_t);
}
#endif
using namespace nanovllm;
using namespace nanovllm::kernels::cpu;
namespace {
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
bool close(float a, float b) {
  return std::abs(a-b) <= 1e-5f*std::max(1.0f,std::abs(b));
}
bool equal(float a, float b) {
  return (std::isnan(a) && std::isnan(b)) ||
         std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}
// Scalar emulation of the previous AVX2 reduction, independent of SIMD decode.
float reference(const std::uint8_t* w, const float* s, const float* x,
                std::size_t n, std::size_t block) {
  float a[8]{}, b[8]{}, tail = 0;
  for (std::size_t start = 0; start < n; start += block) {
    const auto end = std::min(start + block, n);
    const auto scale = s[start / block];
    auto col = start;
    for (; col + 16 <= end; col += 16) {
      for (int i = 0; i < 8; ++i) {
        a[i] = std::fma(fp8_e4m3fn_to_float(w[col+i])*scale, x[col+i], a[i]);
        b[i] = std::fma(fp8_e4m3fn_to_float(w[col+8+i])*scale, x[col+8+i], b[i]);
      }
    }
    for (; col + 8 <= end; col += 8) {
      for (int i = 0; i < 8; ++i) {
        a[i] = std::fma(fp8_e4m3fn_to_float(w[col+i])*scale, x[col+i], a[i]);
      }
    }
    for (; col < end; ++col) {
      tail = std::fma(fp8_e4m3fn_to_float(w[col])*scale, x[col], tail);
    }
  }
  for (int i=0;i<8;++i) a[i] += b[i];
  for (int i=0;i<4;++i) a[i] += a[i+4];
  return ((a[0]+a[1])+(a[2]+a[3]))+tail;
}
void check_kernels() {
  // Exhaust all byte encodings, including subnormals, NaNs and signed zero.
  std::array<std::uint8_t, 16> bytes;
  alignas(32) float decoded[8];
  for (unsigned first=0; first<256; first+=8) {
    for (unsigned i=0;i<8;++i) bytes[i]=first+i;
    _mm256_store_ps(decoded, detail::decode_fp8_avx2(bytes.data()));
    for (unsigned i=0;i<8;++i) {
      require(equal(decoded[i],fp8_e4m3fn_to_float(bytes[i])), "FP8 decode");
    }
  }
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> value(-0.5f,0.5f);
  for (std::size_t k : {1,7,8,15,16,17,127,128,129,256,511,512,2048}) {
    for (std::size_t block : {7,128}) {
      const std::size_t rows=137, blocks=(k+block-1)/block;
      std::vector<std::uint8_t> w(rows*k), u(rows*k);
      std::vector<float> s(((rows+block-1)/block)*blocks), us(s.size());
      std::vector<float> x(k), y(rows), activation(rows);
      for(auto& v:w) {v=rng()%256;if((v&127)==127) v=0;}
      for(auto& v:u) {v=rng()%256;if((v&127)==127) v=0;}
      for(auto& v:s) v=value(rng)/128;
      for(auto& v:us) v=value(rng)/128;
      for(auto& v:x) v=value(rng);
      fp8_matvec(w.data(),s.data(),x.data(),rows,k,block,y.data());
      fp8_gate_up(w.data(),u.data(),s.data(),us.data(),x.data(),rows,k,
                   block,activation.data());
      for(std::size_t row=0;row<rows;++row) {
        const auto* scale=s.data()+(row/block)*blocks;
        const auto g=reference(w.data()+row*k,scale,x.data(),k,block);
        const auto up=reference(u.data()+row*k,us.data()+(row/block)*blocks,
                                 x.data(),k,block);
        const bool full = block==128 && k%128==0;
        if(!(full ? equal(y[row],g) : close(y[row],g))) {
          std::cerr<<"matvec k="<<k<<" block="<<block<<" row="<<row
                   <<" got="<<y[row]<<" expected="<<g<<" full="<<full<<'\n';
          throw std::runtime_error("matvec reference");
        }
        const auto expected=(g/(1+std::exp(-g)))*up;
        require(full ? equal(activation[row],expected) :
                       close(activation[row],expected), "Gate/Up reference");
      }
    }
  }
  // Exercise every encoding in the tiled AVX-512/AVX2 path as well.
  std::vector<std::uint8_t> w(8*256);
  std::vector<float> x(256,0), y(8), scale(2,1);
  for(unsigned encoding=0;encoding<256;++encoding) {
    std::fill(w.begin(),w.end(),std::uint8_t(0));
    for(unsigned r=0;r<8;++r) w[r*256]=encoding;
    x[0]=1;
    fp8_matvec(w.data(),scale.data(),x.data(),8,256,128,y.data());
    for(auto v:y) require(equal(v,reference(w.data(),scale.data(),x.data(),256,128)),
                          "all FP8 encodings in tiled matvec");
  }
}
void check_experts() {
  std::mt19937 rng(7);
  for (const auto dims : {std::pair{17u,5u}, std::pair{128u,129u},
                          std::pair{2048u,512u}}) {
    const std::size_t h=dims.first, mid=dims.second;
    const std::size_t hb=(h+127)/128, ib=(mid+127)/128;
    std::vector<std::uint8_t> gate(h*mid),up(h*mid),down(h*mid);
    for(auto* weights:{&gate,&up,&down}) for(auto& v:*weights) {
      v=std::uint8_t(rng()%256);if((v&127)==127)v=0;
    }
    std::vector<float> gs(hb*ib,0.001f),us(hb*ib,0.002f),ds(hb*ib,0.003f);
    const auto view=[](const void* p,DataType dtype,std::size_t r,std::size_t c) {
      return ConstTensorView(p,dtype,TensorLayout({r,c}),Device::cpu());
    };
    ExpertWeights weights{view(gate.data(),DataType::Float8E4M3FN,mid,h),
                          view(up.data(),DataType::Float8E4M3FN,mid,h),
                          view(down.data(),DataType::Float8E4M3FN,h,mid),
                          view(gs.data(),DataType::Float32,ib,hb),
                          view(us.data(),DataType::Float32,ib,hb),
                          view(ds.data(),DataType::Float32,hb,ib)};
    for(std::size_t tokens:{1,2,3,4,7}) {
      std::vector<float> x(tokens*h),y(tokens*h),expected(tokens*h),activated(mid);
      for(auto& v:x)v=float(int(rng()%31)-15)/32;
      expert_forward(view(x.data(),DataType::Float32,tokens,h),weights,
                     TensorView(y.data(),DataType::Float32,
                                  TensorLayout({tokens,h}),Device::cpu()),128);
      std::vector<std::uint16_t> bf16_input(x.size());
      for (std::size_t i = 0; i < x.size(); ++i) {
        bf16_input[i] = std::bit_cast<std::uint32_t>(x[i]) >> 16;
      }
      std::vector<float> bf16_output(y.size());
      expert_forward(view(bf16_input.data(), DataType::BFloat16, tokens, h),
                     weights,
                     TensorView(bf16_output.data(), DataType::Float32,
                                TensorLayout({tokens, h}), Device::cpu()),
                     128);
      for (std::size_t i = 0; i < y.size(); ++i) {
        require(equal(bf16_output[i], y[i]), "BF16 expert input parity");
      }
#ifdef NANOVLLM_CPU_REFERENCE
      std::vector<float> previous(y.size());
      nanovllm::kernels::cpu::reference::expert_forward(
          view(x.data(),DataType::Float32,tokens,h),weights,
          TensorView(previous.data(),DataType::Float32,
                       TensorLayout({tokens,h}),Device::cpu()),128);
      for(std::size_t i=0;i<y.size();++i) {
        require(equal(y[i],previous[i]),"previous full expert bitwise comparison");
      }
#endif
      for(std::size_t t=0;t<tokens;++t) {
        for(std::size_t row=0;row<mid;++row) {
          const auto offset=(row/128)*hb;
          const auto g=reference(gate.data()+row*h,gs.data()+offset,
                                   x.data()+t*h,h,128);
          const auto u=reference(up.data()+row*h,us.data()+offset,
                                   x.data()+t*h,h,128);
          activated[row]=(g/(1+std::exp(-g)))*u;
        }
        for(std::size_t row=0;row<h;++row) {
          expected[t*h+row]=reference(down.data()+row*mid,
              ds.data()+(row/128)*ib,activated.data(),mid,128);
        }
      }
      for(std::size_t i=0;i<y.size();++i) {
        // The existing multi-token scalar tails may be auto-vectorized with
        // different contraction from a single-token dot product.
        const bool full = tokens==1 && h%128==0 && mid%128==0;
        const bool matches = full ? equal(y[i],expected[i]) :
                                   close(y[i],expected[i]);
        if(!matches) {
          std::cerr<<"H="<<h<<" I="<<mid<<" tokens="<<tokens<<" index="<<i
                   <<" got="<<std::hex<<std::bit_cast<std::uint32_t>(y[i])
                   <<" expected="<<std::bit_cast<std::uint32_t>(expected[i])
                   <<std::dec<<'\n';
          throw std::runtime_error("full decode/prefill expert reference");
        }
      }
    }
  }
}
void pool() {
  std::atomic<unsigned> count=0;
  std::array<ThreadPool::Task, 31> jobs;
  for(auto& task:jobs) task={[](void* p){++*static_cast<std::atomic<unsigned>*>(p);},&count};
  {
    ThreadPool pool(4);
    auto empty=pool.submit_batch({});empty.wait();
    for(int i=0;i<100;++i) {
      ThreadPool::BusyScope active(pool, i % 2 == 0);
      ThreadPool::BusyScope nested(pool, i % 3 == 0);
      auto group=pool.submit_batch(jobs);
      auto single=pool.submit([&]{++count;});
      group.wait();single.get();group.wait();
    }
    require(count==3200,"mixed single and bulk tasks");
    ThreadPool::Task fail{[](void*){throw std::runtime_error("expected");},nullptr};
    auto group=pool.submit_batch({&fail,1});
    bool caught=false;try{group.wait();}catch(const std::runtime_error&){caught=true;}
    require(caught,"bulk exception propagation");
    {auto old=pool.submit_batch(jobs);auto moved=std::move(old);}
    require(count==3231,"group destructor drains moved tasks");
    auto first=pool.submit_batch(jobs);auto second=pool.submit_batch(jobs);
    first=std::move(second);first.wait();
    require(count==3293,"group assignment drains old tasks");
    // Submit concurrently from multiple producers while sharing the pool.
    std::vector<std::thread> producers;
    for(int i=0;i<4;++i) producers.emplace_back([&]{
      ThreadPool::BusyScope active(pool);
      for(int n=0;n<50;++n) {auto batch=pool.submit_batch(jobs);batch.wait();}
    });
    for(auto& thread:producers) thread.join();
    require(count==9493,"concurrent bulk producers");
    {ThreadPool::BusyScope active(pool);auto group=pool.submit_batch(jobs);throw 1;}
  }
}
}
int main() {
  try {
    check_kernels();
    check_experts();
    // The deliberate unwind also checks TaskGroup argument lifetime.
    try {pool();} catch(int) {}
    std::cout<<"PASS: "<<fp8_cpu_isa()<<" FP8 encodings, reductions, tiles, tails, "
               "bulk tasks and exceptions\n";
  }catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
