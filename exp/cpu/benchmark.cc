#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <nlohmann/json.hpp>
#include <vector>

#include "common/thread_pool.hh"
#include "core/checkpoint.hh"
#include "kernels/cpu/expert.hh"
#include "kernels/cpu/fp8_matvec.hh"
#ifdef NANOVLLM_CPU_REFERENCE
namespace nanovllm::kernels::cpu::reference {
void expert_forward(ConstTensorView,const ExpertWeights&,TensorView,std::size_t);
}
#endif
using namespace nanovllm;
using namespace nanovllm::kernels::cpu;
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
namespace {
struct Expert {
  std::array<std::vector<std::uint8_t>,3> weight;
  std::array<std::vector<float>,3> scale;
  ExpertWeights views() const {
    auto w=[&](int i,std::size_t rows,std::size_t cols) {
      return ConstTensorView(weight[i].data(),DataType::Float8E4M3FN,
                             TensorLayout({rows,cols}),Device::cpu());
    };
    auto s=[&](int i,std::size_t rows,std::size_t cols) {
      return ConstTensorView(scale[i].data(),DataType::Float32,
                             TensorLayout({rows,cols}),Device::cpu());
    };
    return {w(0,512,2048),w(1,512,2048),w(2,2048,512),
            s(0,4,16),s(1,4,16),s(2,16,4)};
  }
};
using Forward=decltype(&expert_forward);
struct Job {
  const Expert* expert=nullptr;
  const float* input=nullptr;
  std::array<float,2048> output{};
  Forward forward=&expert_forward;
  static void run(void* context) {
    auto& job=*static_cast<Job*>(context);
    job.forward(ConstTensorView(job.input,DataType::Float32,
                                 TensorLayout({1,2048}),Device::cpu()),
                job.expert->views(),
                TensorView(job.output.data(),DataType::Float32,
                             TensorLayout({1,2048}),Device::cpu()),128);
  }
  static void noop(void*) {}
};
double median(std::vector<double> v) {
  std::sort(v.begin(),v.end());return v[v.size()/2];
}
}
int main(int argc,char** argv) {
 try {
  if(argc<3) throw std::invalid_argument("cpu-bench MODEL OUTPUT [rounds=80]");
  const int rounds=argc>3?std::stoi(argv[3]):80;
  if(rounds<1) throw std::invalid_argument("rounds must be positive");
  Checkpoint checkpoint(argv[1]);std::vector<Expert> experts(64);
  const std::array<std::string,3> names={"gate_proj","up_proj","down_proj"};
  for(std::size_t id=0;id<experts.size();++id) for(int p=0;p<3;++p) {
    const auto name="model.language_model.layers.0.mlp.experts."+
                    std::to_string(id)+"."+names[p]+".weight";
    const auto raw=checkpoint.raw_data(name);
    if(raw.size()!=2048*512) throw std::runtime_error("expected Qwen H2048/I512");
    auto& w=experts[id].weight[p];w.resize(raw.size());
    std::memcpy(w.data(),raw.data(),raw.size());
    const auto sn=name+"_scale_inv";
    const auto source=checkpoint.raw_data(sn);const auto dtype=checkpoint.tensor(sn).dtype;
    auto& s=experts[id].scale[p];s.resize(64);
    if(dtype==DataType::Float32 && source.size()==s.size()*4) {
      std::memcpy(s.data(),source.data(),source.size());
    } else if(dtype==DataType::BFloat16 && source.size()==s.size()*2) {
      for(std::size_t i=0;i<s.size();++i) {
        std::uint16_t bits;std::memcpy(&bits,source.data()+2*i,2);
        s[i]=std::bit_cast<float>(std::uint32_t(bits)<<16);
      }
    } else throw std::runtime_error("unsupported expert scales");
  }
  std::array<float,2048> input;
  for(std::size_t i=0;i<input.size();++i) input[i]=float(int(i%31)-15)/32;
  Json report={{"isa",fp8_cpu_isa()},{"experts",experts.size()},
               {"weights_mib",192},{"rounds",rounds},{"cases",Json::array()}};
  std::vector<std::pair<std::string,Forward>> backends={{"current",expert_forward}};
#ifdef NANOVLLM_CPU_REFERENCE
  backends.push_back({"before",reference::expert_forward});
  // Exact differential checks on real weights, including prefill batches.
  for(std::size_t id=0;id<64;++id) {
    Job a{&experts[id],input.data(),{},expert_forward};
    Job b{&experts[id],input.data(),{},reference::expert_forward};
    Job::run(&a);Job::run(&b);
    if(std::memcmp(a.output.data(),b.output.data(),sizeof(a.output))) {
      throw std::runtime_error("real expert output differs: "+std::to_string(id));
    }
  }
  report["bitwise_experts_checked"]=64;
#endif
  for(const bool pinned:{false,true}) for(const unsigned workers:{6,7,8}) {
    ThreadPool pool(workers,pinned);auto affinity=pool.bind_caller();
    for(const unsigned jobs:{1,2,4,8}) for(const bool rotating:{false,true}) {
      std::array<Job,8> context;
      std::array<ThreadPool::Task,8> descriptors;
      for(unsigned j=0;j<jobs;++j) {context[j].input=input.data();descriptors[j]={Job::run,&context[j]};}
      std::vector<std::vector<double>> times(backends.size());
      // Interleave variants to reduce CPU temperature/clock bias.
      for(int round=-12;round<rounds;++round) {
        for(std::size_t order=0;order<backends.size();++order) {
          const auto b=(order+(round+12)%backends.size())%backends.size();
          for(unsigned j=0;j<jobs;++j) {
            context[j].expert=&experts[rotating ? ((round+12)*jobs+j)%experts.size():0];
            context[j].forward=backends[b].second;
          }
          const auto begin=Clock::now();
          if(b==0) {auto group=pool.submit_batch({descriptors.data(),jobs});group.wait();}
          else {
            std::vector<std::future<void>> futures;futures.reserve(jobs);
            for(unsigned j=0;j<jobs;++j) futures.push_back(pool.submit([&,j]{Job::run(&context[j]);}));
            for(auto& future:futures) future.get();
          }
          const auto us=std::chrono::duration<double,std::micro>(Clock::now()-begin).count();
          if(round>=0) times[b].push_back(us);
        }
      }
      for(std::size_t b=0;b<backends.size();++b) {
        Json result={{"backend",backends[b].first},{"workers",workers},
                     {"pinned",pinned},{"jobs",jobs},{"rotating",rotating},
                     {"median_us",median(times[b])},{"samples_us",times[b]}};
        report["cases"].push_back(result);
        std::cout<<backends[b].first<<" workers="<<workers<<" pinned="<<pinned
                 <<" jobs="<<jobs<<" rotating="<<rotating<<" us="<<median(times[b])<<std::endl;
      }
    }
    for(bool bulk:{false,true}) {
      std::array<ThreadPool::Task,8> descriptors;
      for(auto& d:descriptors)d={Job::noop,nullptr};
      std::vector<double> times;
      for(int i=0;i<1000;++i) {
        const auto begin=Clock::now();
        if(bulk){auto group=pool.submit_batch(descriptors);group.wait();}
        else {
          std::array<std::future<void>,8> futures;
          for(auto& f:futures)f=pool.submit([]{});
          for(auto& f:futures)f.get();
        }
        times.push_back(std::chrono::duration<double,std::micro>(Clock::now()-begin).count());
      }
      report["dispatch"].push_back({{"workers",workers},{"pinned",pinned},
                                     {"bulk",bulk},{"median_us",median(times)}});
    }
  }
  std::ofstream(argv[2])<<report.dump(2)<<'\n';
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
