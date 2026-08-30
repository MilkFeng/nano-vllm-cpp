#include <iostream>
#include <mutex>
#include <nanovllm/nanovllm.hh>

class ConsoleLogger final : public nanovllm::Logger {
 public:
  void log(nanovllm::LogLevel level, std::string_view message) override {
    std::lock_guard lock(mutex_);

    std::cerr << '[' << level_name(level) << "] " << message << '\n';
  }

 private:
  static std::string_view level_name(nanovllm::LogLevel level) noexcept {
    using nanovllm::LogLevel;

    switch (level) {
      case LogLevel::Trace:
        return "TRACE";
      case LogLevel::Debug:
        return "DEBUG";
      case LogLevel::Info:
        return "INFO";
      case LogLevel::Warning:
        return "WARNING";
      case LogLevel::Error:
        return "ERROR";
    }

    return "UNKNOWN";
  }

  std::mutex mutex_;
};

int main() {
  const auto logger = std::make_shared<ConsoleLogger>();

  nanovllm::EngineParams params;
  nanovllm::LLMEngine engine("/home/feng/models/Qwen3-0.6B", params, logger);
}
