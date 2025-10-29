#include <queue>
#include <functional>
#include <cstdint>
#include <cassert>
#include <future> 
#include <chrono>

class EventQueue {
public:
    enum class OperationType {
        COMPUTE,
        DPU_TRANSFER,
        HOST_TRANSFER,
        FENCE
    };

    using Event = std::pair<OperationType, std::function<void()>>;
    
    EventQueue() = default;
    ~EventQueue() = default;


    template <typename Callable>
    void submit(OperationType op, Callable&& cb) {
        Event e(op, std::function<void()>(std::forward<Callable>(cb)));
        operations_.push(std::move(e));
    }
    void wait();
    void process_next();
    void process_events();

    bool has_pending() const { return !operations_.empty(); }
    std::size_t pending_count() const { return operations_.size(); }

private:
    std::queue<Event> operations_;
};