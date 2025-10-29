#include "queue.h"

void EventQueue::process_next() {
    auto [type, cb] = operations_.front();  // Peek

    switch (type) {
        case OperationType::FENCE:
            assert(0 && "we don't have fencing yet");
            break;
        case OperationType::COMPUTE:
            break;
        case OperationType::DPU_TRANSFER:
            cb();
            break;
        case OperationType::HOST_TRANSFER:
            cb();
            break;
    }
    operations_.pop();  // Remove
}

void EventQueue::process_events() {
    while (!operations_.empty()) {
        this->process_next();
    }
}