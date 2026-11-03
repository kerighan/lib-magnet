// Persistent thread pool: parallel_for(nt, f) runs f(t) for t in [0, nt) on pool workers plus the caller.
// Creating and joining std::threads per call costs ~50 us per thread, which dominated the small
// per-iteration kernels of the full-batch optimiser (six calls per iteration). Synchronisation uses the
// pthread primitives directly: std::condition_variable pulls GLIBCXX_3.4.30, absent from conda's
// libstdc++ (the module would not import). The task counter is tagged with the generation so that a worker
// finishing late cannot pick up an index of the next call with a stale job pointer. The pool is created on
// first use and never destroyed (workers are detached; process exit reclaims them).
#pragma once
#include <thread>
#include <pthread.h>
#include <functional>
#include <atomic>
#include <cstdint>

class ThreadPool {
public:
    static ThreadPool& get() {
        static ThreadPool* p = new ThreadPool();
        return *p;
    }
    int size() const { return n_workers_ + 1; }

    void run(int nt, const std::function<void(int)>& f) {
        if (nt <= 1) { f(0); return; }
        if (nt > n_workers_ + 1) nt = n_workers_ + 1;
        uint64_t gen;
        pthread_mutex_lock(&m_);
        gen = ++generation_;
        job_ = &f; nt_ = nt; done_.store(0);
        next_.store((gen << 32) | 1u);
        pthread_cond_broadcast(&cv_);
        pthread_mutex_unlock(&m_);
        f(0);
        for (;;) {   // help with the remaining indices
            const int t = take(gen, nt);
            if (t < 0) break;
            f(t); done_.fetch_add(1);
        }
        pthread_mutex_lock(&m_);
        while (done_.load() < nt - 1) pthread_cond_wait(&cv_done_, &m_);
        job_ = nullptr;
        pthread_mutex_unlock(&m_);
    }

private:
    // next index of generation gen, or -1 when exhausted or when the counter already belongs to a later generation
    int take(uint64_t gen, int nt) {
        uint64_t v = next_.load();
        for (;;) {
            if ((v >> 32) != gen) return -1;
            const int t = (int)(v & 0xffffffffu);
            if (t >= nt) return -1;
            if (next_.compare_exchange_weak(v, v + 1)) return t;
        }
    }
    ThreadPool() {
        pthread_mutex_init(&m_, nullptr); pthread_cond_init(&cv_, nullptr); pthread_cond_init(&cv_done_, nullptr);
        unsigned h = std::thread::hardware_concurrency();
        n_workers_ = (h == 0 ? 4 : (int)h) - 1;
        if (n_workers_ < 1) n_workers_ = 1;
        for (int w = 0; w < n_workers_; ++w) std::thread([this] { worker(); }).detach();
    }
    void worker() {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(int)>* job; int nt; uint64_t gen;
            pthread_mutex_lock(&m_);
            while (generation_ == seen) pthread_cond_wait(&cv_, &m_);
            seen = gen = generation_; job = job_; nt = nt_;
            pthread_mutex_unlock(&m_);
            if (!job) continue;
            for (;;) {
                const int t = take(gen, nt);
                if (t < 0) break;
                (*job)(t);
                if (done_.fetch_add(1) + 1 >= nt - 1) {
                    pthread_mutex_lock(&m_); pthread_cond_broadcast(&cv_done_); pthread_mutex_unlock(&m_);
                }
            }
        }
    }
    int n_workers_ = 1;
    pthread_mutex_t m_;
    pthread_cond_t cv_, cv_done_;
    const std::function<void(int)>* job_ = nullptr;
    int nt_ = 0;
    std::atomic<uint64_t> next_{0};
    std::atomic<int> done_{0};
    uint64_t generation_ = 0;
};

template <class F> inline void pool_run(int nt, F&& f) {
    std::function<void(int)> fn(std::forward<F>(f));
    ThreadPool::get().run(nt, fn);
}
