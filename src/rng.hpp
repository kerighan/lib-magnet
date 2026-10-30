#pragma once
#include <cstdint>

// Small, fast, thread-local PRNG (xorshift64*). One instance per thread: no shared state.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) {
        // splitmix64 to spread the seed bits
        uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        s = (z ^ (z >> 31)) | 1ULL;
    }
    inline uint64_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 0x2545F4914F6CDD1DULL;
    }
    // uniform integer in [0, n)
    inline uint32_t below(uint32_t n) {
        return (uint32_t)(((next() >> 32) * (uint64_t)n) >> 32);
    }
    // uniform float in [0, 1)
    inline float uniform() {
        return (float)(next() >> 40) * (1.0f / 16777216.0f);
    }
};
