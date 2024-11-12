#ifndef RAND_INDICES_H
#define RAND_INDICES_H

#include <random>
#include <set>
#include <vector>

constexpr bool Seeded = true;
constexpr int Seed = 42;

// inline void randomInvIndices(std::vector<std::pair<int, int>>& vec, int rows) {
inline void randomInvIndices(std::set<std::pair<int, int>>& set, int rows) {
    std::mt19937 gen;
    if (Seeded) {
        gen.seed(Seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    std::uniform_int_distribution<> dis(0, rows - 1);

    // Determine the random number of elements to add to the vector (between 1 and rows)
    std::uniform_int_distribution<> sizeDis(1, rows / 10);   // Random number of elements from 1 to rows/10
    int numElements = sizeDis(gen);

    std::set<std::pair<int, int>> uniqueNumbers;   // Set to ensure uniqueness

    // Fill the set with unique numbers
    while (uniqueNumbers.size() < numElements) { uniqueNumbers.insert({dis(gen), dis(gen)}); };

    set = uniqueNumbers;

    // Convert the set to a vector (which will automatically be sorted in increasing order)
    // vec.assign(uniqueNumbers.begin(), uniqueNumbers.end());
}

// inline void randomSchurIndices(std::vector<int>& vec, int rows) {
inline void randomSchurIndices(std::set<int>& set, int rows) {
    std::mt19937 gen;
    if (Seeded) {
        gen.seed(Seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    std::uniform_int_distribution<> dis(0, rows - 1);

    // Determine the random number of elements to add to the vector (between 1 and rows)
    std::uniform_int_distribution<> sizeDis(1, rows / 10);   // Random number of elements from 1 to rows/10
    int numElements = sizeDis(gen);

    std::set<int> uniqueNumbers;   // Set to ensure uniqueness

    // Fill the set with unique numbers
    while (uniqueNumbers.size() < numElements) { uniqueNumbers.insert(dis(gen)); }

    set = uniqueNumbers;

    // Convert the set to a vector (which will automatically be sorted in increasing order)
    // vec.assign(uniqueNumbers.begin(), uniqueNumbers.end());
}

#endif   // RAND_INDICES_H
