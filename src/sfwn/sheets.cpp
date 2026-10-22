// How far is the nearest *opposing* piece of surface?
//
// The Gram is an unsigned sum of squared kernel responses, so every piece of
// surface within about one kernel width adds to it, however it is oriented.
// The signed mean, by contrast, cancels between opposed faces. That predicts a
// threshold: once t exceeds the gap between the two faces of a wall, the
// kernel ball starts swallowing both, the Gram jumps while the mean does not,
// and the prior's variance (priorVariance(t) - Gram) heads for zero.
//
// Measured within one object, Gram*t rises with t on church and pantheon and
// falls on David. This tool tests whether the opposing-face gap explains that:
// the prediction is that each object's gap sits near the t at which its ratio
// crosses 1 - small for thin architectural shells, enormous for a solid body.
//
// No field evaluation. Positions and unit normals go into a uniform grid and
// the search is a widening ring of cells, so cost is set by the gap rather
// than by the point count.
#include <mitsuba/render/sfwn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using mitsuba::SfwnField;

namespace {

using Vec = std::array<double, 3>;
using Key = std::array<long long, 3>;

struct KeyHash {
    std::size_t operator()(const Key &k) const noexcept {
        // Three odd primes: cheap and good enough for a scratch grid.
        return std::size_t(k[0] * 73856093LL ^ k[1] * 19349663LL ^
                           k[2] * 83492791LL);
    }
};

double dot(const Vec &a, const Vec &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double dist(const Vec &a, const Vec &b) {
    const double x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
    return std::sqrt(x * x + y * y + z * z);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: sfwn_sheets POINTS.ply t_divisor [samples] "
                     "[cos_threshold] [max_rings]\n";
        return 2;
    }
    const double t_divisor = std::stod(argv[2]);
    const std::size_t samples = argc >= 4 ? std::stoull(argv[3]) : 20000;
    // How opposed a face has to be to count. -0.5 is 120 degrees apart, which
    // admits the far side of a wall and rejects the local curvature of one.
    const double cos_max = argc >= 5 ? std::stod(argv[4]) : -0.5;
    const int max_rings = argc >= 6 ? std::stoi(argv[5]) : 24;

    auto field = SfwnField::load(argv[1], false, t_divisor, 1.0, "exact",
                                 "gaussian", "sensitivity");
    const double t = field->regularization_parameter();

    std::vector<Vec> p, n;
    field->point_frames(p, n);
    const std::size_t count = p.size();
    if (count < 2) {
        std::cerr << "too few points\n";
        return 1;
    }

    // Cell size t: the quantity of interest is the gap in units of t, so a
    // ring of cells is a ring of kernel widths.
    const double cell = t > 0.0 ? t : 1.0;
    std::unordered_map<Key, std::vector<std::uint32_t>, KeyHash> grid;
    grid.reserve(count / 4 + 1);
    const auto key_of = [&](const Vec &v) {
        return Key{ (long long)std::floor(v[0] / cell),
                    (long long)std::floor(v[1] / cell),
                    (long long)std::floor(v[2] / cell) };
    };
    for (std::size_t i = 0; i < count; ++i)
        grid[key_of(p[i])].push_back(std::uint32_t(i));

    std::mt19937_64 rng(4242);
    std::uniform_int_distribution<std::size_t> pick(0, count - 1);
    std::vector<double> gaps;
    gaps.reserve(samples);
    std::size_t unfound = 0;

    for (std::size_t s = 0; s < samples; ++s) {
        const std::size_t i = pick(rng);
        if (n[i][0] == 0.0 && n[i][1] == 0.0 && n[i][2] == 0.0)
            continue;
        const Key k0 = key_of(p[i]);
        double best = std::numeric_limits<double>::infinity();
        for (int ring = 0; ring <= max_rings; ++ring) {
            // Everything closer than this ring has already been seen, so once
            // a hit beats the ring's inner radius it cannot be improved on.
            if (std::isfinite(best) && best <= double(ring - 1) * cell)
                break;
            for (int dx = -ring; dx <= ring; ++dx)
                for (int dy = -ring; dy <= ring; ++dy)
                    for (int dz = -ring; dz <= ring; ++dz) {
                        if (std::max({ std::abs(dx), std::abs(dy),
                                       std::abs(dz) }) != ring)
                            continue;   // shell only; the interior is done
                        auto it = grid.find(Key{ k0[0] + dx, k0[1] + dy,
                                                 k0[2] + dz });
                        if (it == grid.end())
                            continue;
                        for (const auto j : it->second) {
                            if (j == i)
                                continue;
                            if (dot(n[i], n[j]) > cos_max)
                                continue;   // not an opposing face
                            best = std::min(best, dist(p[i], p[j]));
                        }
                    }
        }
        if (std::isfinite(best))
            gaps.push_back(best);
        else
            ++unfound;
    }

    std::cout << "# " << argv[1] << "\n# points=" << count
              << " t_divisor=" << t_divisor << " t=" << t
              << " cos_threshold=" << cos_max << " sampled=" << samples
              << " no_opposing_face_within_" << max_rings << "t=" << unfound
              << '\n';
    if (gaps.empty()) {
        std::cout << "  no opposing face found for any sample\n";
        return 0;
    }
    std::sort(gaps.begin(), gaps.end());
    const auto q = [&](double f) {
        return gaps[std::min(gaps.size() - 1,
                             std::size_t(f * double(gaps.size())))];
    };
    std::cout << std::setprecision(4);
    std::cout << "  gap to nearest opposing face (world units):"
              << "  p5=" << q(0.05) << "  median=" << q(0.5)
              << "  p95=" << q(0.95) << '\n';
    std::cout << "  the same in units of t:                   "
              << "  p5=" << q(0.05) / t << "  median=" << q(0.5) / t
              << "  p95=" << q(0.95) / t << '\n';
    std::cout << "  fraction with an opposing face inside t:  "
              << double(std::lower_bound(gaps.begin(), gaps.end(), t) -
                        gaps.begin()) / double(gaps.size())
              << '\n';
    return 0;
}
