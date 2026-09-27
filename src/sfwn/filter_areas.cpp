// Report a cloud's Voronoi-area distribution and, optionally, write a copy
// with the degenerate tail removed.
//
// The area a point carries sets its local smoothing scale, so a point with a
// near-zero area behaves as a near-singular spike: on Church the smallest area
// is 1.45e-7 against a median of 4.98e-4, and once the majorant estimator
// probes every point it finds those spikes and the resulting bound is two
// orders of magnitude above anything the image depends on. Such points are
// reconstruction artifacts - near-duplicates whose Voronoi cell collapses -
// rather than geometry, so dropping them is a statement about the data, not a
// numerical convenience.
//
//   sfwn_filter_areas POINTS.ply t_divisor [ratio] [OUT.ply]
//
// `ratio` is a fraction of the MEDIAN area; a point below it is dropped.
// With no OUT.ply the tool only reports, changing nothing.
#include <mitsuba/render/sfwn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using mitsuba::SfwnField;

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr,
                     "Usage: sfwn_filter_areas POINTS.ply t_divisor [ratio] "
                     "[OUT.ply]\n");
        return 2;
    }
    const std::string in = argv[1];
    const double t_divisor = std::stod(argv[2]);
    const double ratio = argc >= 4 ? std::stod(argv[3]) : 0.0;
    const std::string out = argc >= 5 ? argv[4] : std::string();

    auto field = SfwnField::load(in, false, t_divisor, 1.0, "exact", "gaussian");
    std::vector<double> areas = field->point_areas();
    if (areas.empty()) {
        std::fprintf(stderr, "error: no points\n");
        return 1;
    }

    std::vector<double> sorted = areas;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double q) {
        return sorted[std::min(sorted.size() - 1,
                               std::size_t(q * double(sorted.size() - 1)))];
    };
    const double median = pct(0.5);
    std::printf("points=%zu\n", areas.size());
    std::printf("area percentiles: min=%.6g p0.1=%.6g p1=%.6g p50=%.6g "
                "p99=%.6g max=%.6g\n",
                sorted.front(), pct(0.001), pct(0.01), median, pct(0.99),
                sorted.back());
    std::printf("median/min = %.4g\n", median / std::max(sorted.front(), 1e-300));

    // How much of the tail each candidate threshold would remove.
    for (double r : { 1e-4, 1e-3, 1e-2, 0.05, 0.1 }) {
        const double cut = r * median;
        const auto n = std::size_t(
            std::lower_bound(sorted.begin(), sorted.end(), cut) - sorted.begin());
        std::printf("  ratio %-7g cut=%-12.6g drops %zu points (%.4f%%)\n", r,
                    cut, n, 100.0 * double(n) / double(areas.size()));
    }

    if (ratio > 0.0 && !out.empty()) {
        const std::size_t kept = field->write_filtered_ply(out, ratio * median);
        std::printf("wrote %s: kept %zu of %zu points (dropped %zu below %.6g)\n",
                    out.c_str(), kept, areas.size(), areas.size() - kept,
                    ratio * median);
    }
    return 0;
}
