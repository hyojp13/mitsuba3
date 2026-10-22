// Where does a configuration's Gram sit relative to the prior's floor?
//
// The two posteriors fail at opposite ends of one quantity. Sensitivity's
// variance IS the data Gram; the prior's is priorVariance(t) - Gram, with
// priorVariance(t) = pwVar_coeff / t. Density goes as 1/variance wherever
// mean > 0.5, so:
//
//   Gram reaching the floor   -> the prior's variance crosses zero
//   Gram collapsing to zero   -> sensitivity's density diverges
//
// The ratio Gram * t / pwVar_coeff is dimensionless, so it compares across
// objects whose world units differ by two orders of magnitude - which raw t
// values never could.
//
// Samples are stratified by mean rather than by distance from the points: a
// shell around the samples is what the majorant estimator walks, but it only
// reaches the near-surface band, and the sensitivity half of the question
// lives in the deep interior. Bucketing by mean reaches both and needs no
// access to the point positions.
#include <mitsuba/render/sfwn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using mitsuba::SfwnField;

namespace {

// 1 / (2 sqrt(2) pi^(3/2)), the Gaussian prior's winding-number variance
// coefficient: priorVariance(t) = pwVar_coeff / t. Mirrors pwVar_coeff_v in
// point_cloud_utils types.h.
constexpr double kPwVarCoeff = 0.063493635934240969;

struct Bucket {
    const char *name;
    double lo, hi;              // mean range
    std::vector<double> ratio;  // Gram / floor
};

void report(const Bucket &b) {
    if (b.ratio.empty()) {
        std::cout << "  " << std::left << std::setw(16) << b.name
                  << "       (no samples)\n";
        return;
    }
    auto r = b.ratio;
    std::sort(r.begin(), r.end());
    const auto q = [&](double f) {
        return r[std::min(r.size() - 1,
                          static_cast<std::size_t>(f * double(r.size())))];
    };
    std::cout << "  " << std::left << std::setw(16) << b.name << std::right
              << std::setw(9) << r.size() << std::scientific
              << std::setprecision(2) << std::setw(12) << r.front()
              << std::setw(12) << q(0.5) << std::setw(12) << q(0.99)
              << std::setw(12) << r.back() << '\n';
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: sfwn_gram POINTS.ply t_divisor [samples] "
                     "[inv_beta] [regularization]\n";
        return 2;
    }
    const double t_divisor = std::stod(argv[2]);
    const std::size_t samples = argc >= 4 ? std::stoull(argv[3]) : 200000;
    const double inv_beta = argc >= 5 ? std::stod(argv[4]) : 1.0;
    const std::string regularization = argc >= 6 ? argv[5] : "gaussian";

    // Sensitivity, so the reported variance is the Gram itself.
    auto field = SfwnField::load(argv[1], false, t_divisor, inv_beta, "exact",
                                 regularization, "sensitivity");
    const double t = field->regularization_parameter();
    const double floor_value = kPwVarCoeff / t;
    const auto &b = field->bounds();

    std::cout << "# " << argv[1] << "\n# points=" << field->point_count()
              << " t_divisor=" << t_divisor << " t=" << t
              << " floor=" << floor_value << " samples=" << samples << '\n';

    std::vector<Bucket> buckets = {
        { "outside",      -1e30, 0.10, {} },
        { "near-surface",  0.40, 0.60, {} },
        { "inside",        0.90, 1e30, {} },
    };
    std::vector<double> all;
    all.reserve(samples);

    // A fixed seed: this is a measurement that should reproduce.
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const std::array<double, 3> w = { 0.0, 0.0, 1.0 };

    for (std::size_t i = 0; i < samples; ++i) {
        std::array<double, 3> p;
        for (int a = 0; a < 3; ++a)
            p[a] = b.min[a] + (b.max[a] - b.min[a]) * u(rng);
        const auto d = field->diagnostics(p, w);
        if (!std::isfinite(d.variance) || !std::isfinite(d.mean))
            continue;
        const double ratio = d.variance / floor_value;
        all.push_back(ratio);
        for (auto &bucket : buckets)
            if (d.mean >= bucket.lo && d.mean < bucket.hi)
                bucket.ratio.push_back(ratio);
    }

    std::cout << "# Gram / floor, so >= 1 means the prior's variance crosses "
                 "zero there\n";
    std::cout << "  " << std::left << std::setw(16) << "bucket" << std::right
              << std::setw(9) << "n" << std::setw(12) << "min"
              << std::setw(12) << "median" << std::setw(12) << "p99"
              << std::setw(12) << "max" << '\n';
    for (const auto &bucket : buckets)
        report(bucket);
    Bucket every{ "ALL", 0, 0, all };
    report(every);
    return 0;
}
