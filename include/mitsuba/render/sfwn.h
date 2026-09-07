#pragma once

#include <mitsuba/core/platform.h>

#include <array>
#include <cstddef>
#include <memory>
#include <string>

NAMESPACE_BEGIN(mitsuba)

struct SfwnBounds {
    std::array<double, 3> min;
    std::array<double, 3> max;
};

struct SfwnVndfSample {
    std::array<double, 3> normal;
    double pdf;
};

struct SfwnScalarDiagnostics {
    double mean;
    double variance;
    double occupancy;
    double surfaceness;
    double density;
    double sigma;
};

struct SfwnSurfaceSample {
    double value;
    std::array<double, 3> gradient;
};

/** Shared scalar-CPU bridge to the SFWN point-cloud implementation. */
class MI_EXPORT_LIB SfwnField {
public:
    using Ptr = std::shared_ptr<const SfwnField>;

    static Ptr load(const std::string &filename, bool weighted_normals,
                    double t_divisor, double inv_beta,
                    const std::string &model,
                    const std::string &regularization);

    ~SfwnField();

    double sigma(const std::array<double, 3> &position,
                 const std::array<double, 3> &direction) const;
    SfwnScalarDiagnostics diagnostics(
        const std::array<double, 3> &position,
        const std::array<double, 3> &direction) const;
    SfwnSurfaceSample surface_sample(
        const std::array<double, 3> &position, bool use_mean) const;
    double extinction(const std::array<double, 3> &position,
                      const std::array<double, 3> &direction,
                      bool use_surfaceness) const;
    double vndf(const std::array<double, 3> &position,
                const std::array<double, 3> &direction,
                const std::array<double, 3> &micro_normal) const;
    SfwnVndfSample sample_vndf(
        const std::array<double, 3> &position,
        const std::array<double, 3> &direction,
        const std::array<double, 3> &sample) const;

    double estimate_majorant(std::size_t direction_count = 24,
                             std::size_t max_points = 2000,
                             bool use_surfaceness = false,
                             double extinction_offset = 0.0) const;

    const SfwnBounds &bounds() const;
    std::size_t point_count() const;
    double regularization_parameter() const;
    double regularization_divisor() const;
    double inv_beta() const;
    const std::string &volume_model() const;
    const std::string &regularization() const;
    const std::string &filename() const;

private:
    struct Impl;
    explicit SfwnField(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> m_impl;
};

NAMESPACE_END(mitsuba)
