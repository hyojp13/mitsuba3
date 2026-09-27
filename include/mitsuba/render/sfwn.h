#pragma once

#include <mitsuba/core/platform.h>

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <cmath>
#include <vector>

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

/**
 * Measured far-field extinction plateau of a field.
 *
 * SFWN extinction does not decay to zero away from the surface: it approaches
 * a nonzero, very nearly isotropic asymptote. `mean` estimates that asymptote,
 * and `min`/`max` bracket the residual spread over the sampled positions and
 * directions. `samples` counts the finite queries that contributed, and is
 * zero if the field produced no usable value at all.
 */
/**
 * Distribution of sampled extinction, for choosing a majorant.
 *
 * The maximum is a true bound over the sampled set but is set by a handful of
 * near-singular points: on Church the max is 24217 while the bulk of the
 * distribution sits orders of magnitude lower. Quantiles let the bound cover
 * the body of the distribution instead, trading a small, *measured* clamping
 * rate for a majorant that delta tracking can actually afford.
 */
struct SfwnMajorantStats {
    double max = 0.0;
    double p50 = 0.0, p90 = 0.0, p99 = 0.0, p999 = 0.0, p9999 = 0.0;
    std::size_t samples = 0;

    /// Log-spaced counts of every sampled value: bin 0 is exactly zero, the
    /// rest cover [log_min, log_max] in equal decades, and the last catches
    /// anything above. Retained so any quantile can be asked for afterwards
    /// rather than only the handful named above.
    std::vector<std::size_t> histogram;
    double log_min = 0.0, log_max = 0.0;

    /// (defined inline so plugins can call it without an exported symbol)
        double quantile(double q) const {
        if (samples == 0 || histogram.empty())
            return 0.0;
        if (q >= 1.0)
            return max;
        // Interior bins only; bin 0 holds exact zeros and the last holds the
        // overflow, both of which the edge formula below would misreport.
        const std::size_t bins = histogram.size() - 2;
        const double scale = double(bins) / (log_max - log_min);
        const auto target = std::size_t(q * double(samples));
        std::size_t seen = 0;
        for (std::size_t b = 0; b < histogram.size(); ++b) {
            seen += histogram[b];
            if (seen >= target) {
                if (b == 0)
                    return 0.0;
                if (b > bins)
                    return max;
                return std::pow(10.0, log_min + double(b) / scale);
            }
        }
        return max;
    }
};

struct SfwnBaseline {
    double mean;
    double min;
    double max;
    std::size_t samples;
};

/** Shared scalar-CPU bridge to the SFWN point-cloud implementation. */
class MI_EXPORT_LIB SfwnField {
public:
    using Ptr = std::shared_ptr<const SfwnField>;

    static Ptr load(const std::string &filename, bool weighted_normals,
                    double t_divisor, double inv_beta,
                    const std::string &model,
                    const std::string &regularization,
                    const std::string &process = "sensitivity");

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

    /**
     * Upper bound on sigma_t over the field, for delta tracking.
     *
     * Sampling only at the cloud's own points underestimates badly: extinction
     * peaks in a shell just *inside* the surface, not on it. On bunny10k with
     * gaussian t_divisor=10 the maximum at the points is ~119 while the true
     * maximum is ~226 -- a factor of 1.9 that a blanket 2x safety factor was
     * only accidentally covering. Each sampled point is therefore also probed
     * at `shell_samples` offsets either side along its normal.
     *
     * The reach is `shell_t_scale * t`, because the extinction peak is a shell
     * roughly `t` thick: scaling it to the bounding box instead (the old
     * `shell_radius` behaviour, kept for `shell_t_scale <= 0`) makes the step
     * 1.9x to 27x wider than that shell on these clouds, so the probe brackets
     * the peak rather than landing on it and the resulting majorant is
     * exceeded at render time.
     */
    /// Per-point Voronoi areas, in the field's own order.
    std::vector<double> point_areas() const;

    /**
     * Write a PLY holding only the points whose area is at least `min_area`.
     *
     * Normals are written as unit vectors, so the copy reloads like any other
     * cloud and has its areas recomputed from the surviving neighbourhood.
     * Returns the number of points kept.
     */
    std::size_t write_filtered_ply(const std::string &filename,
                                   double min_area) const;

    /// Sampled-extinction distribution behind estimate_majorant().
    SfwnMajorantStats majorant_stats(std::size_t direction_count = 16,
                                     std::size_t max_points = 0,
                                     bool use_surfaceness = false,
                                     double extinction_offset = 0.0,
                                     std::size_t shell_samples = 4,
                                     double shell_radius = 0.02,
                                     double shell_t_scale = 3.0) const;

    double estimate_majorant(std::size_t direction_count = 24,
                             std::size_t max_points = 2000,
                             bool use_surfaceness = false,
                             double extinction_offset = 0.0,
                             std::size_t shell_samples = 8,
                             double shell_radius = 0.02,
                             double shell_t_scale = 3.0) const;

    /**
     * Measure the far-field extinction baseline of this field.
     *
     * Samples `position_count` points spread over a sphere of radius
     * `radius_scale` times the field's bounding-sphere radius, each queried
     * along `direction_count` directions. Both point sets are golden-angle
     * spirals. This is the quantity a medium subtracts as `extinction_offset`.
     */
    SfwnBaseline measure_far_field_baseline(
        bool use_surfaceness, std::size_t position_count = 300,
        std::size_t direction_count = 24, double radius_scale = 8.0) const;

    /**
     * Far-field baseline measured once and cached for the lifetime of the
     * field, using the default sampling parameters above.
     *
     * The baseline is a property of the field -- of every parameter in the
     * field cache key -- and additionally of the spatial weight, which is not
     * part of that key. It is therefore cached separately per
     * `use_surfaceness`. Measurement is lazy: fields used only by
     * `sfwnsurface` never pay for it.
     */
    const SfwnBaseline &far_field_baseline(bool use_surfaceness) const;

    /// One-line parameter summary, for logs and mismatch diagnostics.
    std::string describe() const;

    const SfwnBounds &bounds() const;
    std::size_t point_count() const;
    double regularization_parameter() const;
    double regularization_divisor() const;
    double inv_beta() const;
    const std::string &volume_model() const;
    const std::string &regularization() const;
    /// "sensitivity" (no prior term; variance is the data Gram, zero far from
    /// the cloud) or "prior" (GP posterior, prior minus data Gram).
    const std::string &process() const;
    const std::string &filename() const;

private:
    struct Impl;
    explicit SfwnField(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> m_impl;
};

/**
 * Implemented by plugins that own an `SfwnField`, so a sibling plugin can check
 * that both were configured against the same field.
 *
 * Fields are cached and shared, so two plugins given identical parameters hold
 * the *same* `SfwnField`: comparing pointers is an exact comparison of all six
 * field parameters, with no need to re-read or re-compare them individually.
 *
 * This base deliberately lives in libmitsuba rather than in one of the plugin
 * modules, so the cross-module `dynamic_cast` performed by `sfwnmedium` on its
 * phase function has a single exported typeinfo to match against.
 */
class MI_EXPORT_LIB SfwnFieldHolder {
public:
    virtual ~SfwnFieldHolder();
    /// The field this plugin was configured with; never null after construction.
    virtual const SfwnField *sfwn_field() const = 0;

    /**
     * Adopt the enclosing medium's placement, as a row-major 4x4 world-to-object
     * matrix.
     *
     * A phase function is evaluated with world-space directions but queries a
     * field that lives in object space, so it needs the same transform as its
     * medium. Pushing it down here rather than re-declaring `to_world` on the
     * phase keeps the two from drifting apart, in the same spirit as the field
     * comparison above. Implementations must reject a second, different
     * transform: that means one phase function is shared by two differently
     * placed media, which cannot be honoured.
     */
    virtual void set_sfwn_world_to_object(const std::array<double, 16> &m) = 0;
};

NAMESPACE_END(mitsuba)
