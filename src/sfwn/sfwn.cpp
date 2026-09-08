#define TINYPLY_IMPLEMENTATION
#define DRJIT_NO_CORE_LIB

#include <mitsuba/render/sfwn.h>

#include <sfwn/pc.h>
#include <sfwn/octree.h>
#include <sfwn/bh.h>
#include <sfwn/gpis.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <variant>

NAMESPACE_BEGIN(mitsuba)

namespace {

using PointCloud = sfwn::PointCloud<sfwn::Traits<double>>;
using Octree = sfwn::Octree<PointCloud>;
using BarnesHut = sfwn::BarnesHut<Octree>;
using ExactGaussian = sfwn::Gpis<BarnesHut, sfwn::Accumulation::Direct, 32,
                                sfwn::VolumeModel::Exact,
                                sfwn::Regularization::Gaussian>;
using SmithGaussian = sfwn::Gpis<BarnesHut, sfwn::Accumulation::Direct, 32,
                                sfwn::VolumeModel::Smith,
                                sfwn::Regularization::Gaussian>;
using ExactMiyamotoNagai =
    sfwn::Gpis<BarnesHut, sfwn::Accumulation::Direct, 32,
               sfwn::VolumeModel::Exact,
               sfwn::Regularization::MiyamotoNagai>;
using SmithMiyamotoNagai =
    sfwn::Gpis<BarnesHut, sfwn::Accumulation::Direct, 32,
               sfwn::VolumeModel::Smith,
               sfwn::Regularization::MiyamotoNagai>;

template <typename T> using Owned = std::unique_ptr<T>;
using FieldVariant =
    std::variant<Owned<ExactGaussian>, Owned<SmithGaussian>,
                 Owned<ExactMiyamotoNagai>, Owned<SmithMiyamotoNagai>>;

template <typename Gpis>
Owned<Gpis> make_field(const std::string &filename, bool weighted_normals) {
    return std::make_unique<Gpis>(
        filename, weighted_normals,
        sfwn::BuildOptions<sfwn::Threading::Multi, PointCloud::nodeWidth>{});
}

ExactGaussian::Point to_point(const std::array<double, 3> &value) {
    return ExactGaussian::Point(value[0], value[1], value[2]);
}

std::array<double, 3> from_point(const ExactGaussian::Point &value) {
    return { value[0], value[1], value[2] };
}

/// `index`-th point of a golden-angle spiral of `count` points on the unit
/// sphere. Used for both the probe positions and the query directions.
std::array<double, 3> spiral_point(std::size_t index, std::size_t count) {
    constexpr double golden_angle = 2.39996322972865332;
    double z = 1.0 - 2.0 * (double(index) + 0.5) / double(count);
    double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
    double phi = golden_angle * double(index);
    return { radius * std::cos(phi), radius * std::sin(phi), z };
}

std::string cache_key(const std::string &filename, bool weighted_normals,
                      double t_divisor, double inv_beta,
                      const std::string &model,
                      const std::string &regularization) {
    std::ostringstream oss;
    oss << std::filesystem::weakly_canonical(filename).string() << '|'
        << weighted_normals << '|' << std::setprecision(17) << t_divisor << '|'
        << inv_beta << '|' << model << '|' << regularization;
    return oss.str();
}

} // namespace

struct SfwnField::Impl {
    FieldVariant field;
    SfwnBounds bounds;
    std::string filename;
    bool weighted_normals;
    double t;
    double t_divisor;
    double inv_beta;
    std::string model;
    std::string regularization;

    // Lazily measured far-field baselines, indexed by the spatial weight
    // ([0] = density, [1] = surfaceness). Populated at most once each and
    // never reset, so references handed out by far_field_baseline() stay valid.
    mutable std::mutex baseline_mutex;
    mutable std::optional<SfwnBaseline> baseline_cache[2];

    Impl(const std::string &filename_, bool weighted_normals, double t_divisor_,
         double inv_beta_, const std::string &model,
         const std::string &regularization)
        : filename(filename_), weighted_normals(weighted_normals),
          t(0.0), t_divisor(t_divisor_), inv_beta(inv_beta_), model(model),
          regularization(regularization) {
        if (t_divisor <= 0.0 || !std::isfinite(t_divisor))
            throw std::runtime_error(
                "SFWN t_divisor must be finite and positive");
        if (inv_beta < 0.0)
            throw std::runtime_error("SFWN inv_beta must be nonnegative");

        if (model == "exact" && regularization == "gaussian")
            field = make_field<ExactGaussian>(filename, weighted_normals);
        else if (model == "smith" && regularization == "gaussian")
            field = make_field<SmithGaussian>(filename, weighted_normals);
        else if (model == "exact" && regularization == "miyamoto_nagai")
            field = make_field<ExactMiyamotoNagai>(filename, weighted_normals);
        else if (model == "smith" && regularization == "miyamoto_nagai")
            field = make_field<SmithMiyamotoNagai>(filename, weighted_normals);
        else
            throw std::runtime_error(
                "SFWN model must be exact/smith and regularization must be "
                "gaussian/miyamoto_nagai");

        std::visit(
            [&](const auto &owned) {
                if (!owned || owned->dataset().num() == 0)
                    throw std::runtime_error("SFWN failed to load point cloud: " +
                                             filename);
                const auto bbox = owned->dataset().bbox();
                bounds.min = from_point(bbox.low);
                bounds.max = from_point(bbox.high);
                t = owned->areaToRegularizationParameter(
                        owned->dataset().minArea()) /
                    t_divisor;
            },
            field);

        if (!(t > 0.0) || !std::isfinite(t))
            throw std::runtime_error(
                "SFWN produced an invalid regularization parameter");
    }
};

SfwnField::SfwnField(std::unique_ptr<Impl> impl) : m_impl(std::move(impl)) { }
SfwnField::~SfwnField() = default;

SfwnField::Ptr SfwnField::load(const std::string &filename,
                               bool weighted_normals, double t_divisor,
                               double inv_beta, const std::string &model,
                               const std::string &regularization) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::weak_ptr<const SfwnField>> cache;

    std::string key = cache_key(filename, weighted_normals, t_divisor, inv_beta,
                                model, regularization);
    std::lock_guard<std::mutex> lock(mutex);
    if (auto existing = cache[key].lock())
        return existing;

    auto result = std::shared_ptr<const SfwnField>(new SfwnField(
        std::make_unique<Impl>(filename, weighted_normals, t_divisor, inv_beta,
                               model, regularization)));
    cache[key] = result;
    return result;
}

double SfwnField::sigma(const std::array<double, 3> &position,
                        const std::array<double, 3> &direction) const {
    auto p = to_point(position);
    auto w = to_point(direction);
    return std::visit(
        [&](const auto &owned) {
            return owned->template query<sfwn::Output::Sigma>(
                            p, m_impl->t, m_impl->inv_beta, w)
                .sigma;
        },
        m_impl->field);
}

SfwnScalarDiagnostics SfwnField::diagnostics(
    const std::array<double, 3> &position,
    const std::array<double, 3> &direction) const {
    auto p = to_point(position);
    auto w = to_point(direction);
    return std::visit(
        [&](const auto &owned) {
            constexpr sfwn::Output outputs =
                sfwn::Output::Mean | sfwn::Output::Variance |
                sfwn::Output::Occupancy | sfwn::Output::Surfaceness |
                sfwn::Output::Density | sfwn::Output::Sigma;
            auto result = owned->template query<outputs>(
                p, m_impl->t, m_impl->inv_beta, w);
            return SfwnScalarDiagnostics{
                result.mean, result.variance, result.occupancy,
                result.surfaceness, result.density, result.sigma
            };
        },
        m_impl->field);
}

SfwnSurfaceSample SfwnField::surface_sample(
    const std::array<double, 3> &position, bool use_mean) const {
    auto p = to_point(position);
    return std::visit(
        [&](const auto &owned) {
            constexpr sfwn::Output outputs =
                sfwn::Output::Mean | sfwn::Output::Occupancy |
                sfwn::Output::NormalMean;
            auto result = owned->template query<outputs>(
                p, m_impl->t, m_impl->inv_beta);
            const auto &gradient = result.normalMean;
            return SfwnSurfaceSample{
                use_mean ? double(result.mean) : double(result.occupancy),
                { double(gradient[0]), double(gradient[1]),
                  double(gradient[2]) }
            };
        },
        m_impl->field);
}

double SfwnField::extinction(const std::array<double, 3> &position,
                             const std::array<double, 3> &direction,
                             bool use_surfaceness) const {
    auto p = to_point(position);
    auto w = to_point(direction);
    return std::visit(
        [&](const auto &owned) {
            if (use_surfaceness) {
                constexpr sfwn::Output outputs =
                    sfwn::Output::Surfaceness | sfwn::Output::Sigma;
                auto result = owned->template query<outputs>(
                    p, m_impl->t, m_impl->inv_beta, w);
                return result.surfaceness * result.sigma;
            } else {
                constexpr sfwn::Output outputs =
                    sfwn::Output::Density | sfwn::Output::Sigma;
                auto result = owned->template query<outputs>(
                    p, m_impl->t, m_impl->inv_beta, w);
                return result.density * result.sigma;
            }
        },
        m_impl->field);
}

double SfwnField::vndf(const std::array<double, 3> &position,
                       const std::array<double, 3> &direction,
                       const std::array<double, 3> &micro_normal) const {
    auto p = to_point(position);
    auto w = to_point(direction);
    auto m = to_point(micro_normal);
    return std::visit(
        [&](const auto &owned) {
            return owned->template query<sfwn::Output::VNDF>(
                            p, m_impl->t, m_impl->inv_beta, w, m)
                .vndf;
        },
        m_impl->field);
}

SfwnVndfSample SfwnField::sample_vndf(
    const std::array<double, 3> &position,
    const std::array<double, 3> &direction,
    const std::array<double, 3> &sample) const {
    auto p = to_point(position);
    auto w = to_point(direction);
    auto u = to_point(sample);
    return std::visit(
        [&](const auto &owned) {
            auto result = owned->template query<sfwn::Output::VndfSample>(
                p, m_impl->t, m_impl->inv_beta, w, w, u);
            return SfwnVndfSample{ from_point(result.vndfSample),
                                   result.vndfSamplePdf };
        },
        m_impl->field);
}

double SfwnField::estimate_majorant(std::size_t direction_count,
                                    std::size_t max_points,
                                    bool use_surfaceness,
                                    double extinction_offset) const {
    direction_count = std::max<std::size_t>(direction_count, 6);
    max_points = std::max<std::size_t>(max_points, 1);
    constexpr double golden_angle = 2.39996322972865332;
    double maximum = 0.0;

    std::visit(
        [&](const auto &owned) {
            const auto count = static_cast<std::size_t>(owned->dataset().num());
            const auto stride = std::max<std::size_t>(1, count / max_points);
            for (std::size_t i = 0; i < count; i += stride) {
                auto p = owned->dataset().getPoint(i);
                for (std::size_t j = 0; j < direction_count; ++j) {
                    double z = 1.0 - 2.0 * (double(j) + 0.5) /
                                         double(direction_count);
                    double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
                    double phi = golden_angle * double(j);
                    typename std::remove_reference_t<decltype(*owned)>::Point w(
                        radius * std::cos(phi), radius * std::sin(phi), z);
                    double value;
                    if (use_surfaceness) {
                        constexpr sfwn::Output outputs =
                            sfwn::Output::Surfaceness | sfwn::Output::Sigma;
                        auto result = owned->template query<outputs>(
                            p, m_impl->t, m_impl->inv_beta, w);
                        value = result.surfaceness * result.sigma;
                    } else {
                        constexpr sfwn::Output outputs =
                            sfwn::Output::Density | sfwn::Output::Sigma;
                        auto result = owned->template query<outputs>(
                            p, m_impl->t, m_impl->inv_beta, w);
                        value = result.density * result.sigma;
                    }
                    if (std::isfinite(value))
                        maximum = std::max(
                            maximum, std::max(0.0, value - extinction_offset));
                }
            }
        },
        m_impl->field);
    return maximum;
}

SfwnBaseline SfwnField::measure_far_field_baseline(
    bool use_surfaceness, std::size_t position_count,
    std::size_t direction_count, double radius_scale) const {
    position_count = std::max<std::size_t>(position_count, 1);
    direction_count = std::max<std::size_t>(direction_count, 1);

    const auto &b = m_impl->bounds;
    const std::array<double, 3> center = { 0.5 * (b.min[0] + b.max[0]),
                                           0.5 * (b.min[1] + b.max[1]),
                                           0.5 * (b.min[2] + b.max[2]) };
    double dx = b.max[0] - b.min[0], dy = b.max[1] - b.min[1],
           dz = b.max[2] - b.min[2];
    double bounding_radius = 0.5 * std::sqrt(dx * dx + dy * dy + dz * dz);
    if (!(bounding_radius > 0.0) || !std::isfinite(bounding_radius))
        bounding_radius = 1.0;
    const double radius = radius_scale * bounding_radius;

    double sum = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    std::size_t samples = 0;

    std::visit(
        [&](const auto &owned) {
            using Point =
                typename std::remove_reference_t<decltype(*owned)>::Point;
            for (std::size_t i = 0; i < position_count; ++i) {
                auto offset = spiral_point(i, position_count);
                Point p(center[0] + radius * offset[0],
                        center[1] + radius * offset[1],
                        center[2] + radius * offset[2]);
                for (std::size_t j = 0; j < direction_count; ++j) {
                    auto d = spiral_point(j, direction_count);
                    Point w(d[0], d[1], d[2]);
                    double value;
                    if (use_surfaceness) {
                        constexpr sfwn::Output outputs =
                            sfwn::Output::Surfaceness | sfwn::Output::Sigma;
                        auto result = owned->template query<outputs>(
                            p, m_impl->t, m_impl->inv_beta, w);
                        value = result.surfaceness * result.sigma;
                    } else {
                        constexpr sfwn::Output outputs =
                            sfwn::Output::Density | sfwn::Output::Sigma;
                        auto result = owned->template query<outputs>(
                            p, m_impl->t, m_impl->inv_beta, w);
                        value = result.density * result.sigma;
                    }
                    if (!std::isfinite(value))
                        continue;
                    sum += value;
                    minimum = std::min(minimum, value);
                    maximum = std::max(maximum, value);
                    ++samples;
                }
            }
        },
        m_impl->field);

    if (samples == 0)
        return SfwnBaseline{ 0.0, 0.0, 0.0, 0 };
    return SfwnBaseline{ sum / double(samples), minimum, maximum, samples };
}

const SfwnBaseline &SfwnField::far_field_baseline(bool use_surfaceness) const {
    const std::size_t slot = use_surfaceness ? 1u : 0u;
    std::lock_guard<std::mutex> lock(m_impl->baseline_mutex);
    if (!m_impl->baseline_cache[slot])
        m_impl->baseline_cache[slot] =
            measure_far_field_baseline(use_surfaceness);
    return *m_impl->baseline_cache[slot];
}

std::string SfwnField::describe() const {
    std::ostringstream oss;
    oss << std::filesystem::path(m_impl->filename).filename().string()
        << ", model=" << m_impl->model
        << ", regularization=" << m_impl->regularization
        << ", t_divisor=" << m_impl->t_divisor
        << ", t=" << std::setprecision(9) << m_impl->t
        << ", inv_beta=" << m_impl->inv_beta
        << ", weighted_normals=" << (m_impl->weighted_normals ? "true" : "false");
    return oss.str();
}

SfwnFieldHolder::~SfwnFieldHolder() = default;

const SfwnBounds &SfwnField::bounds() const { return m_impl->bounds; }

std::size_t SfwnField::point_count() const {
    return std::visit(
        [](const auto &owned) {
            return static_cast<std::size_t>(owned->dataset().num());
        },
        m_impl->field);
}

double SfwnField::regularization_parameter() const { return m_impl->t; }
double SfwnField::regularization_divisor() const { return m_impl->t_divisor; }
double SfwnField::inv_beta() const { return m_impl->inv_beta; }
const std::string &SfwnField::volume_model() const { return m_impl->model; }
const std::string &SfwnField::regularization() const {
    return m_impl->regularization;
}
const std::string &SfwnField::filename() const { return m_impl->filename; }

NAMESPACE_END(mitsuba)
