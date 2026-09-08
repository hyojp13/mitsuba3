#include <mitsuba/core/bbox.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/sfwn.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class SfwnMedium final : public Medium<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Medium, m_is_homogeneous, m_has_spectral_extinction,
                   m_phase_function)
    MI_IMPORT_TYPES()

    SfwnMedium(const Properties &props) : Base(props) {
        static_assert(!dr::is_jit_v<Float>,
                      "The SFWN medium currently supports scalar variants");
        m_is_homogeneous = false;
        m_has_spectral_extinction = false;

        auto filename = file_resolver()->resolve(
            props.get<std::string_view>("filename"));
        bool weighted_normals = props.get<bool>("weighted_normals", false);
        ScalarFloat t_divisor =
            props.get<ScalarFloat>("t_divisor", 10.f);
        ScalarFloat inv_beta = props.get<ScalarFloat>("inv_beta", 0.2f);
        std::string model = props.get<std::string>("model", "exact");
        std::string regularization =
            props.get<std::string>("regularization", "gaussian");

        m_field = SfwnField::load(filename.string(), weighted_normals,
                                  t_divisor, inv_beta, model, regularization);

        // Medium and phase must describe the same field, or the extinction and
        // the scattering lobe come from different geometry. Because fields are
        // cached, identical parameters give an identical pointer, so this is an
        // exact comparison of every field parameter at once.
        if (const auto *holder =
                dynamic_cast<const SfwnFieldHolder *>(m_phase_function.get())) {
            if (holder->sfwn_field() != m_field.get())
                Throw("SFWN medium and its phase function were configured "
                      "against different fields. Every field parameter must "
                      "match.\n  medium: %s\n  phase:  %s",
                      m_field->describe(), holder->sfwn_field()->describe());
        } else {
            Log(Warn,
                "SFWN medium has a non-SFWN phase function (%s). Extinction "
                "will use the directional SFWN sigma while scattering does "
                "not; this is only meaningful as a deliberate experiment.",
                m_phase_function->class_name());
        }

        m_scale = props.get<ScalarFloat>("scale", 1.f);
        m_albedo = props.get<ScalarFloat>("albedo", 1.f);
        m_albedo = std::clamp(m_albedo, ScalarFloat(0), ScalarFloat(1));
        m_zero_invalid_extinction =
            props.get<bool>("zero_invalid_extinction", false);

        // The spatial weight selects which extinction the medium reads, and the
        // two have different far-field baselines, so it has to be resolved
        // before the baseline is requested.
        std::string spatial_weight =
            props.get<std::string>("spatial_weight", "density");
        if (spatial_weight == "surfaceness")
            m_use_surfaceness = true;
        else if (spatial_weight != "density")
            Throw("SFWN spatial_weight must be 'density' or 'surfaceness'");

        // The far-field baseline belongs to the field, not to this medium, so
        // take it from the field by default. Measurement is lazy and cached, so
        // several media sharing a field measure it once between them.
        const SfwnBaseline &baseline =
            m_field->far_field_baseline(m_use_surfaceness);
        // Round up: this clamps the background to exactly zero, at the cost of
        // at most one unit of the faintest signal. See the README.
        ScalarFloat measured =
            baseline.samples ? ScalarFloat(std::ceil(baseline.mean))
                             : ScalarFloat(0);
        if (baseline.samples)
            Log(Info,
                "SFWN far-field baseline: mean=%g (spread %g..%g over %zu "
                "samples), ceil=%g",
                baseline.mean, baseline.min, baseline.max, baseline.samples,
                measured);

        if (props.has_property("extinction_offset")) {
            m_extinction_offset = props.get<ScalarFloat>("extinction_offset");
            if (baseline.samples &&
                std::abs(m_extinction_offset - measured) > 1.f)
                Log(Warn,
                    "SFWN extinction_offset=%g was set explicitly, but the "
                    "measured far-field baseline for this field is %g. One of "
                    "the two is stale -- the offset is a property of the field "
                    "(%s) and of spatial_weight=%s.",
                    m_extinction_offset, measured, m_field->describe(),
                    spatial_weight);
        } else if (baseline.samples) {
            m_extinction_offset = measured;
        } else {
            m_extinction_offset = 0.f;
            Log(Warn,
                "SFWN could not measure a far-field baseline for this field "
                "(no finite samples); falling back to extinction_offset=0");
        }
        if (m_extinction_offset < 0.f)
            Throw("SFWN extinction_offset must be nonnegative");

        ScalarFloat padding = props.get<ScalarFloat>("bbox_padding", 0.05f);
        const auto &bounds = m_field->bounds();
        m_bbox = ScalarBoundingBox3f(
            ScalarPoint3f(bounds.min[0] - padding, bounds.min[1] - padding,
                          bounds.min[2] - padding),
            ScalarPoint3f(bounds.max[0] + padding, bounds.max[1] + padding,
                          bounds.max[2] + padding));

        m_majorant = props.get<ScalarFloat>("majorant", 0.f);
        if (m_majorant <= 0.f) {
            size_t directions = props.get<size_t>("majorant_directions", 24);
            size_t points = props.get<size_t>("majorant_points", 2000);
            ScalarFloat safety =
                props.get<ScalarFloat>("majorant_safety", 2.f);
            m_majorant = m_scale * safety * ScalarFloat(
                m_field->estimate_majorant(directions, points,
                                            m_use_surfaceness,
                                            m_extinction_offset));
        }
        if (!(m_majorant > 0.f) || !std::isfinite(m_majorant))
            Throw("SFWN medium could not determine a valid majorant");

        std::array<double, 3> center = {
            0.5 * (bounds.min[0] + bounds.max[0]),
            0.5 * (bounds.min[1] + bounds.max[1]),
            0.5 * (bounds.min[2] + bounds.max[2])
        };
        std::array<double, 3> corner = bounds.min;
        std::array<double, 3> probe_direction = { 0.0, 0.0, 1.0 };
        auto adjusted_extinction = [&](const std::array<double, 3> &p) {
            double raw = m_field->extinction(p, probe_direction,
                                             m_use_surfaceness);
            return m_scale * std::max(0.0, raw - m_extinction_offset);
        };
        Log(Info, "SFWN scaled extinction probes: center=%g, bbox_min=%g",
            adjusted_extinction(center), adjusted_extinction(corner));

        Log(Info,
            "Loaded SFWN medium: %zu points, model=%s, regularization=%s, "
            "t_divisor=%g, t=%g, inv_beta=%g, majorant=%g",
            m_field->point_count(), m_field->volume_model(),
            m_field->regularization(), m_field->regularization_divisor(),
            m_field->regularization_parameter(),
            m_field->inv_beta(), m_majorant);
    }

    UnpolarizedSpectrum get_majorant(const MediumInteraction3f &,
                                     Mask active) const override {
        return dr::select(active, UnpolarizedSpectrum(m_majorant),
                          UnpolarizedSpectrum(0.f));
    }

    std::tuple<UnpolarizedSpectrum, UnpolarizedSpectrum,
               UnpolarizedSpectrum>
    get_scattering_coefficients(const MediumInteraction3f &mi,
                                Mask active) const override {
        if (!active)
            return { 0.f, 0.f, 0.f };

        std::array<double, 3> p = { double(mi.p[0]), double(mi.p[1]),
                                    double(mi.p[2]) };
        std::array<double, 3> w = { -double(mi.wi[0]), -double(mi.wi[1]),
                                    -double(mi.wi[2]) };
        ScalarFloat raw =
            ScalarFloat(m_field->extinction(p, w, m_use_surfaceness));
        if (!std::isfinite(raw) || raw < 0.f) {
            if (!m_zero_invalid_extinction)
                Throw("SFWN produced invalid extinction at [%g, %g, %g]",
                      p[0], p[1], p[2]);
            if (!m_invalid_reported.exchange(true, std::memory_order_relaxed))
                Log(Warn,
                    "Replacing non-finite SFWN extinction with zero for the "
                    "faulty-field diagnostic render");
            raw = 0.f;
        }
        ScalarFloat sigma =
            m_scale * std::max(ScalarFloat(0), raw - m_extinction_offset);
        if (sigma > m_majorant) {
            ScalarFloat reported =
                m_max_reported.load(std::memory_order_relaxed);
            ScalarFloat report_threshold =
                std::max(m_majorant, ScalarFloat(2) * reported);
            if (sigma > report_threshold &&
                m_max_reported.compare_exchange_strong(
                    reported, sigma, std::memory_order_relaxed))
                Log(Warn,
                    "SFWN extinction exceeded the configured majorant (%g > %g); "
                    "increase majorant_safety or set majorant explicitly",
                    sigma, m_majorant);
            sigma = m_majorant;
        }

        UnpolarizedSpectrum sigma_t(sigma);
        UnpolarizedSpectrum sigma_s = sigma_t * m_albedo;
        UnpolarizedSpectrum sigma_n =
            UnpolarizedSpectrum(m_majorant) - sigma_t;
        return { sigma_s, sigma_n, sigma_t };
    }

    std::tuple<Mask, Float, Float>
    intersect_aabb(const Ray3f &ray) const override {
        return m_bbox.ray_intersect(ray);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SfwnMedium[" << std::endl
            << "  filename = " << m_field->filename() << "," << std::endl
            << "  points = " << m_field->point_count() << "," << std::endl
            << "  model = " << m_field->volume_model() << "," << std::endl
            << "  regularization = " << m_field->regularization() << ","
            << std::endl
            << "  t = " << m_field->regularization_parameter() << ","
            << std::endl
            << "  t_divisor = " << m_field->regularization_divisor() << ","
            << std::endl
            << "  inv_beta = " << m_field->inv_beta() << "," << std::endl
            << "  spatial_weight = "
            << (m_use_surfaceness ? "surfaceness" : "density") << ","
            << std::endl
            << "  extinction_offset = " << m_extinction_offset << ","
            << std::endl
            << "  zero_invalid_extinction = " << m_zero_invalid_extinction
            << "," << std::endl
            << "  majorant = " << m_majorant << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(SfwnMedium)

private:
    SfwnField::Ptr m_field;
    ScalarBoundingBox3f m_bbox;
    ScalarFloat m_scale;
    ScalarFloat m_extinction_offset;
    ScalarFloat m_albedo;
    ScalarFloat m_majorant;
    bool m_use_surfaceness = false;
    bool m_zero_invalid_extinction = false;
    mutable std::atomic<ScalarFloat> m_max_reported{ 0.f };
    mutable std::atomic<bool> m_invalid_reported{ false };
};

MI_EXPORT_PLUGIN(SfwnMedium)
NAMESPACE_END(mitsuba)
