#include <mitsuba/core/bbox.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/sfwn.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>

NAMESPACE_BEGIN(mitsuba)

/**
 * Majorant-exceedance totals, reported once at process exit.
 *
 * A medium cannot report this itself: Mitsuba never destroys the scene before
 * exiting, so ~SfwnMedium does not run (verified -- not even a bare fprintf in
 * it produces output). An atexit handler is the one hook that reliably fires,
 * and it writes to stderr directly because the logger may already be gone.
 *
 * Totals are aggregated across media rather than kept per-instance; a scene
 * with several SFWN media is rare, and the peak ratio is the actionable
 * number either way.
 */
namespace {
std::atomic<std::size_t> g_exceed_count{ 0 };
std::atomic<double> g_exceed_peak{ 0.0 };
std::atomic<double> g_exceed_peak_majorant{ 0.0 };
std::once_flag g_exceed_atexit;

void sfwn_report_exceedances() {
    std::size_t count = g_exceed_count.load(std::memory_order_relaxed);
    if (count == 0)
        return;
    double peak = g_exceed_peak.load(std::memory_order_relaxed);
    double majorant = g_exceed_peak_majorant.load(std::memory_order_relaxed);
    std::fprintf(stderr,
                 "WARN  [SfwnMedium] majorant exceeded %zu times; sigma_t was "
                 "clamped, peaking at %g against a majorant of %g (%.2fx). "
                 "Delta tracking is biased thin wherever that happened -- "
                 "raise majorant_safety, widen majorant_shell_radius, or set "
                 "majorant explicitly.\n",
                 count, peak, majorant,
                 majorant > 0.0 ? peak / majorant : 0.0);
}

/**
 * Zeroed-sample totals, reported once at process exit alongside the majorant
 * exceedances and for the same reason: a long render that survives an invalid
 * pocket has still been biased thin there, and a one-shot warning says nothing
 * about how much. Counting makes the trade-off auditable after the fact.
 */
std::atomic<std::size_t> g_invalid_count{ 0 };
std::once_flag g_invalid_atexit;

void sfwn_report_invalid() {
    std::size_t count = g_invalid_count.load(std::memory_order_relaxed);
    if (count == 0)
        return;
    std::fprintf(stderr,
                 "WARN  [SfwnMedium] zeroed %zu invalid extinction samples "
                 "(zero_invalid_extinction=true); the medium is biased thin "
                 "wherever the field was non-finite or negative. A large count "
                 "means the field itself is unusable -- raise t_divisor.\n",
                 count);
}

void sfwn_note_invalid() {
    g_invalid_count.fetch_add(1, std::memory_order_relaxed);
    std::call_once(g_invalid_atexit,
                   [] { std::atexit(sfwn_report_invalid); });
}

void sfwn_note_exceedance(double sigma, double majorant) {
    g_exceed_count.fetch_add(1, std::memory_order_relaxed);
    double prev = g_exceed_peak.load(std::memory_order_relaxed);
    while (sigma > prev &&
           !g_exceed_peak.compare_exchange_weak(prev, sigma,
                                                std::memory_order_relaxed))
        ;
    if (sigma >= g_exceed_peak.load(std::memory_order_relaxed))
        g_exceed_peak_majorant.store(majorant, std::memory_order_relaxed);
    std::call_once(g_exceed_atexit,
                   [] { std::atexit(sfwn_report_exceedances); });
}
}  // namespace

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
        // "sensitivity" (the current model: variance is the data Gram, so it is
        // non-negative at any t and zero far from the cloud) or "prior" (the GP
        // posterior, prior minus data Gram, whose far field is a nonzero
        // constant and whose variance can go negative).
        std::string process =
            props.get<std::string>("process", "sensitivity");

        m_field = SfwnField::load(filename.string(), weighted_normals,
                                  t_divisor, inv_beta, model, regularization, process);

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

        // Medium has no to_world of its own, so read it here. The field is in
        // object space from now on; rays are transformed into it.
        m_to_world = props.get<ScalarAffineTransform4f>(
            "to_world", ScalarAffineTransform4f());
        if (!m_to_world.is_similarity())
            Throw("SFWN medium to_world must be a similarity transform "
                  "(rotation, uniform scale, translation). sigma_t has units "
                  "of inverse length, so a non-uniform scale would need a "
                  "direction-dependent correction the field does not model.");
        m_to_object = m_to_world.inverse();
        // Uniform scale factor: object lengths times m_world_scale are world
        // lengths, so world-space sigma_t is the field's value divided by it.
        m_world_scale =
            dr::norm(m_to_world * ScalarVector3f(1.f, 0.f, 0.f));
        if (!(m_world_scale > 0.f) || !std::isfinite(m_world_scale))
            Throw("SFWN medium to_world has a degenerate scale");

        if (auto *holder =
                dynamic_cast<SfwnFieldHolder *>(m_phase_function.get())) {
            std::array<double, 16> m;
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    m[r * 4 + c] = double(m_to_object.matrix.entry(r, c));
            holder->set_sfwn_world_to_object(m);
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
            // The estimator probes a shell around each point, not just the
            // point itself, because sigma_t peaks just inside the surface.
            // That removes a systematic ~1.9x underestimate, so the safety
            // factor no longer has to silently cover one and can be modest.
            size_t shell_samples =
                props.get<size_t>("majorant_shell_samples", 5);
            ScalarFloat shell_radius =
                props.get<ScalarFloat>("majorant_shell_radius", 0.02f);
            ScalarFloat safety =
                props.get<ScalarFloat>("majorant_safety", 2.f);
            ScalarFloat raw = ScalarFloat(
                m_field->estimate_majorant(directions, points,
                                            m_use_surfaceness,
                                            m_extinction_offset,
                                            shell_samples,
                                            double(shell_radius)));
            m_majorant = m_scale * safety * raw;
            Log(Info,
                "SFWN majorant: shell-sampled max=%g (%zu points x %zu "
                "directions x %zu shell offsets, radius=%g of bbox diagonal), "
                "safety=%g => majorant=%g",
                raw, points, directions, 2 * shell_samples + 1, shell_radius,
                safety, m_majorant);
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
        // m_majorant bounds the field in object units; tracking happens along
        // world-space rays.
        return dr::select(active,
                          UnpolarizedSpectrum(m_majorant / m_world_scale),
                          UnpolarizedSpectrum(0.f));
    }

    std::tuple<UnpolarizedSpectrum, UnpolarizedSpectrum,
               UnpolarizedSpectrum>
    get_scattering_coefficients(const MediumInteraction3f &mi,
                                Mask active) const override {
        if (!active)
            return { 0.f, 0.f, 0.f };

        ScalarPoint3f po = m_to_object * ScalarPoint3f(mi.p);
        ScalarVector3f wo_dir =
            dr::normalize(m_to_object * ScalarVector3f(-mi.wi));
        std::array<double, 3> p = { double(po[0]), double(po[1]),
                                    double(po[2]) };
        std::array<double, 3> w = { double(wo_dir[0]), double(wo_dir[1]),
                                    double(wo_dir[2]) };
        ScalarFloat raw =
            ScalarFloat(m_field->extinction(p, w, m_use_surfaceness));
        if (!std::isfinite(raw) || raw < 0.f) {
            if (!m_zero_invalid_extinction)
                Throw("SFWN produced invalid extinction at [%g, %g, %g]",
                      p[0], p[1], p[2]);
            sfwn_note_invalid();
            if (!m_invalid_reported.exchange(true, std::memory_order_relaxed))
                Log(Warn,
                    "Replacing invalid SFWN extinction with zero at "
                    "[%g, %g, %g]; the running total is reported at exit",
                    p[0], p[1], p[2]);
            raw = 0.f;
        }
        ScalarFloat sigma =
            m_scale * std::max(ScalarFloat(0), raw - m_extinction_offset);
        if (sigma > m_majorant) {
            // Count every exceedance, not just the ones worth warning about:
            // clamping biases the medium thin wherever it happens, and a total
            // is the only way to know whether that mattered. This branch is
            // rare for a valid majorant, so the atomics stay off the hot path.
            sfwn_note_exceedance(double(sigma), double(m_majorant));
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

        // Convert from the field's object-space units to world units. Both
        // sigma_t and sigma_n are divided, so sigma_t + sigma_n still equals
        // the majorant get_majorant() reports.
        UnpolarizedSpectrum sigma_t(sigma / m_world_scale);
        UnpolarizedSpectrum sigma_s = sigma_t * m_albedo;
        UnpolarizedSpectrum sigma_n =
            UnpolarizedSpectrum(m_majorant / m_world_scale) - sigma_t;
        return { sigma_s, sigma_n, sigma_t };
    }

    std::tuple<Mask, Float, Float>
    intersect_aabb(const Ray3f &ray) const override {
        // m_bbox is in object space. Transforming the ray without renormalising
        // its direction keeps the returned parametric distances valid for the
        // original world-space ray.
        Ray3f local(ray);
        local.o = m_to_object * ray.o;
        local.d = m_to_object * ray.d;
        return m_bbox.ray_intersect(local);
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
    ScalarBoundingBox3f m_bbox;   //< object space
    ScalarAffineTransform4f m_to_world;
    ScalarAffineTransform4f m_to_object;
    ScalarFloat m_world_scale = 1.f;
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
