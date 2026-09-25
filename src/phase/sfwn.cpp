#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/sfwn.h>

#include <array>
#include <cmath>
#include <stdexcept>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class SfwnPhaseFunction final : public PhaseFunction<Float, Spectrum>,
                                public SfwnFieldHolder {
public:
    MI_IMPORT_BASE(PhaseFunction, m_flags)
    MI_IMPORT_TYPES(PhaseFunctionContext)

    SfwnPhaseFunction(const Properties &props) : Base(props) {
        static_assert(!dr::is_jit_v<Float>,
                      "The SFWN phase function currently supports scalar variants");
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

        // The companion medium already evaluates the full directional sigma.
        // Do not set Microflake, which would make Mitsuba multiply it again.
        m_flags = uint32_t(PhaseFunctionFlags::Anisotropic);
    }

    std::tuple<Vector3f, Spectrum, Float>
    sample(const PhaseFunctionContext &, const MediumInteraction3f &mi,
           const Float sample1, const Point2f &sample2,
           Mask active) const override {
        if (!active)
            return { Vector3f(0.f), Spectrum(0.f), 0.f };

        // The field lives in object space; work there throughout and rotate
        // the sampled direction back at the end. A rotation preserves solid
        // angle, so the pdf needs no Jacobian for the change of frame.
        Vector3f wi = rotate(mi.wi, false);
        std::array<double, 3> p = to_object(mi.p);
        std::array<double, 3> w = { -double(wi[0]), -double(wi[1]),
                                    -double(wi[2]) };
        std::array<double, 3> u = { double(sample1), double(sample2[0]),
                                    double(sample2[1]) };
        SfwnVndfSample sample = m_field->sample_vndf(p, w, u);
        Normal3f m(ScalarFloat(sample.normal[0]),
                   ScalarFloat(sample.normal[1]),
                   ScalarFloat(sample.normal[2]));
        Float jacobian = 4.f * dr::abs(dr::dot(wi, m));
        Float pdf = Float(sample.pdf) / jacobian;
        Vector3f wo = rotate(dr::normalize(reflect(wi, m)), true);

        bool valid = std::isfinite(double(pdf)) && pdf > 0.f &&
                     std::isfinite(double(wo[0])) &&
                     std::isfinite(double(wo[1])) &&
                     std::isfinite(double(wo[2]));
        if (!valid)
            return { Vector3f(0.f), Spectrum(0.f), 0.f };
        return { wo, Spectrum(1.f), pdf };
    }

    std::pair<Spectrum, Float>
    eval_pdf(const PhaseFunctionContext &, const MediumInteraction3f &mi,
             const Vector3f &wo, Mask active) const override {
        if (!active)
            return { 0.f, 0.f };
        Vector3f wi_o = rotate(mi.wi, false);
        Vector3f wo_o = rotate(wo, false);
        Vector3f sum = wo_o + wi_o;
        Float sum_norm = dr::norm(sum);
        if (!(sum_norm > 0.f))
            return { 0.f, 0.f };
        Vector3f m = sum / sum_norm;
        Vector3f propagation = -wi_o;
        if (dr::dot(m, propagation) < 0.f)
            m = -m;

        std::array<double, 3> p = to_object(mi.p);
        std::array<double, 3> w = { double(propagation[0]),
                                    double(propagation[1]),
                                    double(propagation[2]) };
        std::array<double, 3> normal = { double(m[0]), double(m[1]),
                                         double(m[2]) };
        Float jacobian = 4.f * dr::abs(dr::dot(wo_o, m));
        Float pdf = Float(m_field->vndf(p, w, normal)) / jacobian;
        if (!std::isfinite(double(pdf)) || pdf < 0.f)
            pdf = 0.f;
        return { Spectrum(pdf), pdf };
    }

    /// SfwnFieldHolder: lets the enclosing sfwnmedium confirm that medium and
    /// phase were configured against the same field. Fields are cached, so an
    /// identical configuration yields an identical pointer.
    const SfwnField *sfwn_field() const override { return m_field.get(); }

    void set_sfwn_world_to_object(const std::array<double, 16> &m) override {
        if (m_has_transform && m != m_world_to_object)
            Throw("A single sfwnphase is shared by two sfwnmedium instances "
                  "with different to_world transforms. Give each medium its "
                  "own phase function.");
        m_world_to_object = m;
        m_has_transform = true;
        // Directions only need the linear part. For the similarity transforms
        // sfwnmedium admits this is a uniformly scaled rotation, so normalising
        // after applying it (or its transpose, for the inverse) recovers the
        // rotation exactly and the uniform scale drops out.
        m_identity = (m == identity_matrix());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SfwnPhaseFunction[" << std::endl
            << "  filename = " << m_field->filename() << "," << std::endl
            << "  points = " << m_field->point_count() << "," << std::endl
            << "  model = " << m_field->volume_model() << "," << std::endl
            << "  regularization = " << m_field->regularization() << ","
            << std::endl
            << "  t = " << m_field->regularization_parameter() << std::endl
            << "  t_divisor = " << m_field->regularization_divisor()
            << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(SfwnPhaseFunction)

private:
    static constexpr std::array<double, 16> identity_matrix() {
        return { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    }

    /// World-space point to the field's object space.
    std::array<double, 3> to_object(const Point3f &p) const {
        if (m_identity)
            return { double(p[0]), double(p[1]), double(p[2]) };
        const auto &m = m_world_to_object;
        double x = double(p[0]), y = double(p[1]), z = double(p[2]);
        return { m[0] * x + m[1] * y + m[2] * z + m[3],
                 m[4] * x + m[5] * y + m[6] * z + m[7],
                 m[8] * x + m[9] * y + m[10] * z + m[11] };
    }

    /// World-space direction to object space. `transpose` inverts it: for a
    /// uniformly scaled rotation the transpose is the inverse up to the scale,
    /// which normalisation removes.
    Vector3f rotate(const Vector3f &v, bool transpose) const {
        if (m_identity)
            return v;
        const auto &m = m_world_to_object;
        double x = double(v[0]), y = double(v[1]), z = double(v[2]);
        double a, b, c;
        if (!transpose) {
            a = m[0] * x + m[1] * y + m[2] * z;
            b = m[4] * x + m[5] * y + m[6] * z;
            c = m[8] * x + m[9] * y + m[10] * z;
        } else {
            a = m[0] * x + m[4] * y + m[8] * z;
            b = m[1] * x + m[5] * y + m[9] * z;
            c = m[2] * x + m[6] * y + m[10] * z;
        }
        return dr::normalize(Vector3f(ScalarFloat(a), ScalarFloat(b),
                                      ScalarFloat(c)));
    }

    SfwnField::Ptr m_field;
    std::array<double, 16> m_world_to_object = identity_matrix();
    bool m_has_transform = false;
    bool m_identity = true;
};

MI_EXPORT_PLUGIN(SfwnPhaseFunction)
NAMESPACE_END(mitsuba)
