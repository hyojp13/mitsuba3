#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/sfwn.h>

#include <array>
#include <cmath>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class SfwnPhaseFunction final : public PhaseFunction<Float, Spectrum> {
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
        m_field = SfwnField::load(filename.string(), weighted_normals,
                                  t_divisor, inv_beta, model, regularization);

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

        std::array<double, 3> p = { double(mi.p[0]), double(mi.p[1]),
                                    double(mi.p[2]) };
        std::array<double, 3> w = { -double(mi.wi[0]), -double(mi.wi[1]),
                                    -double(mi.wi[2]) };
        std::array<double, 3> u = { double(sample1), double(sample2[0]),
                                    double(sample2[1]) };
        SfwnVndfSample sample = m_field->sample_vndf(p, w, u);
        Normal3f m(ScalarFloat(sample.normal[0]),
                   ScalarFloat(sample.normal[1]),
                   ScalarFloat(sample.normal[2]));
        Float jacobian = 4.f * dr::abs(dr::dot(mi.wi, m));
        Float pdf = Float(sample.pdf) / jacobian;
        Vector3f wo = dr::normalize(reflect(mi.wi, m));

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
        Vector3f sum = wo + mi.wi;
        Float sum_norm = dr::norm(sum);
        if (!(sum_norm > 0.f))
            return { 0.f, 0.f };
        Vector3f m = sum / sum_norm;
        Vector3f propagation = -mi.wi;
        if (dr::dot(m, propagation) < 0.f)
            m = -m;

        std::array<double, 3> p = { double(mi.p[0]), double(mi.p[1]),
                                    double(mi.p[2]) };
        std::array<double, 3> w = { double(propagation[0]),
                                    double(propagation[1]),
                                    double(propagation[2]) };
        std::array<double, 3> normal = { double(m[0]), double(m[1]),
                                         double(m[2]) };
        Float jacobian = 4.f * dr::abs(dr::dot(wo, m));
        Float pdf = Float(m_field->vndf(p, w, normal)) / jacobian;
        if (!std::isfinite(double(pdf)) || pdf < 0.f)
            pdf = 0.f;
        return { Spectrum(pdf), pdf };
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
    SfwnField::Ptr m_field;
};

MI_EXPORT_PLUGIN(SfwnPhaseFunction)
NAMESPACE_END(mitsuba)
