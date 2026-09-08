#include <mitsuba/core/properties.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/util.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/sfwn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>

NAMESPACE_BEGIN(mitsuba)

/**
 * Implicit GPIS surface shape.
 *
 * Rays are sampled through the cached SFWN field and the first outside-to-
 * inside level crossing is refined by bisection. No mesh or voxel grid is
 * constructed. This plugin intentionally supports scalar CPU variants only,
 * matching SfwnField and the companion medium/phase plugins.
 */
template <typename Float, typename Spectrum>
class SfwnSurface final : public Shape<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Shape, m_to_world, m_is_instance, m_shape_type,
                   initialize, get_children_string)
    MI_IMPORT_TYPES()

    using typename Base::ScalarSize;
    using typename Base::ScalarIndex;

    SfwnSurface(const Properties &props) : Base(props) {
        static_assert(!dr::is_jit_v<Float>,
                      "The SFWN surface currently supports scalar variants");

        auto filename = file_resolver()->resolve(
            props.get<std::string_view>("filename"));
        bool weighted_normals = props.get<bool>("weighted_normals", false);
        ScalarFloat t_divisor =
            props.get<ScalarFloat>("t_divisor", 10.f);
        ScalarFloat inv_beta = props.get<ScalarFloat>("inv_beta", 0.2f);
        std::string model = props.get<std::string>("model", "exact");
        std::string regularization =
            props.get<std::string>("regularization", "gaussian");
        m_use_mean = props.get<std::string>("field", "occupancy") == "mean";
        if (!m_use_mean && props.get<std::string>("field", "occupancy") != "occupancy")
            Throw("SFWN surface field must be 'mean' or 'occupancy'");
        m_surface_level = props.get<ScalarFloat>(
            "surface_level", m_use_mean ? ScalarFloat(0) : ScalarFloat(0.5));
        m_ray_steps = props.get<uint32_t>("ray_steps", 128u);
        m_refine_steps = props.get<uint32_t>("refine_steps", 6u);
        if (m_ray_steps < 2)
            Throw("SFWN surface ray_steps must be at least 2");

        m_field = SfwnField::load(filename.string(), weighted_normals,
                                  t_divisor, inv_beta, model, regularization);
        const auto &bounds = m_field->bounds();
        ScalarFloat padding = props.get<ScalarFloat>("bbox_padding", 0.05f);
        m_bbox = ScalarBoundingBox3f(
            ScalarPoint3f(bounds.min[0] - padding, bounds.min[1] - padding,
                          bounds.min[2] - padding),
            ScalarPoint3f(bounds.max[0] + padding, bounds.max[1] + padding,
                          bounds.max[2] + padding));

        // m_bbox above is in the field's object space. The accelerator needs a
        // world-space box, so transform the eight corners; the Shape base has
        // already read to_world from the properties.
        const ScalarAffineTransform4f &to_world = m_to_world.scalar();
        m_world_bbox = ScalarBoundingBox3f();
        for (int i = 0; i < 8; ++i)
            m_world_bbox.expand(to_world * m_bbox.corner(i));

        m_shape_type = ShapeType::SDFGrid;
        initialize();
    }

    ScalarSize primitive_count() const override { return 1; }

    ScalarBoundingBox3f bbox() const override { return m_world_bbox; }

    Float surface_area() const override { return 0.f; }

    PositionSample3f sample_position(Float, const Point2f &, Mask active) const override {
        MI_MASK_ARGUMENT(active);
        return dr::zeros<PositionSample3f>();
    }

    Float pdf_position(const PositionSample3f &, Mask active) const override {
        MI_MASK_ARGUMENT(active);
        return 0.f;
    }

    SurfaceInteraction3f eval_parameterization(const Point2f &, uint32_t,
                                               Mask active) const override {
        MI_MASK_ARGUMENT(active);
        return dr::zeros<SurfaceInteraction3f>();
    }

    template <typename FloatP, typename Ray3fP>
    std::tuple<dr::mask_t<FloatP>, FloatP, Point<FloatP, 2>,
               dr::uint32_array_t<FloatP>, dr::uint32_array_t<FloatP>>
    ray_intersect_preliminary_impl(const Ray3fP &ray_, ScalarIndex,
                                   dr::mask_t<FloatP> active) const {
        MI_MASK_ARGUMENT(active);
        // The SFWN query is scalar. Mitsuba may still instantiate packet
        // entry points for its shape interface; leave those unsupported and
        // use the scalar entry point for actual intersection work.
        if constexpr (dr::is_array_v<FloatP>) {
            return { dr::zeros<dr::mask_t<FloatP>>(),
                     dr::Infinity<FloatP>, Point<FloatP, 2>(0.f),
                     dr::zeros<dr::uint32_array_t<FloatP>>(),
                     dr::zeros<dr::uint32_array_t<FloatP>>() };
        } else {

        // Transform into object space without renormalising the direction, so
        // the parametric distances found below remain valid for the world ray.
        auto to_object = m_to_world.scalar().inverse();
        auto o_local = to_object * ray_.o;
        auto d_local = to_object * ray_.d;
        std::array<double, 3> origin = {
            double(o_local[0]), double(o_local[1]), double(o_local[2]) };
        std::array<double, 3> direction = {
            double(d_local[0]), double(d_local[1]), double(d_local[2]) };
        double t_near = 0.0;
        double t_far = double(ray_.maxt);
        if (!intersect_bbox(origin, direction, t_near, t_far))
            return { false, dr::Infinity<FloatP>, Point<FloatP, 2>(0.f), 0, 0 };

        double previous_t = t_near;
        SfwnSurfaceSample previous = sample(origin, direction, previous_t);
        bool previous_inside = valid(previous) &&
                               previous.value > double(m_surface_level);
        for (uint32_t i = 1; i <= m_ray_steps; ++i) {
            double current_t = t_near + (t_far - t_near) *
                               double(i) / double(m_ray_steps);
            SfwnSurfaceSample current = sample(origin, direction, current_t);
            bool current_inside = valid(current) &&
                                  current.value > double(m_surface_level);
            if (!previous_inside && current_inside) {
                double lo = previous_t, hi = current_t;
                for (uint32_t j = 0; j < m_refine_steps; ++j) {
                    double mid = 0.5 * (lo + hi);
                    SfwnSurfaceSample middle = sample(origin, direction, mid);
                    if (valid(middle) &&
                        middle.value > double(m_surface_level))
                        hi = mid;
                    else
                        lo = mid;
                }
                return { active, FloatP(0.5 * (lo + hi)),
                         Point<FloatP, 2>(0.f), 0, 0 };
            }
            previous_t = current_t;
            previous_inside = current_inside;
        }

        return { false, dr::Infinity<FloatP>, Point<FloatP, 2>(0.f), 0, 0 };
        }
    }

    template <typename FloatP, typename Ray3fP>
    dr::mask_t<FloatP> ray_test_impl(const Ray3fP &ray_, ScalarIndex prim_index,
                                     dr::mask_t<FloatP> active) const {
        auto [hit, t, uv, shape_index, primitive] =
            ray_intersect_preliminary_impl<FloatP>(ray_, prim_index, active);
        return hit;
    }

    MI_SHAPE_DEFINE_RAY_INTERSECT_METHODS()

    SurfaceInteraction3f compute_surface_interaction(
        const Ray3f &ray, const PreliminaryIntersection3f &pi,
        uint32_t ray_flags, uint32_t recursion_depth,
        Mask active) const override {
        MI_MASK_ARGUMENT(active);
        if (!m_is_instance && recursion_depth > 0)
            return dr::zeros<SurfaceInteraction3f>();

        Point3f p = ray(pi.t);
        const AffineTransform4f &to_world = m_to_world.value();
        Point3f p_local = to_world.inverse() * p;
        std::array<double, 3> position = {
            double(p_local[0]), double(p_local[1]), double(p_local[2]) };
        SfwnSurfaceSample value = sample(position);
        // The gradient is an object-space normal, so it maps to world space
        // through the inverse transpose, which Transform applies to Normal3f.
        Normal3f gradient(ScalarFloat(value.gradient[0]),
                          ScalarFloat(value.gradient[1]),
                          ScalarFloat(value.gradient[2]));
        Normal3f normal = dr::normalize(to_world * Normal3f(-gradient));

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.t = pi.t;
        si.p = dr::detach(p);
        si.n = dr::detach(normal);
        si.uv = Point2f(0.f);
        si.sh_frame = Frame3f(si.n);
        si.attach_motion(ray, si.p, ray_flags);
        si.prim_index = pi.prim_index;
        si.shape = this;
        si.instance = nullptr;
        return si;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SfwnSurface[" << std::endl
            << "  filename = " << m_field->filename() << "," << std::endl
            << "  field = " << (m_use_mean ? "mean" : "occupancy") << ","
            << std::endl
            << "  surface_level = " << m_surface_level << "," << std::endl
            << "  ray_steps = " << m_ray_steps << "," << std::endl
            << "  refine_steps = " << m_refine_steps << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(SfwnSurface)

private:
    static bool valid(const SfwnSurfaceSample &sample) {
        return std::isfinite(sample.value) &&
               std::isfinite(sample.gradient[0]) &&
               std::isfinite(sample.gradient[1]) &&
               std::isfinite(sample.gradient[2]);
    }

    SfwnSurfaceSample sample(const std::array<double, 3> &position) const {
        return m_field->surface_sample(position, m_use_mean);
    }

    SfwnSurfaceSample sample(const std::array<double, 3> &origin,
                             const std::array<double, 3> &direction,
                             double t) const {
        return sample({ origin[0] + t * direction[0],
                        origin[1] + t * direction[1],
                        origin[2] + t * direction[2] });
    }

    bool intersect_bbox(const std::array<double, 3> &origin,
                        const std::array<double, 3> &direction,
                        double &t_near, double &t_far) const {
        for (int axis = 0; axis < 3; ++axis) {
            double o = origin[axis], d = direction[axis];
            double lo = double(m_bbox.min[axis]), hi = double(m_bbox.max[axis]);
            if (std::abs(d) < 1e-12) {
                if (o < lo || o > hi)
                    return false;
                continue;
            }
            double a = (lo - o) / d, b = (hi - o) / d;
            if (a > b)
                std::swap(a, b);
            t_near = std::max(t_near, a);
            t_far = std::min(t_far, b);
            if (t_near > t_far)
                return false;
        }
        return t_far >= std::max(t_near, 0.0);
    }

    SfwnField::Ptr m_field;
    ScalarBoundingBox3f m_bbox;        //< object space
    ScalarBoundingBox3f m_world_bbox;  //< m_bbox under to_world
    ScalarFloat m_surface_level = 0.5f;
    uint32_t m_ray_steps = 128;
    uint32_t m_refine_steps = 6;
    bool m_use_mean = false;
};

MI_EXPORT_PLUGIN(SfwnSurface)
NAMESPACE_END(mitsuba)
