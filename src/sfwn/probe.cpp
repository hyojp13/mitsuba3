#include <mitsuba/render/sfwn.h>

#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using mitsuba::SfwnField;

int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: sfwn_probe POINTS.ply t_divisor [inv_beta] "
                     "[regularization] [model]\n";
        return 2;
    }

    double t_divisor = std::stod(argv[2]);
    double inv_beta = argc >= 4 ? std::stod(argv[3]) : 0.2;
    std::string regularization = argc >= 5 ? argv[4] : "gaussian";
    std::string model = argc >= 6 ? argv[5] : "exact";
    auto field = SfwnField::load(argv[1], false, t_divisor, inv_beta,
                                 model, regularization);
    const auto &bounds = field->bounds();
    std::array<double, 3> p = {
        0.5 * (bounds.min[0] + bounds.max[0]),
        0.5 * (bounds.min[1] + bounds.max[1]),
        0.0
    };
    const std::array<double, 3> w = { 0.0, 0.0, 1.0 };

    std::cout << std::setprecision(12)
              << "# points=" << field->point_count()
              << " t_divisor=" << field->regularization_divisor()
              << " t=" << field->regularization_parameter()
              << " inv_beta=" << field->inv_beta()
              << " model=" << field->volume_model()
              << " regularization=" << field->regularization() << '\n'
              << "z,mean,variance,occupancy,surfaceness,density,sigma,"
                 "extinction_surfaceness,extinction_density\n";
    double span = bounds.max[2] - bounds.min[2];
    for (int i = 0; i <= 100; ++i) {
        p[2] = bounds.min[2] - 0.25 * span + 1.5 * span * i / 100.0;
        auto d = field->diagnostics(p, w);
        std::cout << p[2] << ',' << d.mean << ',' << d.variance << ','
                  << d.occupancy << ',' << d.surfaceness << ',' << d.density
                  << ',' << d.sigma << ',' << d.surfaceness * d.sigma << ','
                  << d.density * d.sigma << '\n';
    }
    return 0;
}
