#pragma once

#include "euler.hpp"

#include <hdf5.h>

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

// HDF5 snapshots in LUNA's layout (src/luna/file_io/write_snapshot_distributed.py,
// checkpoint_utils.py:_write_units): <dir>/snapdir_NNN/snapshot_NNN.0.hdf5,
// group `cells` with LUNA's attributes and datasets (density, velocity_x/y/z,
// internal_energy per unit volume, pressure, U as [5][N]), plus for the
// Voronoi mesh mesh = "voronoi", N_cells, and the datasets x, y, z
// (generating points) and volume. Group `units` as LUNA writes it.
namespace hydro {

struct Units {
    double length_cm = 1, mass_g = 1, velocity_cm_s = 1;
    bool enabled() const { return !(length_cm == 1 && mass_g == 1 && velocity_cm_s == 1); }
};

// What a snapshot holds, on the host: per cell the generator x[3], the
// volume and the conserved state U[5].
struct SnapshotData {
    int index = 0;
    double time = 0;
    double box = 1;
    int order = 1;
    Gas gas;
    Units units;
    std::vector<double> x, y, z, volume;
    std::vector<double> U[5];
};

namespace detail {
inline void attribute(hid_t group, const char* name, hid_t type, const void* value) {
    const hid_t space = H5Screate(H5S_SCALAR);
    const hid_t attr = H5Acreate2(group, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, type, value);
    H5Aclose(attr);
    H5Sclose(space);
}
inline void attribute(hid_t group, const char* name, double value) { attribute(group, name, H5T_NATIVE_DOUBLE, &value); }
inline void attribute(hid_t group, const char* name, long long value) { attribute(group, name, H5T_NATIVE_LLONG, &value); }
inline void attribute(hid_t group, const char* name, const std::string& value) {
    const hid_t type = H5Tcopy(H5T_C_S1);
    H5Tset_size(type, value.size());
    attribute(group, name, type, value.c_str());
    H5Tclose(type);
}
inline void dataset(hid_t group, const char* name, const std::vector<double>& values, hsize_t rows = 1) {
    const hsize_t dims[2] = {rows, values.size() / rows};
    const hid_t space = H5Screate_simple(rows == 1 ? 1 : 2, rows == 1 ? dims + 1 : dims, nullptr);
    const hid_t set = H5Dcreate2(group, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(set, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
    H5Dclose(set);
    H5Sclose(space);
}
}  // namespace detail

// Writes the snapshot; returns the file name.
inline std::string write_snapshot(const std::string& dir, const SnapshotData& s) {
    char number[16];
    std::snprintf(number, sizeof number, "%03d", s.index);
    const std::string folder = dir + "/snapdir_" + number;
    std::filesystem::create_directories(folder);
    const std::string name = folder + "/snapshot_" + number + ".0.hdf5";
    const hid_t file = H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("hydro::write_snapshot: cannot create " + name);

    const std::size_t n = s.x.size();
    std::vector<double> rho(n), vx(n), vy(n), vz(n), e_int(n), P(n), U(5 * n);
    for (std::size_t i = 0; i < n; ++i) {
        const State W = primitive({s.U[0][i], s.U[1][i], s.U[2][i], s.U[3][i], s.U[4][i]}, s.gas);
        rho[i] = W[0];
        vx[i] = W[1];
        vy[i] = W[2];
        vz[i] = W[3];
        P[i] = W[4];
        e_int[i] = W[4] / (s.gas.gamma - 1.0);
        for (int v = 0; v < 5; ++v) U[v * n + i] = s.U[v][i];
    }

    const hid_t cells = H5Gcreate2(file, "cells", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    detail::attribute(cells, "time", s.time);
    detail::attribute(cells, "step", static_cast<long long>(s.index));
    detail::attribute(cells, "gamma", s.gas.gamma);
    detail::attribute(cells, "spatial_dimension", 3LL);
    detail::attribute(cells, "order", static_cast<long long>(s.order));
    detail::attribute(cells, "Lx", s.box);
    detail::attribute(cells, "Ly", s.box);
    detail::attribute(cells, "Lz", s.box);
    detail::attribute(cells, "mesh", std::string("voronoi"));
    detail::attribute(cells, "N_cells", static_cast<long long>(n));
    detail::dataset(cells, "x", s.x);
    detail::dataset(cells, "y", s.y);
    detail::dataset(cells, "z", s.z);
    detail::dataset(cells, "volume", s.volume);
    detail::dataset(cells, "density", rho);
    detail::dataset(cells, "velocity_x", vx);
    detail::dataset(cells, "velocity_y", vy);
    detail::dataset(cells, "velocity_z", vz);
    detail::dataset(cells, "internal_energy", e_int);
    detail::dataset(cells, "pressure", P);
    detail::dataset(cells, "U", U, 5);
    H5Gclose(cells);

    if (s.units.enabled()) {
        const hid_t units = H5Gcreate2(file, "units", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        const double time_s = s.units.length_cm / s.units.velocity_cm_s;
        const double density = s.units.mass_g / (s.units.length_cm * s.units.length_cm * s.units.length_cm);
        detail::attribute(units, "UnitLength_in_cm", s.units.length_cm);
        detail::attribute(units, "UnitMass_in_g", s.units.mass_g);
        detail::attribute(units, "UnitVelocity_in_cm_per_s", s.units.velocity_cm_s);
        detail::attribute(units, "UnitTime_in_s", time_s);
        detail::attribute(units, "UnitDensity_in_cgs", density);
        detail::attribute(units, "UnitPressure_in_cgs", density * s.units.velocity_cm_s * s.units.velocity_cm_s);
        detail::attribute(units, "UnitEnergy_in_cgs", s.units.mass_g * s.units.velocity_cm_s * s.units.velocity_cm_s);
        H5Gclose(units);
    }
    H5Fclose(file);
    return name;
}

}  // namespace hydro
