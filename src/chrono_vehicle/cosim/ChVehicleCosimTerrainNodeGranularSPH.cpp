// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2020 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Wei Hu, Radu Serban
// =============================================================================
//
// Definition of the SPH granular TERRAIN NODE (using Chrono::FSI).
//
// The global reference frame has Z up, X towards the front of the vehicle, and
// Y pointing to the left.
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <set>

#include <mpi.h>

#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono/assets/ChBoxShape.h"
#include "chrono/assets/ChTriangleMeshShape.h"

#include "chrono_fsi/utils/ChUtilsJSON.h"
#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"

#include "chrono_vehicle/cosim/ChVehicleCosimTerrainNodeGranularSPH.h"

#ifdef CHRONO_OPENGL
#include "chrono_opengl/ChOpenGLWindow.h"
#endif

using namespace chrono::fsi;

using std::cout;
using std::endl;

namespace chrono {
namespace vehicle {

// -----------------------------------------------------------------------------
// Construction of the terrain node:
// - create the Chrono system and set solver parameters
// - create the Chrono FSI system
// -----------------------------------------------------------------------------
ChVehicleCosimTerrainNodeGranularSPH::ChVehicleCosimTerrainNodeGranularSPH(bool render)
    : ChVehicleCosimTerrainNode(Type::GRANULAR_SPH, ChContactMethod::SMC, render), m_depth(0) {
    cout << "[Terrain node] GRANULAR_SPH " << endl;

    // Create systems
    m_system = new ChSystemSMC;
    m_systemFSI = new ChSystemFsi(*m_system);

    // Solver settings independent of method type
    m_system->Set_G_acc(ChVector<>(0, 0, m_gacc));

    // Set number of threads
    m_system->SetNumThreads(1);

#ifdef CHRONO_OPENGL
    // Create the visualization window
    if (m_render) {
        opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
        gl_window.Initialize(1280, 720, "Terrain Node", m_system);
        gl_window.SetCamera(ChVector<>(0, -6, 0), ChVector<>(0, 0, 0), ChVector<>(0, 0, 1), 0.05f);
        gl_window.SetRenderMode(opengl::WIREFRAME);
    }
#endif
}

ChVehicleCosimTerrainNodeGranularSPH::~ChVehicleCosimTerrainNodeGranularSPH() {
    delete m_systemFSI;
    delete m_system;
}

void ChVehicleCosimTerrainNodeGranularSPH::SetPropertiesSPH(const std::string& filename, double depth) {
    m_depth = depth;

    // Get the pointer to the system parameter and use a JSON file to fill it out with the user parameters
    m_params = m_systemFSI->GetSimParams();
    fsi::utils::ParseJSON(filename, m_params, fsi::mR3(0, 0, 0));
}

// -----------------------------------------------------------------------------
// Complete construction of the mechanical system.
// This function is invoked automatically from Initialize.
// - adjust system settings
// - create the container body
// - set m_init_height
// -----------------------------------------------------------------------------
void ChVehicleCosimTerrainNodeGranularSPH::Construct() {
    // Domain size
    fsi::Real bxDim = (fsi::Real)(2 * m_hdimX);
    fsi::Real byDim = (fsi::Real)(2 * m_hdimY);
    fsi::Real bzDim = (fsi::Real)(1.25 * m_depth);

    // Set up the periodic boundary condition (if not, set relative larger values)
    fsi::Real initSpace0 = m_params->MULT_INITSPACE * m_params->HSML;
    m_params->boxDimX = bxDim;
    m_params->boxDimY = byDim;
    m_params->boxDimZ = bzDim;
    m_params->cMin = chrono::fsi::mR3(-bxDim / 2, -byDim / 2, -bzDim - 10 * initSpace0) * 10;
    m_params->cMax = chrono::fsi::mR3(bxDim / 2, byDim / 2, bzDim + 10 * initSpace0) * 10;

    // Set the time integration type and the linear solver type (only for ISPH)
    m_systemFSI->SetFluidDynamics(m_params->fluid_dynamic_type);
    m_systemFSI->SetFluidSystemLinearSolver(m_params->LinearSolver);

    // Call FinalizeDomain to setup the binning for neighbor search
    fsi::utils::FinalizeDomain(m_params);

    // Create the fluid particles...
    // Fluid domain:  bxDim x byDim x m_depth
    // Dimension of the fluid domain
    fsi::Real fxDim = m_params->fluidDimX;
    fsi::Real fyDim = m_params->fluidDimY;
    fsi::Real fzDim = m_params->fluidDimZ;
    // Create Fluid region and discretize with SPH particles
    ChVector<> boxCenter(0.0, 0.0, fzDim / 2);
    ChVector<> boxHalfDim(fxDim / 2, fyDim / 2, fzDim / 2);

    // Use a chrono sampler to create a bucket of points
    utils::GridSampler<> sampler(initSpace0);
    utils::Generator::PointVector points = sampler.SampleBox(boxCenter, boxHalfDim);

    // Add fluid particles from the sampler points to the FSI system
    int numPart = (int)points.size();
    for (int i = 0; i < numPart; i++) {
        // Calculate the pressure of a steady state (p = rho*g*h)
        Real pre_ini = m_params->rho0 * abs(m_params->gravity.z) * (-points[i].z() + fzDim);
        Real rho_ini = m_params->rho0 + pre_ini / (m_params->Cs * m_params->Cs);
        m_systemFSI->GetDataManager()->AddSphMarker(
            fsi::mR4(points[i].x(), points[i].y(), points[i].z(), m_params->HSML), fsi::mR3(1e-10),
            fsi::mR4(rho_ini, pre_ini, m_params->mu0, -1));
    }

    size_t numPhases = m_systemFSI->GetDataManager()->fsiGeneralData->referenceArray.size();
    if (numPhases != 0) {
        std::cout << "ERROR: incorrect number of phases in SPH granular terrain!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    m_systemFSI->GetDataManager()->fsiGeneralData->referenceArray.push_back(fsi::mI4(0, numPart, -1, -1));
    m_systemFSI->GetDataManager()->fsiGeneralData->referenceArray.push_back(fsi::mI4(numPart, numPart, 0, 0));

    //// RADU TODO - this function must set m_init_height!
    m_init_height = fzDim;

    // Create container body
    auto container = std::shared_ptr<ChBody>(m_system->NewBody());
    m_system->AddBody(container);
    container->SetIdentifier(-1);
    container->SetMass(1);
    container->SetBodyFixed(true);
    container->SetCollide(false);

    // Create the geometry of the boundaries

    // Bottom and Top wall - size and position
    ChVector<> size_XY(bxDim / 2 + 3 * initSpace0, byDim / 2 + 3 * initSpace0, 2 * initSpace0);
    ChVector<> pos_zp(0, 0, bzDim + 1 * initSpace0);
    ChVector<> pos_zn(0, 0, -3 * initSpace0);

    // Left and right Wall - size and position
    ChVector<> size_YZ(2 * initSpace0, byDim / 2 + 3 * initSpace0, bzDim / 2);
    ChVector<> pos_xp(bxDim / 2 + initSpace0, 0.0, bzDim / 2 + 0 * initSpace0);
    ChVector<> pos_xn(-bxDim / 2 - 3 * initSpace0, 0.0, bzDim / 2 + 0 * initSpace0);

    // Front and back Wall - size and position
    ChVector<> size_XZ(bxDim / 2, 2 * initSpace0, bzDim / 2);
    ChVector<> pos_yp(0, byDim / 2 + initSpace0, bzDim / 2 + 0 * initSpace0);
    ChVector<> pos_yn(0, -byDim / 2 - 3 * initSpace0, bzDim / 2 + 0 * initSpace0);

    // Add BCE particles attached on the walls into FSI system
    fsi::utils::AddBoxBce(m_systemFSI->GetDataManager(), m_params, container, pos_zp, chrono::QUNIT, size_XY, 12);
    fsi::utils::AddBoxBce(m_systemFSI->GetDataManager(), m_params, container, pos_zn, chrono::QUNIT, size_XY, 12);
    fsi::utils::AddBoxBce(m_systemFSI->GetDataManager(), m_params, container, pos_xp, chrono::QUNIT, size_YZ, 23);
    fsi::utils::AddBoxBce(m_systemFSI->GetDataManager(), m_params, container, pos_xn, chrono::QUNIT, size_YZ, 23);
    fsi::utils::AddBoxBce(m_systemFSI->GetDataManager(), m_params, container, pos_yp, chrono::QUNIT, size_XZ, 13);
    fsi::utils::AddBoxBce(m_systemFSI->GetDataManager(), m_params, container, pos_yn, chrono::QUNIT, size_XZ, 13);

    // Add visualization assets for the container
    {
        auto box = chrono_types::make_shared<ChBoxShape>();
        box->GetBoxGeometry().Size = size_XY;
        box->Pos = pos_zp;
        container->GetAssets().push_back(box);
    }
    {
        auto box = chrono_types::make_shared<ChBoxShape>();
        box->GetBoxGeometry().Size = size_XY;
        box->Pos = pos_zn;
        container->GetAssets().push_back(box);
    }

    // Write file with terrain node settings
    std::ofstream outf;
    outf.open(m_node_out_dir + "/settings.dat", std::ios::out);
    outf << "System settings" << endl;
    outf << "   Integration step size = " << m_step_size << endl;
    outf << "Patch dimensions" << endl;
    outf << "   X = " << 2 * m_hdimX << "  Y = " << 2 * m_hdimY << endl;
    outf << "   depth = " << m_depth << endl;
}

void CreateMeshMarkers(std::shared_ptr<geometry::ChTriangleMeshConnected> mesh, double delta, std::vector<ChVector<>>& point_cloud) {
    mesh->RepairDuplicateVertexes(1e-9);  // if meshes are not watertight

    ChVector<> minV = mesh->m_vertices[0];
    ChVector<> maxV = mesh->m_vertices[0];
    ChVector<> currV = mesh->m_vertices[0];
    for (unsigned int i = 1; i < mesh->m_vertices.size(); ++i) {
        currV = mesh->m_vertices[i];
        if (minV.x() > currV.x())
            minV.x() = currV.x();
        if (minV.y() > currV.y())
            minV.y() = currV.y();
        if (minV.z() > currV.z())
            minV.z() = currV.z();
        if (maxV.x() < currV.x())
            maxV.x() = currV.x();
        if (maxV.y() < currV.y())
            maxV.y() = currV.y();
        if (maxV.z() < currV.z())
            maxV.z() = currV.z();
    }
    ////printf("start coords: %f, %f, %f\n", minV.x(), minV.y(), minV.z());
    ////printf("end coords: %f, %f, %f\n", maxV.x(), maxV.y(), maxV.z());

    const double EPSI = 1e-6;

    ChVector<> ray_origin;
    for (double x = minV.x(); x < maxV.x(); x += delta) {
        ray_origin.x() = x + 1e-9;
        for (double y = minV.y(); y < maxV.y(); y += delta) {
            ray_origin.y() = y + 1e-9;
            for (double z = minV.z(); z < maxV.z(); z += delta) {
                ray_origin.z() = z + 1e-9;

                ChVector<> ray_dir[2] = {ChVector<>(5, 0.5, 0.25), ChVector<>(-3, 0.7, 10)};
                int intersectCounter[2] = {0, 0};

                for (unsigned int i = 0; i < mesh->m_face_v_indices.size(); ++i) {
                    auto& t_face = mesh->m_face_v_indices[i];
                    auto& v1 = mesh->m_vertices[t_face.x()];
                    auto& v2 = mesh->m_vertices[t_face.y()];
                    auto& v3 = mesh->m_vertices[t_face.z()];

                    // Find vectors for two edges sharing V1
                    auto edge1 = v2 - v1;
                    auto edge2 = v3 - v1;

                    bool t_inter[2] = {false, false};

                    for (unsigned int j = 0; j < 2; j++) {
                        // Begin calculating determinant - also used to calculate uu parameter
                        auto pvec = Vcross(ray_dir[j], edge2);
                        // if determinant is near zero, ray is parallel to plane of triangle
                        double det = Vdot(edge1, pvec);
                        // NOT CULLING
                        if (det > -EPSI && det < EPSI) {
                            t_inter[j] = false;
                            continue;
                        }
                        double inv_det = 1.0 / det;

                        // calculate distance from V1 to ray origin
                        auto tvec = ray_origin - v1;

                        // Calculate uu parameter and test bound
                        double uu = Vdot(tvec, pvec) * inv_det;
                        // The intersection lies outside of the triangle
                        if (uu < 0.0 || uu > 1.0) {
                            t_inter[j] = false;
                            continue;
                        }

                        // Prepare to test vv parameter
                        auto qvec = Vcross(tvec, edge1);

                        // Calculate vv parameter and test bound
                        double vv = Vdot(ray_dir[j], qvec) * inv_det;
                        // The intersection lies outside of the triangle
                        if (vv < 0.0 || ((uu + vv) > 1.0)) {
                            t_inter[j] = false;
                            continue;
                        }

                        double tt = Vdot(edge2, qvec) * inv_det;
                        if (tt > EPSI) {  // ray intersection
                            t_inter[j] = true;
                            continue;
                        }

                        // No hit, no win
                        t_inter[j] = false;
                    }

                    intersectCounter[0] += t_inter[0] ? 1 : 0;
                    intersectCounter[1] += t_inter[1] ? 1 : 0;
                }

                if (((intersectCounter[0] % 2) == 1) && ((intersectCounter[1] % 2) == 1)) // inside mesh
                    point_cloud.push_back(ChVector<>(x, y, z));
            }
        }
    }
}

void ChVehicleCosimTerrainNodeGranularSPH::CreateWheelProxy() {
    // Create wheel proxy body
    auto body = std::shared_ptr<ChBody>(m_system->NewBody());
    body->SetIdentifier(0);
    body->SetMass(m_rig_mass);
    body->SetBodyFixed(true);  // proxy body always fixed
    body->SetCollide(false);

    // Create collision mesh
    auto trimesh = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
    trimesh->getCoordsVertices() = m_mesh_data.vpos;
    trimesh->getCoordsNormals() = m_mesh_data.vnrm;
    trimesh->getIndicesVertexes() = m_mesh_data.tri;

    // Set visualization asset
    auto trimesh_shape = chrono_types::make_shared<ChTriangleMeshShape>();
    trimesh_shape->SetMesh(trimesh);
    trimesh_shape->Pos = ChVector<>(0, 0, 0);
    trimesh_shape->Rot = ChQuaternion<>(1, 0, 0, 0);
    body->GetAssets().push_back(trimesh_shape);

    m_system->AddBody(body);
    m_proxies.push_back(ProxyBody(body, 0));

    // Add this body to the FSI system
    m_systemFSI->AddFsiBody(body);

    // Create BCE markers associated with trimesh
    auto initSpace0 = m_params->MULT_INITSPACE * m_params->HSML;
    std::vector<ChVector<>> point_cloud;
    CreateMeshMarkers(trimesh, (double)initSpace0, point_cloud);
    fsi::utils::AddBCE_FromPoints(m_systemFSI->GetDataManager(), m_params, body, point_cloud, VNULL, QUNIT);    

    // Construction of the FSI system must be finalized before running
    m_systemFSI->Finalize();
}

// Set state of wheel proxy body.
void ChVehicleCosimTerrainNodeGranularSPH::UpdateWheelProxy() {
    m_proxies[0].m_body->SetPos(m_wheel_state.pos);
    m_proxies[0].m_body->SetPos_dt(m_wheel_state.lin_vel);
    m_proxies[0].m_body->SetRot(m_wheel_state.rot);
    m_proxies[0].m_body->SetWvel_par(m_wheel_state.ang_vel);
    m_proxies[0].m_body->SetWacc_par(ChVector<>(0.0, 0.0, 0.0));
}

// Collect resultant contact force and torque on wheel proxy body.
void ChVehicleCosimTerrainNodeGranularSPH::GetForceWheelProxy() {
    m_wheel_contact.point = ChVector<>(0, 0, 0);
    m_wheel_contact.force = m_proxies[0].m_body->Get_accumulated_force();
    m_wheel_contact.moment = m_proxies[0].m_body->Get_accumulated_torque();
}

// -----------------------------------------------------------------------------

void ChVehicleCosimTerrainNodeGranularSPH::Advance(double step_size) {
    //// RADU TODO:  correlate m_step_size with m_params->dT

    m_timer.reset();
    m_timer.start();
    double t = 0;
    while (t < step_size) {
        double h = std::min<>(m_params->dT, step_size - t);
        m_systemFSI->DoStepDynamics_FSI();
        t += h;
    }
    m_timer.stop();
    m_cum_sim_time += m_timer();

#ifdef CHRONO_OPENGL
    if (m_render) {
        opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
        if (gl_window.Active()) {
            ChVector<> cam_point = m_proxies[0].m_body->GetPos();
            ChVector<> cam_loc = cam_point + ChVector<>(0, -3, 0.6);
            gl_window.SetCamera(cam_loc, cam_point, ChVector<>(0, 0, 1), 0.05f);
            gl_window.Render();
        } else {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
#endif

    PrintWheelProxyContactData();
}

// -----------------------------------------------------------------------------

void ChVehicleCosimTerrainNodeGranularSPH::OutputTerrainData(int frame) {
    //// TODO
}

// -----------------------------------------------------------------------------

void ChVehicleCosimTerrainNodeGranularSPH::PrintWheelProxyUpdateData() {
    //// TODO
}

void ChVehicleCosimTerrainNodeGranularSPH::PrintWheelProxyContactData() {
    //// TODO
}

}  // end namespace vehicle
}  // end namespace chrono
