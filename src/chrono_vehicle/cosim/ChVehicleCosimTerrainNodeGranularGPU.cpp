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
// Authors: Radu Serban
// =============================================================================
//
// Implementation of the GPU granular TERRAIN NODE (using Chrono::Granular).
//
// The global reference frame has Z up, X towards the front of the vehicle, and
// Y pointing to the left.
//
// =============================================================================

#include <algorithm>
#include <iomanip>
#include <limits>
#include <mpi.h>

#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono/assets/ChTriangleMeshShape.h"
#include "chrono/assets/ChSphereShape.h"

#include "chrono_granular/api/ChApiGranularChrono.h"

#include "chrono_vehicle/cosim/ChVehicleCosimTerrainNodeGranularGPU.h"

#ifdef CHRONO_OPENGL
#include "chrono_opengl/ChOpenGLWindow.h"
#endif

using std::cout;
using std::endl;

namespace chrono {
namespace vehicle {

ChVehicleCosimTerrainNodeGranularGPU::ChVehicleCosimTerrainNodeGranularGPU(bool render)
    : ChVehicleCosimTerrainNode(Type::GRANULAR_GPU, ChContactMethod::SMC, render),
      m_constructed(false),
      m_use_checkpoint(false),
      m_settling_output(false),
      m_num_particles(0) {
    cout << "[Terrain node] GRANULAR_GPU " << endl;

    // Default granular material properties
    m_radius_g = 0.01;
    m_rho_g = 2000;
    m_num_layers = 5;
    m_time_settling = 0.4;

    // Default granular system settings
    m_integrator_type = granular::GRAN_TIME_INTEGRATOR::CENTERED_DIFFERENCE;
    m_tangential_model = granular::GRAN_FRICTION_MODE::MULTI_STEP;

    // Create systems
    m_system = new ChSystemSMC();
    m_system->Set_G_acc(ChVector<>(0, 0, m_gacc));

    // Defer construction of the granular system to Construct
    //// TODO: why can I not modify parameters AFTER construction?!?
    m_wrapper_gran = nullptr;

#ifdef CHRONO_OPENGL
    // Create the visualization window
    if (m_render) {
        opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
        gl_window.Initialize(1280, 720, "Terrain Node (GranularGPU)", m_system);
        gl_window.SetCamera(ChVector<>(0, -3, 0), ChVector<>(0, 0, 0), ChVector<>(0, 0, 1), 0.05f);
        gl_window.SetRenderMode(opengl::WIREFRAME);
    }
#endif
}

ChVehicleCosimTerrainNodeGranularGPU ::~ChVehicleCosimTerrainNodeGranularGPU() {
    delete m_system;
    delete m_wrapper_gran;
}

// -----------------------------------------------------------------------------

void ChVehicleCosimTerrainNodeGranularGPU::SetGranularMaterial(double radius, double density, int num_layers) {
    m_radius_g = radius;
    m_rho_g = density;
    m_num_layers = num_layers;
}

////void ChVehicleCosimTerrainNodeGranularGPU::SetContactForceModel(ChSystemSMC::ContactForceModel model) {
////}

void ChVehicleCosimTerrainNodeGranularGPU::SetIntegratorType(granular::GRAN_TIME_INTEGRATOR type) {
    m_integrator_type = type;
}

void ChVehicleCosimTerrainNodeGranularGPU::SetTangentialDisplacementModel(granular::GRAN_FRICTION_MODE model) {
    m_tangential_model = model;
}

void ChVehicleCosimTerrainNodeGranularGPU::SetMaterialSurface(const std::shared_ptr<ChMaterialSurfaceSMC>& mat) {
    m_material_terrain = mat;
}

void ChVehicleCosimTerrainNodeGranularGPU::SetInputFromCheckpoint(const std::string& filename) {
    m_use_checkpoint = true;
    m_checkpoint_filename = filename;
}

// -----------------------------------------------------------------------------
// Complete construction of the mechanical system.
// This function is invoked automatically from Settle and Initialize.
// - adjust system settings
// - create the container body (implicit boundaries?)
// - create the granular material
// -----------------------------------------------------------------------------
void ChVehicleCosimTerrainNodeGranularGPU::Construct() {
    if (m_constructed)
        return;

    // Create granular system here
    //// TODO: why can I not modify parameters AFTER construction?!?
    //// TODO: there's an implicit assumption that the origin is at the center of the box!?!
    //// TODO: why is not ChGranularChronoTriMeshAPI in the chrono::granular namespace?!?
    //// TODO: what is the point of this "API" wrapper?!?
    //// TODO: Why can I not specify output mode when I actually do output?!?!

    // Calculate domain size
    float separation_factor = 1.2f;
    float r = separation_factor * (float)m_radius_g; 
    float delta = 2.0f * r;
    float dimX = 2.0f * (float)m_hdimX;
    float dimY = 2.0f * (float)m_hdimY;
    float dimZ = (m_num_layers + 1) * delta;

    auto box = make_float3(dimX, dimY, dimZ);
    m_wrapper_gran = new ChGranularChronoTriMeshAPI((float)m_radius_g, (float)m_rho_g, box);
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_gravitational_acceleration(0, 0, (float)m_gacc);
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_timeIntegrator(m_integrator_type);
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_friction_mode(m_tangential_model);
    m_wrapper_gran->getGranSystemSMC_TriMesh().setOutputMode(granular::GRAN_OUTPUT_MODE::CSV);
    m_wrapper_gran->getGranSystemSMC_TriMesh().setVerbose(granular::GRAN_VERBOSITY::QUIET);

    // Set composite material properties for internal contacts.
    // Defer setting composite material properties for external contacts until creation of proxies (when we have
    // received tire material)
    SetMatPropertiesInternal();

    // Set integration step-size
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_fixed_stepSize((float)m_step_size);

    // Create granular material
    std::vector<ChVector<float>> pos(m_num_particles);

    if (m_use_checkpoint) {
        // Read particle state from checkpoint file
        std::string checkpoint_filename = m_node_out_dir + "/" + m_checkpoint_filename;
        std::ifstream ifile(checkpoint_filename);
        if (!ifile.is_open()) {
            cout << "ERROR: could not open checkpoint file " << checkpoint_filename << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Read and discard line with current time
        std::string line;
        std::getline(ifile, line);

        // Read number of particles in checkpoint
        std::getline(ifile, line);
        std::istringstream iss(line);
        iss >> m_num_particles;

        pos.resize(m_num_particles);
        std::vector<ChVector<float>> vel(m_num_particles);
        std::vector<ChVector<float>> omg(m_num_particles);
        for (unsigned int i = 0; i < m_num_particles; i++) {
            std::getline(ifile, line);
            std::istringstream iss(line);
            int identifier;
            iss >> identifier >> pos[i].x() >> pos[i].y() >> pos[i].z() >> vel[i].x() >> vel[i].y() >> vel[i].z() >>
                omg[i].x() >> omg[i].y() >> omg[i].z();
            assert(identifier == i);
        }
        m_wrapper_gran->setElemsPositions(pos, vel, omg);

        cout << "[Terrain node] read " << checkpoint_filename << "   num. particles = " << m_num_particles << endl;
    } else {
        // Generate particles in layers using a Poisson disk sampler
        utils::PDSampler<float> sampler(delta);
        ChVector<float> hdims(dimX / 2 - r, dimY / 2 - r, 0);
        ChVector<float> center(0, 0, -dimZ / 2 + delta);
        for (int il = 0; il < m_num_layers; il++) {
            auto p = sampler.SampleBox(center, hdims);
            pos.insert(pos.end(), p.begin(), p.end());
            center.z() += delta;
        }
        m_wrapper_gran->setElemsPositions(pos);
        m_num_particles = (unsigned int)pos.size();
        cout << "[Terrain node] Generated num particles = " << m_num_particles << endl;
    }

    m_wrapper_gran->getGranSystemSMC_TriMesh().set_BD_Fixed(true);

    // Find "height" of granular material
    //// TODO: cannot call get_max_z() here!!!
    ////CalcInitHeight();
    auto init_height = -std::numeric_limits<float>::max();
    for (const auto& p : pos) {
        if (p.z() > init_height)
            init_height = p.z();
    }
    m_init_height = (double)init_height + m_radius_g;
    cout << "[Terrain node] initial height = " << m_init_height << endl;

    // Mark system as constructed.
    m_constructed = true;

    // Create bodies in Chrono system (visualization only)
    if (m_render) {
        for (const auto& p : pos) {
            auto body = std::shared_ptr<ChBody>(m_system->NewBody());
            body->SetPos(p);
            body->SetBodyFixed(true);
            auto sph = chrono_types::make_shared<ChSphereShape>();
            sph->GetSphereGeometry().rad = m_radius_g;
            body->AddAsset(sph);
            m_system->AddBody(body);
        }
    }

    // --------------------------------------
    // Write file with terrain node settings
    // --------------------------------------

    std::ofstream outf;
    outf.open(m_node_out_dir + "/settings.dat", std::ios::out);
    outf << "System settings" << endl;
    outf << "   Integration step size = " << m_step_size << endl;
    outf << "Terrain patch dimensions" << endl;
    outf << "   X = " << 2 * m_hdimX << "  Y = " << 2 * m_hdimY << endl;
    outf << "Terrain material properties" << endl;
    auto mat = std::static_pointer_cast<ChMaterialSurfaceSMC>(m_material_terrain);
    outf << "   Coefficient of friction    = " << mat->GetKfriction() << endl;
    outf << "   Coefficient of restitution = " << mat->GetRestitution() << endl;
    outf << "   Young modulus              = " << mat->GetYoungModulus() << endl;
    outf << "   Poisson ratio              = " << mat->GetPoissonRatio() << endl;
    outf << "   Adhesion force             = " << mat->GetAdhesion() << endl;
    outf << "   Kn = " << mat->GetKn() << endl;
    outf << "   Gn = " << mat->GetGn() << endl;
    outf << "   Kt = " << mat->GetKt() << endl;
    outf << "   Gt = " << mat->GetGt() << endl;
    outf << "Granular material properties" << endl;
    outf << "   particle radius  = " << m_radius_g << endl;
    outf << "   particle density = " << m_rho_g << endl;
    outf << "   number layers    = " << m_num_layers << endl;
    outf << "   number particles = " << m_num_particles << endl;
}

// -----------------------------------------------------------------------------
// Settling phase for the terrain node
// - settle terrain through simulation
// - update initial height of terrain
// -----------------------------------------------------------------------------
void ChVehicleCosimTerrainNodeGranularGPU::Settle() {
    Construct();

    // Complete construction of the granular system
    m_wrapper_gran->getGranSystemSMC_TriMesh().initialize();

    // Simulate settling of granular terrain
    double output_fps = 100;
    int sim_steps = (int)std::ceil(m_time_settling / m_step_size);
    int output_steps = (int)std::ceil(1 / (output_fps * m_step_size));
    int output_frame = 0;

    for (int is = 0; is < sim_steps; is++) {
        // Advance step
        m_timer.reset();
        m_timer.start();
        m_system->DoStepDynamics(m_step_size);
        m_wrapper_gran->getGranSystemSMC_TriMesh().advance_simulation((float)m_step_size);
        m_timer.stop();
        m_cum_sim_time += m_timer();
        cout << '\r' << std::fixed << std::setprecision(6) << m_system->GetChTime() << "  [" << m_timer.GetTimeSeconds()
             << "]   " << GetNumContacts() << std::flush;

        // Output (if enabled)
        if (m_settling_output && is % output_steps == 0) {
            char filename[100];
            sprintf(filename, "%s/settling_%04d", m_node_out_dir.c_str(), output_frame + 1);
            m_wrapper_gran->getGranSystemSMC_TriMesh().setOutputFlags(granular::GRAN_OUTPUT_FLAGS::VEL_COMPONENTS |
                                                                      granular::GRAN_OUTPUT_FLAGS::ANG_VEL_COMPONENTS |
                                                                      granular::GRAN_OUTPUT_FLAGS::FORCE_COMPONENTS);
            m_wrapper_gran->getGranSystemSMC_TriMesh().writeFile(filename);
            output_frame++;
        }

#ifdef CHRONO_OPENGL
        // OpenGL rendering
        if (m_render) {
            UpdateVisualizationParticles();
            opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
            if (gl_window.Active()) {
                gl_window.Render();
            } else {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
#endif
    }

    cout << endl;
    cout << "[Terrain node] settling time = " << m_cum_sim_time << endl;
    m_cum_sim_time = 0;

    // Find "height" of granular material after settling
    m_init_height = m_wrapper_gran->getGranSystemSMC_TriMesh().get_max_z() + m_radius_g;
    cout << "[Terrain node] initial height = " << m_init_height << endl;
}

// -----------------------------------------------------------------------------

// Set composite material properties for internal contacts (assume same material for spheres and walls)
void ChVehicleCosimTerrainNodeGranularGPU::SetMatPropertiesInternal() {
    auto material_terrain = std::static_pointer_cast<ChMaterialSurfaceSMC>(m_material_terrain);

    m_wrapper_gran->getGranSystemSMC_TriMesh().set_K_n_SPH2SPH(material_terrain->GetKn());
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_K_n_SPH2WALL(material_terrain->GetKn());

    m_wrapper_gran->getGranSystemSMC_TriMesh().set_Gamma_n_SPH2SPH(material_terrain->GetGn());
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_Gamma_n_SPH2WALL(material_terrain->GetGn());

    m_wrapper_gran->getGranSystemSMC_TriMesh().set_K_t_SPH2SPH(material_terrain->GetKt());
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_K_t_SPH2WALL(material_terrain->GetKt());

    m_wrapper_gran->getGranSystemSMC_TriMesh().set_Gamma_t_SPH2SPH(material_terrain->GetGt());
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_Gamma_t_SPH2WALL(material_terrain->GetGt());

    m_wrapper_gran->getGranSystemSMC_TriMesh().set_static_friction_coeff_SPH2SPH(material_terrain->GetSfriction());
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_static_friction_coeff_SPH2WALL(material_terrain->GetSfriction());

    //// TODO: adhesion/cohesion
    //// TODO: why are cohesion and adhesion defined as ratios to sphere weight?!?
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_Cohesion_ratio(0);
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_Adhesion_ratio_S2W(0);
}

// Set composite material properties for external contacts (granular-tire)
void ChVehicleCosimTerrainNodeGranularGPU::SetMatPropertiesExternal() {
    auto material_terrain = std::static_pointer_cast<ChMaterialSurfaceSMC>(m_material_terrain);
    auto material_tire = std::static_pointer_cast<ChMaterialSurfaceSMC>(m_material_tire);

    const auto& strategy = m_system->GetMaterialCompositionStrategy();
    auto Kn = strategy.CombineStiffnessCoefficient(material_terrain->GetKn(), material_tire->GetKn());
    auto Kt = strategy.CombineStiffnessCoefficient(material_terrain->GetKt(), material_tire->GetKt());
    auto Gn = strategy.CombineDampingCoefficient(material_terrain->GetGn(), material_tire->GetGn());
    auto Gt = strategy.CombineDampingCoefficient(material_terrain->GetGt(), material_tire->GetGt());
    auto mu = strategy.CombineFriction(m_material_terrain->GetSfriction(), m_material_tire->GetSfriction());

    m_wrapper_gran->getGranSystemSMC_TriMesh().set_K_n_SPH2MESH(Kn);
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_Gamma_n_SPH2MESH(Gn);
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_K_t_SPH2MESH(Kt);
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_Gamma_t_SPH2MESH(Gt);
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_static_friction_coeff_SPH2MESH(mu);

    //// TODO: adhesion/cohesion
    //// TODO: why are cohesion and adhesion defined as ratios to sphere weight?!?
    m_wrapper_gran->getGranSystemSMC_TriMesh().set_Adhesion_ratio_S2M(0);
}

// -----------------------------------------------------------------------------

void ChVehicleCosimTerrainNodeGranularGPU::CreateWheelProxy() {
    auto body = std::shared_ptr<ChBody>(m_system->NewBody());
    body->SetIdentifier(0);
    body->SetMass(m_rig_mass);
    ////body->SetInertiaXX();   //// TODO
    body->SetBodyFixed(m_fixed_proxies);
    body->SetCollide(true);

    // Create collision mesh
    auto trimesh = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
    trimesh->getCoordsVertices() = m_mesh_data.verts;
    trimesh->getCoordsNormals() = m_mesh_data.norms;
    trimesh->getIndicesVertexes() = m_mesh_data.idx_verts;
    trimesh->getIndicesNormals() = m_mesh_data.idx_norms;

    // Set visualization asset
    auto trimesh_shape = chrono_types::make_shared<ChTriangleMeshShape>();
    trimesh_shape->SetMesh(trimesh);
    trimesh_shape->Pos = ChVector<>(0, 0, 0);
    trimesh_shape->Rot = ChQuaternion<>(1, 0, 0, 0);
    body->GetAssets().push_back(trimesh_shape);

    m_system->AddBody(body);
    m_proxies.push_back(ProxyBody(body, 0));

    // Set mesh for granular system
    std::vector<geometry::ChTriangleMeshConnected> all_meshes = {*trimesh};
    std::vector<float> masses = {(float)m_rig_mass};
    m_wrapper_gran->set_meshes(all_meshes, masses);

    // Set composite material properties for external contacts and complete construction of the granular system
    SetMatPropertiesExternal();
    m_wrapper_gran->getGranSystemSMC_TriMesh().initialize();
}

// Set state of wheel proxy body.
void ChVehicleCosimTerrainNodeGranularGPU::UpdateWheelProxy() {
    m_proxies[0].m_body->SetPos(m_wheel_state.pos);
    m_proxies[0].m_body->SetPos_dt(m_wheel_state.lin_vel);
    m_proxies[0].m_body->SetRot(m_wheel_state.rot);
    m_proxies[0].m_body->SetWvel_par(m_wheel_state.ang_vel);

    unsigned int nSoupFamilies = m_wrapper_gran->getGranSystemSMC_TriMesh().getNumTriangleFamilies();
    assert(nSoupFamilies == 1);
    double* meshPosRot = new double[7];
    float* meshVel = new float[6]();

    meshPosRot[0] = m_wheel_state.pos.x();
    meshPosRot[1] = m_wheel_state.pos.y();
    meshPosRot[2] = m_wheel_state.pos.z();
    meshPosRot[3] = m_wheel_state.rot[0];
    meshPosRot[4] = m_wheel_state.rot[1];
    meshPosRot[5] = m_wheel_state.rot[2];
    meshPosRot[6] = m_wheel_state.rot[3];

    meshVel[0] = (float)m_wheel_state.lin_vel.x();
    meshVel[1] = (float)m_wheel_state.lin_vel.y();
    meshVel[2] = (float)m_wheel_state.lin_vel.z();
    meshVel[3] = (float)m_wheel_state.ang_vel.x();
    meshVel[4] = (float)m_wheel_state.ang_vel.y();
    meshVel[5] = (float)m_wheel_state.ang_vel.z();

    m_wrapper_gran->getGranSystemSMC_TriMesh().meshSoup_applyRigidBodyMotion(meshPosRot, meshVel);
}

// Collect resultant contact force and torque on wheel proxy body.
void ChVehicleCosimTerrainNodeGranularGPU::GetForceWheelProxy() {
    float force[6];
    m_wrapper_gran->getGranSystemSMC_TriMesh().collectGeneralizedForcesOnMeshSoup(force);

    m_wheel_contact.point = ChVector<>(0, 0, 0);
    m_wheel_contact.force = ChVector<>(force[0], force[1], force[2]);
    m_wheel_contact.moment = ChVector<>(force[3], force[4], force[5]);
}

// -----------------------------------------------------------------------------

void ChVehicleCosimTerrainNodeGranularGPU::Advance(double step_size) {
    m_timer.reset();
    m_timer.start();
    double t = 0;
    while (t < step_size) {
        double h = std::min<>(m_step_size, step_size - t);
        m_system->DoStepDynamics(h);
        m_wrapper_gran->getGranSystemSMC_TriMesh().advance_simulation((float)h);
        t += h;
    }
    m_timer.stop();
    m_cum_sim_time += m_timer();
     
#ifdef CHRONO_OPENGL
    if (m_render) {
        opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
        if (gl_window.Active()) {
            UpdateVisualizationParticles();
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

void ChVehicleCosimTerrainNodeGranularGPU::UpdateVisualizationParticles() {
    // Note: it is assumed that the visualization bodies were created before the proxy body(ies).
    const auto& blist = m_system->Get_bodylist();
    for (unsigned int i = 0; i < m_num_particles; i++) {
        auto pos = m_wrapper_gran->getPosition(i);
        blist[i]->SetPos(pos);
    }
}

// -----------------------------------------------------------------------------

void ChVehicleCosimTerrainNodeGranularGPU::WriteCheckpoint(const std::string& filename) {
    assert(m_num_particles == m_wrapper_gran->getGranSystemSMC_TriMesh().getNumSpheres());
    utils::CSV_writer csv(" ");

    // Write current time and number of granular material bodies.
    csv << m_system->GetChTime() << endl;
    csv << m_num_particles << endl;

    for (unsigned int i = 0; i < m_num_particles; i++) {
        auto pos = m_wrapper_gran->getPosition(i);
        auto vel = m_wrapper_gran->getVelo(i);
        auto omg = m_wrapper_gran->getAngularVelo(i);
        csv << i << pos << vel << omg << endl;
    }

    std::string checkpoint_filename = m_node_out_dir + "/" + filename;
    csv.write_to_file(checkpoint_filename);
    cout << "[Terrain node] write checkpoint ===> " << checkpoint_filename << endl;
}

// -----------------------------------------------------------------------------

void ChVehicleCosimTerrainNodeGranularGPU::OnOutputData(int frame) {
    // Create and write frame output file.
    //// TODO: why do I not specify the file extension as well?!?
    char filename[100];
    sprintf(filename, "%s/data_%04d", m_node_out_dir.c_str(), frame + 1);
    m_wrapper_gran->getGranSystemSMC_TriMesh().setOutputFlags(granular::GRAN_OUTPUT_FLAGS::VEL_COMPONENTS |
                                                              granular::GRAN_OUTPUT_FLAGS::ANG_VEL_COMPONENTS |
                                                              granular::GRAN_OUTPUT_FLAGS::FORCE_COMPONENTS);
    m_wrapper_gran->getGranSystemSMC_TriMesh().writeFile(filename);
}

// -----------------------------------------------------------------------------

void ChVehicleCosimTerrainNodeGranularGPU::PrintWheelProxyUpdateData() {
    //// TODO
}

void ChVehicleCosimTerrainNodeGranularGPU::PrintWheelProxyContactData() {
    //// RADU TODO: implement this
}



}  // end namespace vehicle
}  // end namespace chrono
