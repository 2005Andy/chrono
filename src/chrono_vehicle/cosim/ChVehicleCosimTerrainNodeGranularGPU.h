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
// Definition of the GPU granular TERRAIN NODE (using Chrono::Granular).
//
// The global reference frame has Z up, X towards the front of the vehicle, and
// Y pointing to the left.
//
// =============================================================================

#ifndef TESTRIG_TERRAINNODE_GRANULAR_GPU_H
#define TESTRIG_TERRAINNODE_GRANULAR_GPU_H

#include "chrono/physics/ChSystemSMC.h"
#include "chrono_gpu/physics/ChSystemGpu.h"
#include "chrono_vehicle/cosim/ChVehicleCosimTerrainNode.h"

namespace chrono {
namespace vehicle {

class CH_VEHICLE_API ChVehicleCosimTerrainNodeGranularGPU : public ChVehicleCosimTerrainNode {
  public:
    /// Create a Chrono::Granular terrain subsystem.
    ChVehicleCosimTerrainNodeGranularGPU();

    ~ChVehicleCosimTerrainNodeGranularGPU();

    virtual ChSystem* GetSystem() override { return m_system; }

    /// Set properties of granular material.
    void SetGranularMaterial(double radius,   ///< particle radius (default: 0.01)
                             double density,  ///< particle material density (default: 2000)
                             int num_layers   ///< number of generated particle layers (default: 5)
    );

    /// Set the material properties for terrain.
    /// These parameters characterize the material for the container and the granular material.
    /// Tire contact material is received from the rig node.
    void SetMaterialSurface(const std::shared_ptr<ChMaterialSurfaceSMC>& mat);

    /// Set the normal contact force model (default: Hertz)
    ////void SetContactForceModel(ChSystemSMC::ContactForceModel model);

    /// Set the tangential contact displacement model (default: SINGLE_STEP)
    void SetTangentialDisplacementModel(gpu::CHGPU_FRICTION_MODE model);

    /// Set the integrator type (default: CENTERED_DIFFERENCE)
    void SetIntegratorType(gpu::CHGPU_TIME_INTEGRATOR type);

    /// Initialize granular terrain from the specified checkpoint file (which must exist in the output directory).
    /// By default, particles are created uniformly distributed in the specified domain such that they are initially not
    /// in contact.
    void SetInputFromCheckpoint(const std::string& filename);

    /// Set simulation length for settling of granular material (default: 0.4).
    void SetSettlingTime(double time) { m_time_settling = time; }

    /// Enable/disable output during settling (default: false).
    /// If enabled, output files are generated with the specified frequency.
    void EnableSettlingOutput(bool output, double output_fps = 100);

    /// Obtain settled terrain configuration.
    /// This is an optional operation that can be performed for granular terrain before initiating
    /// communictation with the rig node. For granular terrain, a settled configuration can
    /// be obtained either through simulation or by initializing particles from a previously
    /// generated checkpointing file.
    void Settle();

    /// Write checkpoint to the specified file (which will be created in the output directory).
    virtual void WriteCheckpoint(const std::string& filename) override;

  private:
    ChSystemSMC* m_system;              ///< system for proxy bodies
    gpu::ChSystemGpuMesh* m_systemGPU;  ///< Chrono::Gpu system
    bool m_constructed;                 ///< system construction completed?

    gpu::CHGPU_TIME_INTEGRATOR m_integrator_type;
    gpu::CHGPU_FRICTION_MODE m_tangential_model;

    bool m_use_checkpoint;              ///< initialize granular terrain from checkpoint file
    std::string m_checkpoint_filename;  ///< name of input checkpoint file

    int m_num_layers;              ///< number of generated particle layers
    unsigned int m_num_particles;  ///< number of granular material bodies
    double m_radius_g;             ///< radius of one particle of granular material
    double m_rho_g;                ///< particle material density

    double m_time_settling;  ///< simulation length for settling of granular material
    bool m_settling_output;  ///< output files during settling?
    double m_settling_fps;   ///< frequency of output during settling phase

    virtual bool SupportsFlexibleTire() const override { return false; }

    /// Construct granular terrain
    virtual void Construct() override;

    /// Return current total number of contacts.
    virtual int GetNumContacts() const override { return m_systemGPU->GetNumContacts(); }

    virtual void CreateWheelProxy() override;
    virtual void UpdateWheelProxy() override;
    virtual void GetForceWheelProxy() override;

    virtual void PrintWheelProxyUpdateData() override;
    virtual void PrintWheelProxyContactData() override;

    virtual void OnOutputData(int frame) override;

    virtual void OnRender(double time) override;

    /// Advance simulation.
    /// This function is called after a synchronization to allow the node to advance
    /// its state by the specified time step.  A node is allowed to take as many internal
    /// integration steps as required, but no inter-node communication should occur.
    virtual void Advance(double step_size) override;

    /// Set composite material properties for internal granular system contacts.
    void SetMatPropertiesInternal();

    /// Set composite material properties for granular-tire contacts
    /// (can be invoked only once tire material was received).
    void SetMatPropertiesExternal();

    /// Update position of visualization shapes for granular material.
    /// Note that this requires memory transfer from GPU.
    void UpdateVisualizationParticles();
};

}  // end namespace vehicle
}  // end namespace chrono

#endif