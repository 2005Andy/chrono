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
// Definition of the base class TERRAIN NODE.
//
// The global reference frame has Z up, X towards the front of the vehicle, and
// Y pointing to the left.
//
// =============================================================================

#ifndef CH_VEHCOSIM__TERRAINNODE_H
#define CH_VEHCOSIM__TERRAINNODE_H

#include "chrono/ChConfig.h"

#include "chrono/physics/ChSystem.h"
#include "chrono/physics/ChSystemSMC.h"

#include "chrono_vehicle/ChSubsysDefs.h"
#include "chrono_vehicle/cosim/ChVehicleCosimBaseNode.h"

namespace chrono {
namespace vehicle {

/// Base class for all terrain nodes.
class CH_VEHICLE_API ChVehicleCosimTerrainNode : public ChVehicleCosimBaseNode {
  public:
    enum class Type { RIGID, SCM, GRANULAR_OMP, GRANULAR_GPU, GRANULAR_MPI, GRANULAR_SPH };

    virtual ~ChVehicleCosimTerrainNode() {}

    Type GetType() const { return m_type; }

    /// Set terrain patch dimensions (length and width).
    void SetPatchDimensions(double length,  ///< length in direction X (default: 2)
                            double width    ///< width in Y direction (default: 0.5)
    );

    /// Set properties of proxy bodies.
    /// This can be a single proxy body (for a rigid tire) or a collection of proxy bodies
    /// corresponding to an FEA mesh (for a flexible tire).  In the latter case, a concrete
    /// terrain class may create spherical proxy bodies or triangle proxy bodies.
    void SetProxyProperties(double mass,    ///< mass of a proxy body (default: 1)
                            double radius,  ///< contact radius of a proxy body (default: 0.01)
                            bool fixed      ///< proxies fixed to ground? (default: false)
    );

    /// Set the material properties for terrain.
    /// The type of material must be consistent with the contact method (penalty or complementarity)
    /// specified at construction. These parameters characterize the material for the container and
    /// (if applicable) the granular material.  Tire contact material is received from the rig node.
    virtual void SetMaterialSurface(const std::shared_ptr<ChMaterialSurface>& mat) = 0;

    /// Specify whether contact coefficients are based on material properties (default: true).
    /// Note that this setting is only relevant when using the SMC formulation.
    virtual void UseMaterialProperties(bool flag) = 0;

    /// Set the normal contact force model (default: Hertz)
    /// Note that this setting is only relevant when using the SMC formulation.
    virtual void SetContactForceModel(ChSystemSMC::ContactForceModel model) = 0;

    /// Obtain settled terrain configuration.
    /// This is an optional operation that a terrain subsystem may perform before
    /// communication with the rig node is initiated.
    virtual void Settle() = 0;

    /// Initialize this node.
    /// This function allows the node to initialize itself and, optionally, perform an
    /// initial data exchange with any other node.
    virtual void Initialize() override final;

    /// Synchronize this node.
    /// This function is called at every co-simulation synchronization time to
    /// allow the node to exchange information with any other node.
    virtual void Synchronize(int step_number, double time) override final;

    /// Advance simulation.
    /// This function is called after a synchronization to allow the node to advance
    /// its state by the specified time step.  A node is allowed to take as many internal
    /// integration steps as required, but no inter-node communication should occur.
    virtual void Advance(double step_size) override;

    /// Output logging and debugging data.
    virtual void OutputData(int frame) override final;

  protected:
    /// Association between a proxy body and a mesh index.
    /// The body can be associated with either a mesh vertex or a mesh triangle.
    struct ProxyBody {
        ProxyBody(std::shared_ptr<ChBody> body, int index) : m_body(body), m_index(index) {}
        std::shared_ptr<ChBody> m_body;
        int m_index;
    };

    Type m_type;  ///< terrain type

    bool m_render;  ///< if true, use OpenGL rendering

    ChContactMethod m_method;                               ///< contact method (SMC or NSC)
    std::shared_ptr<ChMaterialSurface> m_material_terrain;  ///< material properties for terrain bodies
    std::shared_ptr<ChMaterialSurface> m_material_tire;     ///< material properties for proxy bodies

    std::vector<ProxyBody> m_proxies;  ///< list of proxy bodies with associated mesh index
    bool m_fixed_proxies;              ///< flag indicating whether or not proxy bodies are fixed to ground
    double m_mass_p;                   ///< mass of a proxy body
    double m_radius_p;                 ///< radius for a proxy body

    double m_hdimX;  ///< patch half-length (X direction)
    double m_hdimY;  ///< patch half-width (Y direction)

    // Communication data
    MeshData m_mesh_data;          ///< tire mesh data
    MeshState m_mesh_state;        ///< tire mesh state (used for flexible tire)
    MeshContact m_mesh_contact;    ///< tire mesh contact forces (used for flexible tire)
    WheelState m_wheel_state;      ///< wheel state (used for rigid tire)
    TerrainForce m_wheel_contact;  ///< wheel contact force (used for rigid tire)

    bool m_flexible_tire;  ///< flag indicating whether the tire is flexible or rigid
    double m_init_height;  ///< initial terrain height (after optional settling)

    /// Construct a base class terrain node.
    ChVehicleCosimTerrainNode(Type type,               ///< terrain type (RIGID or GRANULAR)
                              ChContactMethod method,  ///< contact method (penalty or complementatiry)
                              bool render              ///< use OpenGL rendering
    );

    /// Print vertex and face connectivity data, as received from the rig node at synchronization.
    /// Invoked only for a flexible tire.
    void PrintMeshUpdateData();

    /// Return a pointer to the terrain node underlying Chrono system.
    virtual ChSystem* GetSystem() = 0;

    /// Construct the terrain (indpendent of the rig system).
    virtual void Construct() = 0;

    /// Specify whether or not flexible tire is supported.
    virtual bool SupportsFlexibleTire() const = 0;

    // --- Virtual methods for a flexible tire

    /// Create proxy bodies for a flexible tire mesh.
    /// Use information in the m_mesh_data struct (vertex positions expressed in local frame).
    virtual void CreateMeshProxies() = 0;
    /// Update the state of all proxy bodies for a flexible tire.
    /// Use information in the m_mesh_state struct (vertex positions and velocities expressed in absolute frame).
    virtual void UpdateMeshProxies() = 0;
    /// Collect cumulative contact forces on all proxy bodies for a flexible tire.
    /// Load indices of vertices in contact and the corresponding vertex forces (expressed in absolute frame)
    /// into the m_mesh_contact struct.
    virtual void GetForcesMeshProxies() = 0;
    /// Print information on proxy bodies after update.
    virtual void PrintMeshProxiesUpdateData() = 0;
    /// Print information on contact forces acting on proxy bodies.
    virtual void PrintMeshProxiesContactData() = 0;

    // --- Virtual methods for a rigid tire

    /// Create proxy body for a rigid tire mesh.
    /// Use information in the m_mesh_data struct (vertex positions expressed in local frame).
    virtual void CreateWheelProxy() = 0;
    /// Update the state of the wheel proxy body for a rigid tire.
    /// Use information in the m_wheel_state struct (popse and velocities expressed in absolute frame).
    virtual void UpdateWheelProxy() = 0;
    /// Collect cumulative contact force and torque on the wheel proxy body.
    /// Load contact forces (expressed in absolute frame) into the m_wheel_contact struct.
    virtual void GetForceWheelProxy() = 0;
    /// Print information on wheel proxy body after update.
    virtual void PrintWheelProxyUpdateData() = 0;
    /// Print information on contact forces acting on the wheel proxy body.
    virtual void PrintWheelProxyContactData() = 0;

    /// Output terrain data at the specified frame (called once per integration step).
    virtual void OutputTerrainData(int frame) = 0;

    /// Perform any additional operations after the data exchange and synchronization with the rig node.
    virtual void OnSynchronize(int step_number, double time) {}

    /// Perform any additional operations after advancing the state of the terrain node.
    /// For example, render the terrain simulation.
    virtual void OnAdvance(double step_size) {}

  private:
    void SynchronizeRigidTire(int step_number, double time);
    void SynchronizeFlexibleTire(int step_number, double time);
};

}  // end namespace vehicle
}  // end namespace chrono

#endif
