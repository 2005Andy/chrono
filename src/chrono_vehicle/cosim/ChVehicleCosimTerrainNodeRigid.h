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
// Definition of the rigid TERRAIN NODE (using Chrono::Multicore).
//
// The global reference frame has Z up, X towards the front of the vehicle, and
// Y pointing to the left.
//
// =============================================================================

#ifndef CH_VEHCOSIM__TERRAINNODE_RIGID_H
#define CH_VEHCOSIM__TERRAINNODE_RIGID_H

#include "chrono_multicore/physics/ChSystemMulticore.h"

#include "chrono_vehicle/cosim/ChVehicleCosimTerrainNode.h"

namespace chrono {
namespace vehicle {

class CH_VEHICLE_API ChVehicleCosimTerrainNodeRigid : public ChVehicleCosimTerrainNode {
  public:
    /// Create a rigid terrain subsystem usingn the specified contact method (SMC or NSC).
    ChVehicleCosimTerrainNodeRigid(ChContactMethod method);

    ~ChVehicleCosimTerrainNodeRigid();

    virtual ChSystem* GetSystem() override { return m_system; }

    /// Set the material properties for terrain.
    /// The type of material must be consistent with the contact method (SMC or NSC)
    /// specified at construction. These parameters characterize the material for the container and
    /// (if applicable) the granular material.  Tire contact material is received from the rig node.
    void SetMaterialSurface(const std::shared_ptr<ChMaterialSurface>& mat);

    /// Specify whether contact coefficients are based on material properties (default: true).
    /// Note that this setting is only relevant when using the SMC method.
    void UseMaterialProperties(bool flag);

    /// Set the normal contact force model (default: Hertz)
    /// Note that this setting is only relevant when using the SMC method.
    void SetContactForceModel(ChSystemSMC::ContactForceModel model);

    /// Set proxy contact radius (default: 0.01).
    /// When using a rigid tire mesh, this is a "thickness" for the collision mesh (a non-zero value can improve
    /// robustness of the collision detection algorithm).  When using a flexible tire, this is the radius of the proxy
    /// spheres attached to each FEA mesh node.
    void SetProxyContactRadius(double radius) { m_radius_p = radius; }

  private:
    ChSystemMulticore* m_system;  ///< containing system
    double m_radius_p;            ///< radius for a proxy body

    virtual bool SupportsFlexibleTire() const override { return true; }

    virtual void Construct() override;

    /// Return current total number of contacts.
    virtual int GetNumContacts() const override { return m_system->GetNcontacts(); }

    virtual void CreateMeshProxies() override;
    virtual void UpdateMeshProxies() override;
    virtual void GetForcesMeshProxies() override;
    virtual void PrintMeshProxiesUpdateData() override;
    virtual void PrintMeshProxiesContactData() override;

    virtual void CreateWheelProxy() override;
    virtual void UpdateWheelProxy() override;
    virtual void GetForceWheelProxy() override;
    virtual void PrintWheelProxyUpdateData() override;
    virtual void PrintWheelProxyContactData() override;

    virtual void OnAdvance(double step_size) override;
    virtual void OnRender(double time) override;
};

}  // end namespace vehicle
}  // end namespace chrono

#endif
