// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Arman Pazouki
// =============================================================================
//
// Base class for processing bce forces in fsi system.//
// =============================================================================

#ifndef CH_BCE_CUH_
#define CH_BCE_CUH_

#include "chrono_fsi/ChApiFsi.h"
#include "chrono_fsi/ChFsiGeneral.cuh"
#include "chrono_fsi/ChFsiDataManager.cuh" //for FsiGeneralData

namespace chrono {
namespace fsi {

class CH_FSI_API ChBce : public ChFsiGeneral {
public:
	thrust::device_vector<Real3> velMas_ModifiedBCE;//(numRigidAndBoundaryMarkers);
	thrust::device_vector<Real4> rhoPreMu_ModifiedBCE;//(numRigidAndBoundaryMarkers);


	ChBce(FsiGeneralData* otherFsiGeneralData,
		SimParams* otherParamsH, 
		NumberOfObjects* otherNumObjects);

	~ChBce();

	void ModifyBceVelocity();
	void UpdateRigidMarkersPositionVelocity(
		SphMarkerDataD* sphMarkersD,
		FsiBodiesDataD* fsiBodiesD);

	void Rigid_Forces_Torques(
		SphMarkerDataD* sphMarkersD,
		FsiBodiesDataD* fsiBodiesD);

private:

	FsiGeneralData* fsiGeneralData;
	SphMarkerDataD * sortedSphMarkersD;
	ProximityDataD * markersProximityD;




	thrust::device_vector<Real4> totalSurfaceInteractionRigid4;
	thrust::device_vector<Real3> torqueMarkersD;
	thrust::device_vector<int> dummyIdentify;

	SimParams* paramsH;
	NumberOfObjects* numObjectsH;
};
} // end namespace fsi
} // end namespace chrono

#endif