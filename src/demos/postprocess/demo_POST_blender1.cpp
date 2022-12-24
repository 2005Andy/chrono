// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Alessandro Tasora
// =============================================================================
//
// Demo code about using the assets system to create shapes that can be shown in
// the 3D postprocessing, by using Blender.
//
// - The chrono_import.py add-on must be installed in Blender add-ons.
// - run this exe to generate files in ./DEMO_OUTPUT/BLENDER_ directory
// - Use the Blender menu File/Import/Chrono Import to load the assets.py file saved by this exe.
//
// =============================================================================

#include "chrono/assets/ChBoxShape.h"
#include "chrono/assets/ChCamera.h"
#include "chrono/assets/ChCylinderShape.h"
#include "chrono/assets/ChObjFileShape.h"
#include "chrono/assets/ChSphereShape.h"
#include "chrono/assets/ChTexture.h"
#include "chrono/physics/ChParticleCloud.h"
#include "chrono/physics/ChSystemNSC.h"

#include "chrono_postprocess/ChBlender.h"

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace postprocess;

int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    // Create a Chrono::Engine physical system
    ChSystemNSC sys;

    // Create an exporter to Blender
    ChBlender blender_exporter = ChBlender(&sys);

    // Set the path where it will save all files, a directory will be created if not
    // existing
    blender_exporter.SetBasePath(GetChronoOutputPath() + "BLENDER_1");

    // Optional: change the default naming of the generated files:
    // blender_exporter.SetOutputDataFilebase("state");
    // blender_exporter.SetPictureFilebase("picture");

    // --Optional: add further commands
    //   Remember to use \ at the end of each line for a multiple-line string. Pay attention to tabs in Python.
    blender_exporter.SetCustomBlenderCommandsScript(
"\
# Test \
");

 
    /* Start example */
    /// [Example 1]

    // Create a ChBody, and attach some 'assets'
    // that define 3D shapes. These shapes can be shown
    // by Irrlicht or POV postprocessing, etc...
    // Note: these assets are independent from collision shapes!

    // Create a rigid body as usual, and add it
    // to the physical system:
    auto floor = chrono_types::make_shared<ChBody>();
    floor->SetBodyFixed(true);

    // Define a collision shape
    auto floor_mat = chrono_types::make_shared<ChMaterialSurfaceNSC>();

    floor->GetCollisionModel()->ClearModel();
    floor->GetCollisionModel()->AddBox(floor_mat, 10, 0.5, 10, ChVector<>(0, -1, 0));
    floor->GetCollisionModel()->BuildModel();
    floor->SetCollide(true);

    // Add body to system
    sys.Add(floor);

    // ==Asset== attach a 'box' shape.
    auto boxfloor = chrono_types::make_shared<ChBoxShape>();
    boxfloor->GetBoxGeometry().Size = ChVector<>(10, 0.5, 10);
    boxfloor->SetColor(ChColor(0.3f, 0.3f, 0.6f));
    floor->AddVisualShape(boxfloor, ChFrame<>(ChVector<>(0, -1, 0)));

    /// [Example 1]
    /* End example */

    /* Start example */
    /// [Example 2]

    // Textures, colors, asset levels with transformations.

    // Create the rigid body as usual (this won't move, it is only for visualization tests)
    auto body = chrono_types::make_shared<ChBody>();
    body->SetBodyFixed(true);
    sys.Add(body);

    // ==Asset== Attach a 'sphere' shape
    auto sphere = chrono_types::make_shared<ChSphereShape>();
    sphere->GetSphereGeometry().rad = 0.5;
    body->AddVisualShape(sphere, ChFrame<>(ChVector<>(-1, 0, 0)));

    // ==Asset== Attach also a 'box' shape
    auto mbox = chrono_types::make_shared<ChBoxShape>();
    mbox->GetBoxGeometry().Size = ChVector<>(0.2, 0.5, 0.1);
    body->AddVisualShape(mbox, ChFrame<>(ChVector<>(1, 0, 0)));

    // ==Asset== Attach also a 'cylinder' shape
    auto cyl = chrono_types::make_shared<ChCylinderShape>();
    cyl->GetCylinderGeometry().p1 = ChVector<>(2, -0.2, 0);
    cyl->GetCylinderGeometry().p2 = ChVector<>(2.2, 0.5, 0);
    cyl->GetCylinderGeometry().rad = 0.3;
    body->AddVisualShape(cyl);

    // ==Asset== Attach a 'Wavefront mesh' asset, referencing a .obj file:
    auto objmesh = chrono_types::make_shared<ChObjFileShape>();
    objmesh->SetFilename(GetChronoDataFile("models/forklift/body.obj"));
    objmesh->SetTexture(GetChronoDataFile("textures/bluewhite.png"));
    body->AddVisualShape(objmesh, ChFrame<>(ChVector<>(0, 0, 2)));

    // ==Asset== Attach an array of boxes, each rotated to make a spiral
    for (int j = 0; j < 20; j++) {
        auto smallbox = chrono_types::make_shared<ChBoxShape>();
        smallbox->GetBoxGeometry().Size = ChVector<>(0.1, 0.1, 0.01);
        smallbox->SetColor(ChColor(j * 0.05f, 1 - j * 0.05f, 0.0f));
        ChMatrix33<> rot(Q_from_AngY(j * 21 * CH_C_DEG_TO_RAD));
        ChVector<> pos = rot * ChVector<>(0.4, 0, 0) + ChVector<>(0, j * 0.02, 0);
        body->AddVisualShape(smallbox, ChFrame<>(pos, rot));
    }

    // ==Asset== Attach a video camera.
    // Note that a camera can also be attached to a moving object.
    auto camera = chrono_types::make_shared<ChCamera>();
    camera->SetAngle(50);
    camera->SetPosition(ChVector<>(-3, 4, -5));
    camera->SetAimPoint(ChVector<>(0, 1, 0));
    body->AddCamera(camera);

    /// [Example 2]
    /* End example */

    /* Start example */
    /// [Example 3]

    // Create a ChParticleClones cluster, and attach 'assets'
    // that define a single "sample" 3D shape. This will be shown
    // N times in Irrlicht.

    // Create the ChParticleClones, populate it with some random particles,
    // and add it to physical system:
    auto particles = chrono_types::make_shared<ChParticleCloud>();

    // Note: coll. shape, if needed, must be specified before creating particles
    auto particle_mat = chrono_types::make_shared<ChMaterialSurfaceNSC>();

    particles->GetCollisionModel()->ClearModel();
    particles->GetCollisionModel()->AddSphere(particle_mat, 0.05);
    particles->GetCollisionModel()->BuildModel();
    particles->SetCollide(true);

    // Create the random particles
    for (int np = 0; np < 100; ++np)
        particles->AddParticle(ChCoordsys<>(ChVector<>(ChRandom() - 2, 1, ChRandom() - 0.5)));

    // Do not forget to add the particle cluster to the system:
    sys.Add(particles);

    //  ==Asset== Attach a 'sphere' shape asset.. it will be used as a sample
    // shape to display all particles when rendering in 3D!
    auto sphereparticle = chrono_types::make_shared<ChSphereShape>();
    sphereparticle->GetSphereGeometry().rad = 0.05;
    particles->AddVisualShape(sphereparticle);

    /// [Example 3]
    /* End example */

    // Export all existing visual shapes to POV-Ray
    blender_exporter.AddAll();

    // (Optional: tell selectively which physical items you
    // want to render in the folllowing way...)
    //	blender_exporter.RemoveAll();
    //	blender_exporter.Add(floor);
    //	blender_exporter.Add(body);
    //	blender_exporter.Add(particles);



    //
    // RUN THE SIMULATION AND SAVE THE POVray FILES AT EACH FRAME
    //

    // 1) Create the two .pov and .ini files for POV-Ray (this must be done
    //    only once at the beginning of the simulation).

    blender_exporter.ExportScript();

    while (sys.GetChTime() < 1.5) {
        sys.DoStepDynamics(0.01);

        GetLog() << "time= " << sys.GetChTime() << "\n";

        // 2) Create the incremental nnnnn.dat and nnnnn.pov files in output/ dir, that will be load
        //    by the Blender chrono_import add-on.
        blender_exporter.ExportData();
    }

    // That's all! If all worked ok, this python script should
    // have created a  "xxxx.asset.py"  file that you can
    // load in Blender.

    return 0;
}
