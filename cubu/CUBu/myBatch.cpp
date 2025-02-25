#include <GL/glew.h>
#include <cuda_gl_interop.h>  
#include <iostream>
#include <string>

#include "include/cpubundling.h"
#include "include/gdrawing.h"

using namespace std;

int main(int argc, char** argv)
{
    // ----------------------------------------------------------------------
    // 1. Parse command-line arguments
    // ----------------------------------------------------------------------
    string graphfile;
    int fboSize = 512;         // default
    int maxEdges = 0;          // 0 means “use all”
    bool onlyEndpoints = false;
    bool gpu_bundling = true; // or true if you prefer GPU

    for (int i = 1; i < argc; ++i)
    {
        string opt = argv[i];
        if (opt == "-f" && i + 1 < argc) {
            graphfile = argv[++i];
        }
        else if (opt == "-i" && i + 1 < argc) {
            fboSize = atoi(argv[++i]);
        }
        else if (opt == "-e") {
            onlyEndpoints = true;
        }
        else if (opt == "-n" && i + 1 < argc) {
            maxEdges = atoi(argv[++i]);
        }
        else if (opt == "--gpu") {
            gpu_bundling = true;
        }
        // Add any other custom parameters as needed
    }

    if (graphfile.empty()) {
        cerr << "Usage: " << argv[0] 
             << " -f <filename> [-i <size>] [-e] [-n <max_edges>] [--gpu]" << endl;
        return 1;
    }

    // ----------------------------------------------------------------------
    // 2. Create and load GraphDrawing structures
    // ----------------------------------------------------------------------
    GraphDrawing* gdrawing_orig  = new GraphDrawing();
    GraphDrawing* gdrawing_bund  = new GraphDrawing();
    GraphDrawing* gdrawing_final = new GraphDrawing();

    bool ok = gdrawing_orig->readTrails(graphfile.c_str(), onlyEndpoints, maxEdges);
    if (!ok) {
        cerr << "Error: cannot open input file " << graphfile << endl;
        return 1;
    }

    gdrawing_orig->normalize(Point2d(fboSize, fboSize), 0.1);

    *gdrawing_bund  = *gdrawing_orig;
    *gdrawing_final = *gdrawing_orig;

    // ----------------------------------------------------------------------
    // 3. Initialize CPUBundling (works for CPU or GPU) 
    // ----------------------------------------------------------------------
    CPUBundling* bund = new CPUBundling(fboSize);

    bund->niter          = 15;
    bund->h              = 32.0f;
    bund->lambda         = 0.2f;
    bund->liter          = 1;
    bund->niter_ms       = 0;
    bund->h_ms           = 32.0f;
    bund->lambda_ends    = 0.5f;
    bund->spl            = 15;
    bund->eps            = 0.5f;
    bund->rep_strength = 1;

    bund->polyline_style = false;
    bund->tangent        = false;
    bund->block_endpoints = true;
    // GPU bundling is controlled outside by gpu_bundling = true
    // Auto update, density estimation, and bundle shape would also be set in relevant code if supported

    if (gpu_bundling) {
        glewInit();

        cudaGLSetGLDevice(0);
    }

    // etc.

    // ----------------------------------------------------------------------
    // 4. Do the bundling
    // ----------------------------------------------------------------------
    *gdrawing_bund = *gdrawing_orig;  

    bund->setInput(gdrawing_bund);
    

    gdrawing_bund->resample(bund->spl);
    gdrawing_bund->saveTrails("unbundled_output.trl", true);

    if (gpu_bundling) {
        cout << "Using GPU bundling..." << endl;
        bund->bundleGPU();
    }
    else {
        bund->bundleCPU();
    }
    
    // ----------------------------------------------------------------------
    // 5. Postprocess if needed
    //    e.g. Relaxation,clamp  displacement, or compute shading/density
    // ----------------------------------------------------------------------
    /*float relaxation        = 0.0f;   // 0 means “fully use the bundled positions”
    float dir_separation    = 0.0f;   // no direction separation
    float max_displacement  = 0.2f;   // clamp fraction
    bool  displacement_is_abs = false; // interpret max_displacement as fraction of edge length?

    *gdrawing_final = *gdrawing_bund;
    gdrawing_final->interpolate(*gdrawing_orig,
                                relaxation,
                                dir_separation,
                                max_displacement,
                                !displacement_is_abs);

    // If you want density or shading (tube shading), do:
    bool shading       = true;  // or false
    int shading_tube   = 0;     // 0=tube style, 1=Phong style
    float shading_radius = 3.0f;
    if (shading) {
        bund->computeDensityShading(gdrawing_final, shading_radius, shading, shading_tube);
    }*/

    // ----------------------------------------------------------------------
    // 6. Save the final result
    // ----------------------------------------------------------------------
    *gdrawing_final = *gdrawing_bund;

    gdrawing_final->saveTrails("bundled_output.trl", true);
    cout << "Saved bundled result to 'bundled_output.trl'." << endl;

    delete gdrawing_bund;
    delete gdrawing_orig;
    delete gdrawing_final;
    delete bund;

    return 0;
}