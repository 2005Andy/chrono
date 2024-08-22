#include "chrono_fsi/utils/ChUtilsMeshToPoint.h"
#include <iostream>

int main(){
    std::string objFilePath = "/mnt/d/D_ship/ship/ship.obj";
    std::string outputTxtFilePath = "/mnt/d/D_ship/ship/chrono.txt";

    try {
        chrono::fsi::utils::MeshToPoint(1, objFilePath, outputTxtFilePath, 3, 8.0);
        std::cout << "Point cloud successfully generated and saved to " << outputTxtFilePath << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}