#include <iostream>
#include <vector>


#include <cxxopts.h>
#include <highfive/H5File.hpp>

#include "../include/H5Renderer.h"
#include "../include/ConfigParser.h"
#include "../include/Logger.h"

structlog LOGCFG = {};

int main(int argc, char *argv[]) {

    /** Reading command line options **/
    cxxopts::Options options("h5renderer",
                             "Create images from HDF5 files.");
    options.add_options()
            ("c,config", "Path to config file", cxxopts::value<std::string>()->default_value("h5renderer.info"))
            ("v,verbose", "More printouts for debugging")
            ("i,input", "Read files from given path (otherwise from config file)", cxxopts::value<std::string>()->default_value("-"))
            ("o,output", "Write result files to given path", cxxopts::value<std::string>()->default_value("./output"))
            ("x,mark", "Mark particles as + instead of points", cxxopts::value<bool>()->default_value("false"))
            ("z,zoom", "Zoom in factor to show more details", cxxopts::value<double>()->default_value("1"))
            ("h,help", "Show this help");

    // read and store options provided
    auto opts = options.parse(argc, argv);

    /** Initialize Logger **/
    LOGCFG.headers = true;
    LOGCFG.level = opts.count("verbose") ? DEBUG : INFO;

    // print help on usage and exit
    if (opts.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    /** Parse config file **/
    ConfigParser confP{ ConfigParser(opts["config"].as<std::string>()) };

    std::string inputFolder = confP.getVal<std::string>("h5folder");
    std::string inputFolderCmdArgument = opts["input"].as<std::string>();
    if (inputFolderCmdArgument.compare(std::string{"-"})) {
        inputFolder = inputFolderCmdArgument;
    }

    bool markParticles = opts["mark"].as<bool>();
    std::cout << "Mark particles: " << markParticles << std::endl;

    /** Initialize H5Renderer from config file **/
    auto renderer { H5Renderer( inputFolder,
                    confP.getVal<double>("systemSize"),
            confP.getVal<int>("imgHeight"),
                    opts["zoom"].as<double>(),
                            markParticles,
            confP.getVal<bool>("processColoring")) };

    /** create the images from data in h5 files **/
    renderer.createImages(opts["output"].as<std::string>());

    return 0;
}
