#include "../include/H5Renderer.h"

H5Renderer::H5Renderer(std::string _h5folder, double _systemSize, int _imgHeight, double _zoom, bool _markParticles,
                       bool _processColoring) :
h5folder { _h5folder }, systemSize { _systemSize }, imgHeight { _imgHeight }, zoom { _zoom },
markParticles { _markParticles}, processColoring { _processColoring }, h5files { std::vector<fs::path>() }
{
    // gather files found at h5folder
    fs::path h5path ( h5folder );
    if( !fs::exists(h5path) ){
        Logger(ERROR) << "Bad provided path for 'h5folder/*.h(df)5': '" << h5folder << "' doesn't exist.";
    } else if ( !fs::is_directory(h5path) ){
        Logger(ERROR) << "Bad provided path for 'h5folder/*.(df)h5': '" << h5folder << "' is not a directory.";
    } else {
        Logger(INFO) << "Collecting h5 files from '" << h5folder << "' ...";
        fs::directory_iterator endDirIt; // empty iterator serves as end
        fs::directory_iterator dirIt(h5path);
        while (dirIt != endDirIt){
            if(fs::extension(dirIt->path()) == ".h5" || fs::extension(dirIt->path()) == ".hdf5") {
                // collect h5files in h5folder dir in container
                h5files.push_back(dirIt->path());
                Logger(INFO) << "Found " << dirIt->path().filename();
            }
            ++dirIt;
        }
        Logger(INFO) << "... done.";
    }

    // initialize pixelspace size
    psSize = 2*imgHeight*imgHeight;
}

// public functions
void H5Renderer::createImages(std::string outDir){

    // loop through all found h5 files

    #pragma omp parallel for
    for (std::vector<fs::path>::const_iterator h5PathIt = h5files.begin(); h5PathIt < h5files.end(); h5PathIt++) {

        // initialize data structures
        std::vector<unsigned long> ranges;
        std::vector<unsigned long> k;
        std::vector<std::vector<double>> x; // container for particle positions

        Logger(INFO) << "Reading " << h5PathIt->filename() << " ...";

        Logger(DEBUG) << "    Opening '" << h5PathIt->string() << "' ...";
        // opening file

        #pragma omp critical
        {
            HighFive::File file(h5PathIt->string(), HighFive::File::ReadOnly);
            Logger(DEBUG) << "    file opened.";


            // reading process ranges
            HighFive::DataSet rng = file.getDataSet("/ranges");

            rng.read(ranges);

            // reading particle keys
            HighFive::DataSet key = file.getDataSet("/key");

            key.read(k);

            // reading particle positions
            HighFive::DataSet pos = file.getDataSet("/x");

            pos.read(x);
        }

        Logger(DEBUG) << "    Storing read data to vector<Particle> container ...";
        std::vector<Particle> particles {};

        for (int i = 0; i < x.size(); ++i) {
            particles.push_back(Particle(x[i][0], x[i][1], x[i][2], k[i]));
        }
        Logger(DEBUG) << "    ... done.";

        Logger(INFO) << "... processing data from " << h5PathIt->filename() << " ...";

        Logger(DEBUG) << "  Creating pixel space.";

        ColorRGB *pixelSpace = new ColorRGB[psSize];
        clearPixelSpace(pixelSpace); // drawing black background explicitly

        Logger(DEBUG) << "    Sorting by z-coordinate ...";
        std::sort(particles.rbegin(), particles.rend(), Particle::zComp); // using reverse iterator
        Logger(DEBUG) << "    ... drawing pixels in x-y-plane ...";
        // looping through particles in decreasing z-order
        for (int i = 0; i < particles.size(); ++i) {
            //if (particles[i].z > 0) {
                ColorRGB color = procColor(particles[i].key, ranges);
                particle2PixelXY(particles[i].x, particles[i].y, color, pixelSpace);
                if (markParticles) {
                    for (int ii = -20; ii < 20; ii++) {
                        particle2PixelXY(particles[i].x + ii * (systemSize / 300.), particles[i].y, color, pixelSpace);
                        particle2PixelXY(particles[i].x, particles[i].y + ii * (systemSize / 300.), color, pixelSpace);
                    }
                }
            //}
        }
        Logger(DEBUG) << "    ... done."; // 19 and 34 rho, p, sml, e, material_type, number_of_interactions, soundspeed, proc
        // cs, e, key, m, noi, p, proc, ranges, rho, sml, v

        Logger(DEBUG) << "    Sorting by y-coordinate  ...";
        std::sort(particles.begin(), particles.end(), Particle::yComp);
        Logger(DEBUG) << "    ... drawing pixels in x-z-plane ...";
        // looping through particles in increasing y-order
        for (int i = 0; i < particles.size(); ++i) {
            //if (particles[i].y > 0) {
                ColorRGB color = procColor(particles[i].key, ranges);
                particle2PixelXZ(particles[i].x, particles[i].z, color, pixelSpace);
                if (markParticles) {
                    for (int ii = -20; ii < 20; ii++) {
                        particle2PixelXZ(particles[i].x + ii * (systemSize / 300.), particles[i].z, color, pixelSpace);
                        particle2PixelXZ(particles[i].x, particles[i].z + ii * (systemSize / 300.), color, pixelSpace);
                    }
                }
            //}
        }
        Logger(DEBUG) << "    ... done.";

        std::string outFile = outDir + "/" + h5PathIt->stem().string() + ".ppm";
        Logger(INFO) << "... writing to file '" << outFile << "' ...";
        // writing pixelSpace to png file
        pixelSpace2File(outFile, pixelSpace);

        Logger(DEBUG) << "  Deleting pixel space.";
        delete[] pixelSpace;

        Logger(INFO) << "... done. Results written to '" << outFile << "'.";
    }
}

// private functions
ColorRGB H5Renderer::procColor(unsigned long k, const std::vector<unsigned long> &ranges){
    for(int proc=0; proc < ranges.size()-1; proc++){
        if (ranges[proc] <= k && k < ranges[proc+1]){
            // particle belongs to process proc
            return COLORS[proc];
        }
    }
    return ColorRGB(); // black or white
}

int H5Renderer::procNumber(unsigned long k, const std::vector<unsigned long> &ranges){
    for(int proc=0; proc < ranges.size()-1; proc++){
        if (ranges[proc] <= k && k < ranges[proc+1]){
            // particle belongs to process proc
            return proc;
        }
    }
    return -1; // black
}

void H5Renderer::clearPixelSpace(ColorRGB *pixelSpace){
    for(int px=0; px < psSize; ++px){
        pixelSpace[px] = ColorRGB(); // drawing black background
    }
}

int H5Renderer::pos2pixel(double pos){
    //double zoomBoxRadius = .5*systemSize/zoom;    
    //return abs(pos) > zoomBoxRadius ? -1 : (int)round(imgHeight/2. * (1. + pos/zoomBoxRadius*SCALE2FIT));
    return pos > systemSize/zoom ? -1 : round(imgHeight/2. * (1. + pos/(systemSize/zoom)*SCALE2FIT));
}

void H5Renderer::particle2PixelXY(double x, double y, const ColorRGB &color, ColorRGB *pixelSpace){
    // convert to pixel space
    int xPx = pos2pixel(x);
    int yPx = pos2pixel(y);

    // only draw when not out of zoomed box
    if (xPx >= 0 && yPx >= 0){
        // draw in x-y plane
        pixelSpace[xPx+2*imgHeight*yPx] = color;
    }
}

void H5Renderer::particle2PixelXZ(double x, double z, const ColorRGB &color, ColorRGB *pixelSpace){
    // convert to pixel space
    int xPx = pos2pixel(x);
    int zPx = pos2pixel(z);

    // only draw when not out of zoomed box
    if (xPx >= 0 && zPx >= 0){
        // draw in x-z plane
        pixelSpace[xPx+2*imgHeight*zPx+imgHeight] = color;
    }
}

void H5Renderer::pixelSpace2File(const std::string &outFile, ColorRGB *pixelSpace){
    // using *.ppm
    // https://en.wikipedia.org/wiki/Netpbm#File_formats

    std::ofstream file (outFile, std::ofstream::binary);

    // flatten ColorRGB struct
    char pxData[3*psSize];
    for (int px=0; px < 3*psSize; px+=3){
        pxData[px] = pixelSpace[px/3].r;
        pxData[px+1] = pixelSpace[px/3].g;
        pxData[px+2] = pixelSpace[px/3].b;
    }

    if (file.is_open()){
        file << "P6\n" << 2*imgHeight << " " << imgHeight << "\n" << "255\n";
        file.write(pxData, psSize*3);
        file.close();
    }
}







