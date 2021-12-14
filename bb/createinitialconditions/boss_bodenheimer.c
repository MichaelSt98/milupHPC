/* simple c-routine to build input file for miluphcuda */
/* gravitating sphere */
/* see Boss & Bodenheimer 1979 for reference */
// Christoph Schaefer, May 2015
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TRUE (!0)
#define FALSE 0


/* SI units */
#define RMAX 3.2e14//3.2e14
#define MASS 1.9891e30
#define OMEGAZ 1.6e-12


int main(int argc, char *argv[])
{
    double e, f, g;
    double vx, vy;
    double x1, x2, x3;
    double tmp, tmp1, tmp2, tmp3;
    int draw = FALSE;
    double rmax = RMAX * 1.01;
    double delta = 1.7e13; //3.015e12;
    int n = 0;
    double m = 0;
    double deltam = 0;
    FILE *output;
    double omega = OMEGAZ;


    x1 = x2 = x3 = 0.0;

    e = f = g = -rmax;

    while (e < rmax) {
	f = -rmax;
	g = -rmax;
	while (f < rmax) {
	    g = -rmax;
	    while (g < rmax) {
		tmp1 = (e - x1) * (e - x1);
		tmp2 = (f - x2) * (f - x2);
		tmp3 = (g - x3) * (g - x3);
		tmp = tmp1 + tmp2 + tmp3;
		tmp = sqrt(tmp);
		if (tmp < RMAX)
		    draw = TRUE;
		if (draw) {
		    n++;
		    draw = FALSE;
		}
		g += delta;
	    }
	    f += delta;
	}
	e += delta;
    }

    e = f = g = -rmax;
    m = MASS / (1.0 * n);
    fprintf(stdout, "max radius: %e\n", RMAX);
    fprintf(stdout, "omega z: %e\n", OMEGAZ);
    fprintf(stdout, "delta: %e\n", delta);
    fprintf(stdout, "number of particles: %d particles\nmass of one particle: %0.5f = %e\n",
	    n, m, m);

    output = fopen("gas.output", "w");

    while (e < rmax) {
	f = -rmax;
	g = -rmax;
	while (f < rmax) {
	    g = -rmax;
	    while (g < rmax) {
		tmp1 = (e - x1) * (e - x1);
		tmp2 = (f - x2) * (f - x2);
		tmp3 = (g - x3) * (g - x3);
		tmp = tmp1 + tmp2 + tmp3;
		tmp = sqrt(tmp);
		if (tmp < RMAX)
		    draw = TRUE;
		if (draw) {
		    n++;
		    // x y vx vy m  mt
		    vx = -omega * f;
		    vy = omega * e;
		    deltam = m * 0.5 * cos(2 * atan2(f, e));
		    fprintf(output, "%e %e %e %e %e 0.0 %e 0\n", e, f, g,
			    vx, vy, m + deltam);
		    draw = FALSE;
		}
		g += delta;
	    }
	    f += delta;
	}
	e += delta;
    }

    fclose(output);

    return 0;
}
