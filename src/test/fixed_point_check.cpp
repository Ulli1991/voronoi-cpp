#include "fixed_point_positions.hpp"
#include <cmath>
#include <iostream>

int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);
    int errors=0;
    {
        local_voronoi::FixedPointPositions32<Kokkos::HostSpace> p;
        p.values=decltype(p.values)("fixed_point_contract",5,3);
        const double inputs[5]={0.0,1.0,-0.25,0.5,1.0-0x1p-32};
        for(int i=0;i<5;++i) for(int a=0;a<3;++a) p.set(i,a,inputs[i]);
        const double expected[5]={0,0,0.75,0.5,1.0-0x1p-32};
        for(int i=0;i<5;++i) for(int a=0;a<3;++a)
            if(p(i,a)!=expected[i]) ++errors;
        if(p.displacement(0,2,0)!=0.25 || p.displacement(2,0,0)!=-0.25) ++errors;
        if(p.displacement(3,0,0)!=-0.5 || p.displacement(0,3,0)!=-0.5) ++errors;
        if(p.displacement(0,4,0)!=0x1p-32) ++errors;
        // Quantization is bounded and round-trips every selected exact tick.
        for(int i=0;i<10000;++i){
            const double x=(i+0.314159)/10000.0;
            p.set(0,0,x);
            if(p(0,0)>x || x-p(0,0)>=0x1p-32) ++errors;
            p.values(1,0)=p.values(0,0); p.set(0,0,p(0,0));
            if(p.values(0,0)!=p.values(1,0)) ++errors;
        }
        // Two distinct sites can share a tick; make that limitation explicit.
        p.set(0,0,0.25); p.set(1,0,0.25+0x1p-34);
        if(p.displacement(0,1,0)!=0) ++errors;
    }
    Kokkos::finalize();
    std::cout << "fixed_point_errors=" << errors << '\n';
    return errors ? 1 : 0;
}
