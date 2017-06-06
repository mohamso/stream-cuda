# STREAM CUDA
I have decided to make my CUDA implementation of the famous STREAM benchmark [suite](https://www.cs.virginia.edu/stream/)
publicly available. The code was first implemented during my PhD in December 2013 and updated a year later.

### Compile and Build ###
The code can be built like this:

    cd stream-cuda
    mkdir build
    cd build
    cmake ..
    make -j4
    ../bin/stream-cuda <number-of-iterations> <block-size>

The default array size is 0.5 GB but this can be changed by modifying the code.
The two input to the code is number of iterations and block size.
The recommended block size for Kepler is 192 and 64 for Pascal.

# STREAM Triad Benchmark Results
Below is a collection of STREAM Triad results from different CUDA devices.
These results are useful for example in connection with performance modeling.
All experiments conducted with ECC turned on.

| Machine        | Device           | Result (GB/s)  |
| ------------- |:-------------:| -----:|
Abel     | NVIDIA Tesla K20x     | 180.45 |
EPIC     | NVIDIA Tesla P100     | 557.23 |
Stampede | NVIDIA Tesla K20m     | 151.13 |
Taurus   | NVIDIA Tesla K20xm    | 181.60 |
Taurus   | NVIDIA Tesla K80      | 175.62 |
Wilkes   | NVIDIA Tesla K20c     | 151.02 |

### Bugs
Please report bugs and issues by creating pull requests.
