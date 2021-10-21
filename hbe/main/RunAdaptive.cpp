/*
 *  Adaptive Sampling:
 *     Run adaptive sampling given 1) dataset 2) epsilon 3) RS or HBE
 *
 *
 *  Example usage:
 *      /hbe conf/shuttle.cfg gaussian 0.2 true
 *          => Run adaptive sampling with RS, with eps=0.2
 *
 *      ./hbe conf/shuttle.cfg gaussian 0.9
 *          => Run adaptive sampling with HBE, with eps=0.9
 *
 */

#include <chrono>
#include "../alg/RS.h"
#include "../alg/AdaptiveRS.h"
#include "../alg/AdaptiveHBE.h"
#include "../utils/DataIngest.h"
#include "parseConfig.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    char *scope = argv[2];
    double eps = atof(argv[3]);
    bool random = (argc > 4);

    std::cout << argc << std::endl;

    parseConfig cfg(argv[1], scope);
    DataIngest data(cfg, true);

    shared_ptr<AdaptiveEstimator> est;
    std::cout << "eps = " << eps << std::endl;
    if (random) {
        std::cout << "RS" << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        est = make_shared<AdaptiveRS>(data.X_ptr, data.kernel, data.tau, eps);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Adaptive Table Init: " <<
                  std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / 1000.0 << std::endl;
    } else {
        std::cout << "HBE" << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        est = make_shared<AdaptiveHBE>(data.X_ptr, data.kernel, data.tau, eps, true);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Adaptive Table Init: " <<
                  std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / 1000.0 << std::endl;
    }

    est->totalTime = 0;
    vector<double> estimate;
    vector<double> samples;
    vector<double> times;

    for (int j = 0; j < data.M; j++) {
        int idx = j * 2;
        VectorXd q = data.X_ptr->row(j);
        if (data.hasQuery != 0) {
            q = data.Y_ptr->row(j);
        } else {
            if (!data.sequential) {
                q = data.X_ptr->row(data.exact[idx + 1]);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        vector<double> estimates = est->query(q);
        auto t2 = std::chrono::high_resolution_clock::now();
        estimate.push_back(estimates[0]);
        samples.push_back(estimates[1]);
        times.push_back((t2-t1).count() / 1e6);
    }

    for (int i = 0; i < data.M; i++) {
        std::cout << "RESULT id=" << i << " est=" << estimate[i] << " samples=" << samples[i] << " time=" << times[i] << std::endl;
    }

    //std::cout << "Sampling total time: " << est->totalTime / 1e9 << std::endl;
    //std::cout << "Average Samples: " << results[1] / data.M << std::endl;
    //std::cout << "Relative Error: " << results[0] / data.M << std::endl;
}
