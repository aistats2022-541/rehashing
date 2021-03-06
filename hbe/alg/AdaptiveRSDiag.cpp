#include "AdaptiveRSDiag.h"

#include "AdaptiveRS.h"
#include "dataUtils.h"
#include "mathUtils.h"
#include <math.h>

void AdaptiveRSDiag::buildLevels(double tau, double eps) {
    double tmp = log(1/ tau);
    // Effective diameter
    r = sqrt(tmp);
    // # guess
    I = (int) ceil(tmp / LOG2);
    mui = vector<double>(I);
    Mi = vector<int>(I);
    ti = vector<double>(I);
    ki = vector<int>(I);
    wi = vector<double>(I);

    for (int i = 0; i < I; i ++) {
        if (i == 0) {
            mui[i] = (1 - gamma);
        } else {
            mui[i] = (1 - gamma) * mui[i - 1];
        }
        Mi[i] = (int) (ceil(mathUtils::randomRelVar(mui[i]) / eps / eps));

        // Gaussian Kernel
        ti[i] = sqrt(log(1 / mui[i]));
        ki[i] = (int) (3 * ceil(r * ti[i]));
        wi[i] = ki[i] / ti[i] * SQRT_2PI;
//        std::cout << "Level " << i << ", funcs " << ki[i] <<
//            ", target: "<< mui[i] << std::endl;
    }
}

AdaptiveRSDiag::AdaptiveRSDiag(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double tau, double eps) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    rng = std::mt19937_64(rd());

    X = data;
    kernel = k;
    numPoints = data->rows();
    if (kernel->getName() == EXP_STR) {
        double diam = dataUtils::estimateDiameter(data, tau);
        exp_k = dataUtils::getPower(diam, 0.5);
        exp_w = dataUtils::getWidth(exp_k, 0.5);
    }

    lb = tau;
    buildLevels(tau, eps);
}


AdaptiveRSDiag::AdaptiveRSDiag(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, int samples,
                       double tau, double eps) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    rng = std::mt19937_64(rd());

    int n = data->rows();
    if (samples >= n) {
        X = data;
        kernel = k;
        numPoints = data->rows();
        buildLevels(tau, eps);
        return;
    }
    // Sample input matrix
    std::mt19937_64 gen(rd());
    auto indices = mathUtils::pickSet(n, samples, gen);
    X = make_shared<MatrixXd>(samples, data->cols());
    int i = 0;
    for (auto idx: indices) {
        X->row(i) = data->row(idx);
        i ++;
    }
    kernel = k;
    numPoints = samples;

    if (kernel->getName() == EXP_STR) {
        double diam = dataUtils::estimateDiameter(data, tau);
        exp_k = dataUtils::getPower(diam, 0.5);
        exp_w = dataUtils::getWidth(exp_k, 0.5);
    }
    lb = tau;
    buildLevels(tau, eps);
}


std::vector<double> AdaptiveRSDiag::evaluateQuery(VectorXd q, int level) {
    std::uniform_int_distribution<int> distribution(0, numPoints - 1);

    std::vector<double> results = std::vector<double>(2, 0);
    if (L * Mi[level] > numPoints) {
        // Compute Exact
        results[1] = numPoints;
        samples.clear();
        contrib.clear();
        for (int i = 0; i < numPoints; i ++) {
            double d = kernel->density(q, X->row(i));
            samples.push_back(i);
            contrib.push_back(d);
            results[0] += d;
        }
        results[0] /= results[1];
        return results;
    }

    results[1] =  L * Mi[level];
    std::vector<double> Z = std::vector<double>(L, 0);
    for (int i = 0; i < L; i ++) {
        std::vector<int> indices(results[1]);
        for (int j = 0; j < results[1]; j ++) {
            indices[j] = distribution(rng);
        }
        std::sort(indices.begin(), indices.end());
        for (int j = 0; j < results[1]; j ++) {
            int idx = indices[j];
            double d = kernel->density(q, X->row(idx));
            samples.push_back(idx);
            contrib.push_back(d);
            Z[i] += d;
        }
    }

    results[0] = mathUtils::median(Z) / results[1];
    return results;
}

void AdaptiveRSDiag::clearSamples() {
    contrib.clear();
    samples.clear();
}

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

    // initialize original index locations
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
}

void AdaptiveRSDiag::getConstants() {
    // Sort samples by contribution
    vector<int> tmp_samples;
    vector<double> tmp_weights;
    thresh = 1e-10;
    for (auto i: sort_indexes(contrib)) {
        if (contrib[i] < 1) { // Ignore self, otherwise RS cost goes up
            tmp_samples.push_back(samples[i]);
            tmp_weights.push_back(contrib[i]);
        } else {
            break;
        }
    }

    if (tmp_samples.size() > 50000) {
        std::unordered_set<int> elems = mathUtils::pickSet(tmp_samples.size(), 50000, rng);
        vector<int> indices;
        for (int e : elems) { indices.push_back(e); }
        std::sort(indices.begin(), indices.end());

        samples.clear();
        contrib.clear();
        for (size_t i = 0; i < indices.size(); i ++) {
            samples.push_back(tmp_samples[indices[i]]);
            contrib.push_back(tmp_weights[indices[i]]);
        }
    } else {
        samples = tmp_samples;
        contrib = tmp_weights;
    };


    set_start.clear();
    u_global = 0;
    sample_count = samples.size();
    u = vector<double>(4, 0);
    // Start of S4
    set_start.push_back(0);

    for (size_t i = 0; i < contrib.size(); i ++) {
        u_global += contrib[i];
    }
    u_global /= sample_count;

    // Calcuate S4 stats
    size_t i = 0;
    while (set_start.size() == 1) {
        if (contrib[i] < u_global) {
            u[3] += contrib[i];
        } else if (contrib[i - 1] < u_global) {
            // End of S4
            set_start.push_back(i);
        }
        i ++;
    }
    u[3] /= sample_count;
}


void AdaptiveRSDiag::findRings(int strategy, double eps, VectorXd &q, int level) {
    if (strategy == 0 || sample_count < 3) { // Trivial
        lambda = u_global;
        l = u_global;
        set_start.push_back(set_start[1]);
        set_start.push_back(set_start[1]);
    } else if (strategy == 1)  { // Direct
        double min_u = (eps * u_global - u[3]) / 2;
        // Find lambda (S3)
        double s = 0;
        size_t i = set_start[1];
        while (s < min_u && i < contrib.size()) {
            s += contrib[i] / sample_count;
            i ++;
        }
        lambda = contrib[i-1];
        set_start.push_back(i);

        // Find L (S1)
        s = 0;
        i = contrib.size() - 1;
        while (s < min_u && i >= set_start[2]) {
            s += contrib[i] / sample_count;
            i --;
        }
        i = min(contrib.size()-2, i);
        l = contrib[i+1];
        set_start.push_back(i+1);
    }
    set_start.push_back(contrib.size());

    // Gaussian Kernel
    if (kernel->getName() != EXP_STR) {
        exp_w = wi[level];
        exp_k = ki[level];
    }

    // Calculate set stats
    pmins = vector<double>(4, 1);
    pmaxs = vector<double>(4, 0);
    w_mins = vector<double>(4, 1);
    w_maxs = vector<double>(4, 0);
    w_pps.clear();
    w_pp_idx.clear();
    w_ps.clear();
    w_p_idx.clear();
    for (size_t i = 0; i < 4; i ++) {
        u[i] = 0;
        vector<double> wpp;
        vector<double> wp;
        for (size_t j = set_start[3-i]; j < set_start[4-i]; j ++) {
            u[i] += contrib[j];
            if (contrib[j] < thresh) { continue; }
            w_maxs[i] = max(contrib[j], w_maxs[i]);
            w_mins[i] = min(contrib[j], w_mins[i]);

            int idx = samples[j];
            VectorXd delta = X->row(idx) - q.transpose();
            double c = delta.norm() / exp_w;
            double p = mathUtils::collisionProb(c, exp_k);
            pmins[i] = min(pmins[i], p);
            pmaxs[i] = max(pmaxs[i], p);

            double k_i = contrib[j] / p / p;
            double k_j = contrib[j] / p;
            wpp.push_back(k_i);
            wp.push_back(k_j);
        }
        u[i] /= sample_count;

        // Sort
        vector<int> min_idx;
        vector<double> tmp;
        for (auto i: sort_indexes(wp)) {
            min_idx.push_back(i);
            tmp.push_back(wp[i]);
        }
        w_p_idx.push_back(min_idx);
        w_ps.push_back(tmp);

        vector<int> max_idx;
        tmp.clear();
        for (auto i: sort_indexes(wpp)) {
            max_idx.push_back(i);
            tmp.push_back(wpp[i]);
        }
        std::reverse( tmp.begin(), tmp.end() );
        std::reverse( max_idx.begin(), max_idx.end() );
        w_pp_idx.push_back(max_idx);
        w_pps.push_back(tmp);
    }
}


double AdaptiveRSDiag::vbRS() {
    double up = w_maxs[3] * u[3];
    double t2_factor = (set_start[1] - set_start[0]) * 1.0 / sample_count;
    for (int i = 0; i < 3; i ++) {
        if (set_start[3-i] == set_start[4-i]) {continue; }
        for (int j = 0; j < 3; j ++) {
            if (set_start[3-j] == set_start[4-j]) {continue; }
            up += (w_maxs[i] / w_mins[j]) * u[i] * u[j];
        }
        up += t2_factor * w_maxs[i] * u[i];
    }
    return up;
}


double AdaptiveRSDiag::vbHBE() {
//    double sup3 = w_pps[3][0] * pmaxs[3];
    double sup3 = w_ps[3][w_ps[3].size() - 1] ;

    double up = sup3 * u[3];
    double t2_factor = (set_start[1] - set_start[0]) * 1.0 / sample_count;
    double sup1;
    for (int i = 0; i < 3; i ++) {
        if (set_start[3-i] == set_start[4-i]) {continue; }
        for (int j = 0; j < 3; j ++) {
            if (set_start[3-j] == set_start[4-j]) {continue; }
            if (i == j) { // From the same set
                sup1 = 1 / pmins[i];
                for (size_t k = 0; k < 10; k ++) {
                    size_t l = 0;
                    while(w_p_idx[i][l] > w_pp_idx[i][k]) {
                        l ++;
                    }
                    sup1 = max(sup1, w_pps[i][k] / w_ps[i][l]);
                    if ((k == 0 && l == 0) || w_p_idx[i][l] == 0) { break; }
                }
            } else if (i < j) { // From different sets
                sup1 = w_pps[i][0] / w_ps[j][0];
            } else {
                sup1 = w_ps[i][w_ps[i].size() - 1] / w_mins[j];
            }
            up += sup1 * u[i] * u[j];
        }
        up += t2_factor * w_pps[i][0] * pmaxs[3] * u[i];
    }
    return up;
}


int AdaptiveRSDiag::findActualLevel(VectorXd &q, double truth, double eps) {

    vector<int> indices;
    for (size_t i = 0; i < samples.size(); ++i) indices.push_back(i);
    // using built-in random generator:
    std::random_shuffle ( indices.begin(), indices.end() );

    int i = 0;
    std::vector<double> Z = std::vector<double>(L, 0);
    while (i < I) {
        std::vector<double> results = evaluateQuery(q, i);
        if (fabs(results[0] - truth) / truth < eps) {
            return i;
        }
        i ++;
    }
    return I - 1;
}
