#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE

    // get the number of batches
    size_t num_batches = m / batch;

    

    for (size_t b = 0; b < num_batches; ++b) {

        // calculate the start and end indices for the current batch
        size_t start = b * batch;
        size_t end = start + batch;

        // X, Y: dependent on start and end

        // allocate memory for logits and gradients
        std::vector<float> logits(batch * k, 0.0f);
        std::vector<float> grad(n * k, 0.0f);

        // compute logits for the current batch
        for (size_t i = start; i < end; ++i) {
            for (size_t j = 0; j < k; ++j) {
                size_t idx = (i - start) * k + j; // row-major index
                logits[idx] = 0.0f;
                for (size_t l = 0; l < n; ++l) {
                    // X: b, n
                    // theta: n, k
                    logits[idx] += X[i * n + l] * theta[l * k + j];
                }
            }
        }

        // apply softmax to logits
        for (size_t i = 0; i < batch; ++i) {
            // float max_logit = *std::max_element(logits.begin() + i * k, logits.begin() + (i + 1) * k);
            float sum_exp = 0.0f;
            for (size_t j = 0; j < k; ++j) {
                // logits[i * k + j] = std::exp(logits[i * k + j] - max_logit);
                logits[i * k + j] = std::exp(logits[i * k + j]);
                // accumulate the sum of exponentials
                sum_exp += logits[i * k + j];
            }
            for (size_t j = 0; j < k; ++j) {
                logits[i * k + j] /= sum_exp;
            }
        }

        for (size_t i = 0; i < batch; ++i) {
            logits[i * k + y[start + i]] -= 1.0f;  // logits becomes (probs - I)
        }

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < k; ++j) {
                for (size_t b_idx = 0; b_idx < batch; ++b_idx) {
                    grad[i * k + j] += X[(start + b_idx) * n + i] * logits[b_idx * k + j];
                }
            }
        }

        // update theta
        for (size_t l = 0; l < n; ++l) {
            for (size_t j = 0; j < k; ++j) {
                theta[l * k + j] -= lr * grad[l * k + j] / batch;
            }
        }
    }



    /// END YOUR CODE
}



/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
