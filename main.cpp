#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <random>
#include <Eigen/Dense>
#include <chrono>

// Apparently my system stores LSB first. So this makes it little-endian.
// Data is big-endian, so bytes need to be reversed to get integers.

std::string endianness_test() {
    uint16_t word = 1;
    uint8_t * first_byte = (uint8_t*) &word;
    return (*first_byte == 1) ? "LE" : "BE";
}

uint32_t reverse_bytes(uint32_t num) {
    uint8_t c1, c2, c3, c4;

    c1 = num & 255;
    c2 = (num >> 8) & 255;
    c3 = (num >> 16) & 255;
    c4 = (num >> 24) & 255;

    return uint32_t(c1 << 24) + uint32_t(c2 << 16) + uint32_t(c3 << 8) + uint32_t(c4);
}

std::vector<std::vector<double>> flatten(std::vector<std::vector<std::vector<double>>>& images) {
    int m = images.size();
    int rows = images[0].size();
    int cols = images[0][0].size();

    std::vector<std::vector<double>> flatImages(m, std::vector<double>(rows*cols));

    for (int i = 0; i < m; i++) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                flatImages[i][row * cols + col] = images[i][row][col];
            }
        }
    }
    return flatImages;
}

Eigen::MatrixXd read_file(int data_size) {
    std::ifstream ifs("MNIST_ORG/train-images.idx3-ubyte", std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("File cannot be opened.");
    }

    int32_t magic_number;
    int32_t number_of_images;
    int32_t rows;
    int32_t cols;

    ifs.read((char*)&magic_number, 4);
    ifs.read((char*)&number_of_images, 4);
    ifs.read((char*)&rows, 4);
    ifs.read((char*)&cols, 4);

    magic_number = reverse_bytes(magic_number);
    number_of_images = reverse_bytes(number_of_images);
    rows = reverse_bytes(rows);
    cols = reverse_bytes(cols);

    std::cout << "Magic number: " << magic_number << std::endl;
    std::cout << "Number of items: " << number_of_images << std::endl;
    std::cout << "Rows: " << rows << std::endl;
    std::cout << "Cols: " << cols << std::endl;

    int image_size = rows * cols;

    Eigen::MatrixXd images(image_size, data_size);

    if (data_size > number_of_images) throw std::runtime_error("Can't read that much data.");

    for (int i = 0; i < data_size; i++) {
        std::vector<uint8_t> buffer(image_size);
        ifs.read((char*)buffer.data(), image_size);
        for (int j = 0; j < image_size; j++) {
            images(j, i) = (double) buffer[j] / 255.0;
        }
    }

    ifs.close();
    return images;
}

void initialize_params(std::vector<Eigen::MatrixXd>& weights,
                       std::vector<Eigen::VectorXd>& biases,
                       std::vector<int>& layer_sizes) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);

    for (int i = 1; i < layer_sizes.size(); i++) {
        int rows = layer_sizes[i];
        int cols = layer_sizes[i-1];

        weights[i-1] = Eigen::MatrixXd(layer_sizes[i], layer_sizes[i-1]);
        biases[i-1] = Eigen::VectorXd(layer_sizes[i]);

        for (int row = 0; row < rows; row++) {
            biases[i-1](row) = 0;
            for (int col = 0; col < cols; col++) {
                weights[i-1](row, col) = d(gen);
            }
        }
    }
}

void print_weights(std::vector<std::vector<std::vector<double>>>& weights) {
    for (int i = 0; i < weights.size(); i++) {
        for (int row = 0; row < weights[i].size(); row++) {
            for (int col = 0; col < weights[i][row].size(); col++) {
                std::cout << weights[i][row][col] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void print_biases(std::vector<std::vector<double>>& biases) {
    for (int i = 0; i < biases.size(); i++) {
        for (int j = 0; j < biases[i].size(); j++) {
            std::cout << biases[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void print_matrix(const std::vector<std::vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (const int val: row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<std::vector<double>> matmul(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Matrix dimension mismatch.");
    }

    std::vector<std::vector<double>> C(rowsA, std::vector<double>(colsB, 0));

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

std::vector<std::vector<double>> transpose(std::vector<std::vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<double>> t = std::vector<std::vector<double>>(cols, std::vector<double>(rows));

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            t[col][row] = matrix[row][col];
        }
    }

    return t;
}

std::vector<Eigen::MatrixXd> forward_propogation(const std::vector<Eigen::MatrixXd>& weights, const std::vector<Eigen::VectorXd>& biases, Eigen::MatrixXd& inputs) {
    std::vector<Eigen::MatrixXd> activations(1, inputs);

    // a_n = w_n * a_(n-1)
    // a_n = [n x m]
    // w_n = [k x n]

    for (int i = 0; i < weights.size(); i++) {
        Eigen::MatrixXd z = weights[i] * activations[i];
        Eigen::MatrixXd a = z.unaryExpr([](double x) {return std::max(0.0, x);});
        activations.push_back(a);
    }

    return activations;
}

int main() {

    auto start = std::chrono::high_resolution_clock::now();

    try {
        Eigen::MatrixXd images = read_file(1); // 784 x 60000

        std::vector<int> layer_sizes = {784, 16, 16, 10};
        std::vector<Eigen::MatrixXd> weights(layer_sizes.size()-1);
        std::vector<Eigen::VectorXd> biases(layer_sizes.size()-1);
        initialize_params(weights, biases, layer_sizes);

        std::vector<Eigen::MatrixXd> activations = forward_propogation(weights, biases, images);

        Eigen::MatrixXd layer = activations[3];
        std::cout << layer << std::endl;

    } catch(std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Duration: " << duration.count() << " ms" << std::endl;

    return 0;    
}