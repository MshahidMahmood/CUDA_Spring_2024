#ifndef NEURAL_NET_INC
#define NEURAL_NET_INC
#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <numeric> //std::iota

class GPULayer
{
public:
	GPULayer() :input(), output() {}
	virtual ~GPULayer() {}

	virtual Eigen::MatrixXf forwardPropagation(Eigen::MatrixXf& input) = 0;
	virtual Eigen::MatrixXf backwardPropagation(Eigen::MatrixXf& output, float learningRate) = 0;

protected:
	Eigen::MatrixXf input;
	Eigen::MatrixXf output;
};


__global__ void forwardPropagationKernel(float* input, float* weights, float* bias, float* output, int inputRows, int inputCols, int outputCols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < inputRows) {
        for (int col = 0; col < outputCols; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < inputCols; ++i) {
                sum += input[row * inputCols + i] * weights[i * outputCols + col];
            }
            output[row * outputCols + col] = sum + bias[col];
        }
    }
}

__global__ void backwardPropagationKernel(float* outputError, float* weights, float* inputError, float learningRate, int inputSize, int outputSize, int inputCols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < inputSize) {
        for (int i = 0; i < inputCols; ++i) {
            for (int col = 0; col < outputSize; ++col) {
                inputError[row] += outputError[col] * weights[row * outputSize + col];
                weights[row * outputSize + col + i] -= outputError[col] * learningRate * input[row * inputCols + i];
            }
        }
    }
}

class GPUDenseLayer : public GPULayer
{
public:
	GPUDenseLayer(int inputSize, int  outputSize){
     // alocate memory weights and bias according on dev
     cudaMalloc(&d_weights,inputSize*outputSize*sizeof(float)); //matrix for weights
     cudaMalloc (&d_bias, inputSize*sizeof(float));
     
		//Eigen::MatrixXf::Random returns values from [-1,1] we should scale it to [-0.5,0.5]
		weights = Eigen::MatrixXf::Random(inputSize, outputSize).array() * 0.5f;
		bias = Eigen::MatrixXf::Random(1, outputSize).array() * 0.5f; 
     
    // copy from host to device
     cudaMemcpy(d_weights,weights.data(),inputSize*outputSize*sizeof(float),cudaMemCpyHostToDevice);
     cudaMemcpy(d_bias,bias.data(),inputSize*sizeof(float),cudaMemCpyHostToDevice);
	}

	Eigen::MatrixXf forwardPropagation(Eigen::MatrixXf& input) {
        // Allocate memory for output on the device
        Eigen::MatrixXf output(input.rows(), d_bias.cols());

        // Allocate device memory for output
        float* d_output;
        cudaMalloc(&d_output, output.rows() * output.cols() * sizeof(float));

        // Configure and launch kernel for forward propagation
        int blockSize = 256;
        int numBlocks = (input.rows() + blockSize - 1) / blockSize;
        forwardPropagationKernel<<<numBlocks, blockSize>>>(input.data(), d_weights, d_bias, d_output, input.rows(), input.cols(), output.cols());
        cudaDeviceSyncronize();

        // Copy output from device to host
        cudaMemcpy(output.data(), d_output, output.rows() * output.cols() * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_output);

        return output;
    }

	//computes dE/dW, dE/dB for a given outputError = dE/dY. Returns input_error = dE/dX.

  Eigen::MatrixXf backwardPropagation(Eigen::MatrixXf& outputError, float learningRate) {
    Eigen::MatrixXf inputError = Eigen::MatrixXf::Zero(weights.rows(), weights.cols());

    // Allocate memory for output error on the device
    float* d_outputError;
    cudaMalloc(&d_outputError, outputError.rows() * outputError.cols() * sizeof(float));
    cudaMemcpy(d_outputError, outputError.data(), outputError.rows() * outputError.cols() * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for input error on the device
    float* d_inputError;
    cudaMalloc(&d_inputError, inputError.rows() * inputError.cols() * sizeof(float));

    // Call backwardPropagationKernel
    int blockSize = 256;
    int numBlocks = (inputError.rows() + blockSize - 1) / blockSize;
    backwardPropagationKernel<<<numBlocks, blockSize>>>(d_outputError, d_weights, d_inputError, learningRate, inputError.rows(), inputError.cols());
    cudaDeviceSyncronize();

    // Copy input error from device to host
    cudaMemcpy(inputError.data(), d_inputError, inputError.rows() * inputError.cols() * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_outputError);
    cudaFree(d_inputError);

    return inputError;
}
  ~GPUDenseLayer(){
      cudaFree(d_weights);
      cudaFree(d_bias);
  }

private:
	Eigen::MatrixXf weights;
	Eigen::MatrixXf bias;
};

__global__ void forwardActivationKernel(float* input, float* output, int size, float (*activation)(float)) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = activation(input[index]);
    }
}

__global__ void backwardActivationKernel(float* input, float* outputError, float* inputError, int size, float (*activationPrime)(float)) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        inputError[index] = activationPrime(input[index]) * outputError[index];
    }
}



class GPUActivationLayer : public GPULayer {
public:
    GPUActivationLayer(std::function<float(float)> activation, std::function<float(float)> activationPrime) {
        this->activation = activation;
        this->activationPrime = activationPrime;
    }

    Eigen::MatrixXf forwardPropagation(Eigen::MatrixXf& input) override {
        this->input = input;

        // Allocate device memory for input and output
        float* d_input;
        float* d_output;
        cudaMalloc(&d_input, input.rows() * input.cols() * sizeof(float));
        cudaMalloc(&d_output, input.rows() * input.cols() * sizeof(float));

        // Copy input data from host to device
        cudaMemcpy(d_input, input.data(), input.rows() * input.cols() * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel for forward propagation
        int blockSize = 256;
        int numBlocks = (input.rows() * input.cols() + blockSize - 1) / blockSize;
        forwardActivationKernel<<<numBlocks, blockSize>>>(d_input, d_output, input.rows() * input.cols(), activation);

        // Copy output data from device to host
        Eigen::MatrixXf output(input.rows(), input.cols());
        cudaMemcpy(output.data(), d_output, input.rows() * input.cols() * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);

        this->output = output;
        return output;
    }

    Eigen::MatrixXf backwardPropagation(Eigen::MatrixXf& outputError, float learningRate) override {
        Eigen::MatrixXf inputError = Eigen::MatrixXf::Zero(outputError.rows(), outputError.cols());

        // Allocate device memory for input and output error
        float* d_input;
        float* d_outputError;
        float* d_inputError;
        cudaMalloc(&d_input, outputError.rows() * outputError.cols() * sizeof(float));
        cudaMalloc(&d_outputError, outputError.rows() * outputError.cols() * sizeof(float));
        cudaMalloc(&d_inputError, outputError.rows() * outputError.cols() * sizeof(float));

        // Copy data from host to device
        cudaMemcpy(d_input, input.data(), outputError.rows() * outputError.cols() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_outputError, outputError.data(), outputError.rows() * outputError.cols() * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel for backward propagation
        int blockSize = 256;
        int numBlocks = (outputError.rows() * outputError.cols() + blockSize - 1) / blockSize;
        backwardActivationKernel<<<numBlocks, blockSize>>>(d_input, d_outputError, d_inputError, outputError.rows() * outputError.cols(), activationPrime);

        // Copy input error from device to host
        cudaMemcpy(inputError.data(), d_inputError, outputError.rows() * outputError.cols() * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_outputError);
        cudaFree(d_inputError);

        return inputError;
    }

private:
    std::function<float(float)> activation;
    std::function<float(float)> activationPrime;
};


__global__ void fitKernel(Eigen::MatrixXf x_train, Eigen::MatrixXf y_train, Layer** layers, int* order, int samples, float learningRate, float* errors){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < samples) {
        Eigen::MatrixXf output = x_train.row(order[index]);

        // Forward propagation
        for (int j = 0; j < layers.size(); ++j) {
            output = layers[j]->forwardPropagation(output);
        }

        // Compute loss
        Eigen::MatrixXf y = y_train.row(order[index]);
        errors[index] = loss(y, output);

        // Backward propagation
        Eigen::MatrixXf error = lossPrime(y, output);
        for (int j = layers.size() - 1; j >= 0; --j) {
            error = layers[j]->backwardPropagation(error, learningRate);
        }
    }
}



class GPUNetwork
{
public:
	GPUNetwork() {}
	virtual ~GPUNetwork() {}

	void add(GPULayer* layer)
	{
	  layers.push_back(layer);
	}

	void use(std::function<float(Eigen::MatrixXf&, Eigen::MatrixXf&)> lossF, std::function<Eigen::MatrixXf(Eigen::MatrixXf&, Eigen::MatrixXf&)> lossDer)
	{
		loss = lossF;
		lossPrime = lossDer;
	}
  
  
  std::vector<Eigen::MatrixXf> predict(Eigen::MatrixXf input)
	{
		int samples = input.rows();

		std::vector<Eigen::MatrixXf> result;

		//forward propagation
		for (int j = 0; j < samples; ++j)
		{
			Eigen::MatrixXf output = input.row(j);
			for (GPULayer* layer : layers)
				output = layer->forwardPropagation(output);

			result.push_back(output);
		}

		return result;
	}


	//train the network
	virtual void fit(Eigen::MatrixXf x_train, Eigen::MatrixXf y_train, int epochs, float learningRate)
	{ 
		int samples = x_train.rows();
		std::cout << "Samples: " << samples << std::endl;
		printMatrixSize("x_train", x_train);
		printMatrixSize("y_train", y_train);

    //allocate memory for order
    int* d_order
    cudaMalloc(&d_order,samples * sizeof(int))

		std::vector<int> order(samples);
		std::iota(order.begin(), order.end(), 0);
  
    //alocate memory for xtrain and y train on device
    Eigen::MatrixXf d_x_train, d_y_train;
    cudaMalloc(&d_x_train, x_train.rows() * x_train.cols() * sizeof(float));
    cudaMalloc(&d_y_train, y_train.rows() * y_train.cols() * sizeof(float));
  
    // Copy data from host to device
    cudaMemcpy(d_x_train, x_train.data(), x_train.rows() * x_train.cols() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_train, y_train.data(), y_train.rows() * y_train.cols() * sizeof(float), cudaMemcpyHostToDevice);



		//training loop
		for (int i = 0; i < epochs; ++i)
		{
			float err = 0.0f;
			
			//feed forward
			std::random_shuffle(order.begin(), order.end());

      // Copy order vector from host to device
      cudaMemcpy(d_order, order.data(), samples * sizeof(int), cudaMemcpyHostToDevice);
   
      // Launch kernel for forward and backward propagation
      int blockSize = 256; //chose this
      int numBlocks = (samples + blockSize - 1) / blockSize;
   
      fitKernel<<<numBlocks, blockSize>>>(d_x_train, d_y_train, layers, d_order, samples, learningRate, d_errors);
      cudaDeviceSyncronize();
   
      // Copy error array from device to host
      cudaMemcpy(&err, d_errors, sizeof(float), cudaMemcpyDeviceToHost);

      // Compute average error
      err /= (float)samples;
      std::cout << "Epoch " << (i + 1) << "/" << epochs << " error = " << err << std::endl;
    }

    // Free device memory
    cudaFree(d_order);
    cudaFree(d_errors);
    cudaFree(d_x_train);
    cudaFree(d_y_train);
  }
   


protected:
	std::vector<Layer*> layers;
	std::function<float(Eigen::MatrixXf&, Eigen::MatrixXf&)> loss;
	std::function<Eigen::MatrixXf(Eigen::MatrixXf&, Eigen::MatrixXf&)> lossPrime;
};


#endif
