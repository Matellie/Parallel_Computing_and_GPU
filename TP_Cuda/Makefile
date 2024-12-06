cuda_part_1: cuda_part_1/tp_cuda_pi_1.cu cuda_part_1/tp_cuda_pi_2.cu cuda_part_1/tp_cuda_pi_3.cu
	nvcc cuda_part_1/tp_cuda_pi_1.cu -o pi_1 -O3 -arch=sm_67
	nvcc cuda_part_1/tp_cuda_pi_2.cu -o pi_2 -O3 -arch=sm_67
	nvcc cuda_part_1/tp_cuda_pi_3.cu -o pi_3 -O3 -arch=sm_67

cuda_part_2: cuda_part_2/tp_cuda_vector_1.cu cuda_part_2/tp_cuda_vector_2.cu cuda_part_2/tp_cuda_vector_3.cu
	nvcc cuda_part_2/tp_cuda_vector_1.cu -o matvect_1 -O3 -arch=sm_67
	nvcc cuda_part_2/tp_cuda_vector_2.cu -o matvect_2 -O3 -arch=sm_67
	nvcc cuda_part_2/tp_cuda_vector_3.cu -o matvect_3 -O3 -arch=sm_67

cuda_part_3: cuda_part_3/tp_cuda_matrix_1.cu cuda_part_3/tp_cuda_matrix_2.cu
	nvcc cuda_part_3/tp_cuda_matrix_1.cu -o matmul_1 -O3 -arch=sm_67
	nvcc cuda_part_3/tp_cuda_matrix_2.cu -o matmul_2 -O3 -arch=sm_67
