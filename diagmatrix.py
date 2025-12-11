import h5py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import os
import traceback
import time
import glob

# 设置输入输出路径（使用当前工作目录）
input_dir = "./Hsave/"
output_eng_path = "./Eigen/groundeng.txt"
output_vec_path = "./Eigen/eigenvectors.h5"

print("=" * 60)
print("开始Python对角化程序")
print("=" * 60)

# 检查输入文件夹是否存在
if not os.path.exists(input_dir):
    print(f"错误：输入文件夹 '{input_dir}' 不存在！")
    exit(1)

# 自动检测子空间数量
matrix_files = glob.glob(os.path.join(input_dir, "matrix_*.h5"))
if not matrix_files:
    print(f"错误：在 '{input_dir}' 中没有找到矩阵文件！")
    exit(1)

# 从文件名中提取P值并排序
P_values = []
for file in matrix_files:
    filename = os.path.basename(file)
    # 从 "matrix_P.h5" 中提取P值
    P_str = filename[7:-3]  # 去掉 "matrix_" 和 ".h5"
    try:
        P = int(P_str)
        P_values.append(P)
    except ValueError:
        print(f"警告：无法解析文件名 '{filename}' 中的P值")

P_values.sort()
Ns = len(P_values)

print(f"检测到 {Ns} 个子空间矩阵文件")
print(f"P值范围: {P_values}")
print(f"输入目录: {os.path.abspath(input_dir)}")
print(f"输出本征值文件: {os.path.abspath(output_eng_path)}")
print(f"输出本征向量文件: {os.path.abspath(output_vec_path)}")

t0 = time.time()  # 记录开始时间

# 1. 清空或创建输出文件
# 处理eng.txt - 创建空文件
if os.path.exists(output_eng_path):
    print(f"清空现有文件: {output_eng_path}")
    open(output_eng_path, 'w').close()
else:
    print(f"创建新文件: {output_eng_path}")
    # 确保目录存在
    os.makedirs(os.path.dirname(output_eng_path), exist_ok=True)
    open(output_eng_path, 'w').close()

# 处理eigenvectors.h5
if os.path.exists(output_vec_path):
    print(f"删除现有文件: {output_vec_path}")
    os.remove(output_vec_path)

# 确保目录存在
os.makedirs(os.path.dirname(output_vec_path), exist_ok=True)

# 创建新的HDF5文件
with h5py.File(output_vec_path, 'w') as f:
    print(f"创建新文件: {output_vec_path}")

print("\n输出文件已初始化完成，开始处理矩阵...")

# 处理所有矩阵文件
for P in P_values:
    input_path = os.path.join(input_dir, f"matrix_{P}.h5")
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"\n{'='*60}")
        print(f"警告：文件 '{input_path}' 不存在，跳过 P={P}")
        continue
    
    print(f"\n{'='*60}")
    print(f"开始处理 P={P} 的矩阵")
    print(f"文件路径: {input_path}")
    start_time = time.time()
    
    try:
        # 读取COO格式的稀疏矩阵
        with h5py.File(input_path, 'r') as f:
            # 读取行索引、列索引和值
            rows = f['rows'][:]
            cols = f['cols'][:]
            
            # 检查值的数据类型
            values_dset = f['values']
            
            # 处理复数类型
            if values_dset.dtype.names is not None and 'real' in values_dset.dtype.names and 'imag' in values_dset.dtype.names:
                real_part = values_dset['real'][:]
                imag_part = values_dset['imag'][:]
                values = real_part + 1j * imag_part
            else:
                values = values_dset[:]
            
            # 获取矩阵维度
            n = int(max(np.max(rows), np.max(cols))) + 1
            print(f"矩阵维度: {n}x{n}, 非零元素: {len(values)}")
            
            # 重建COO格式稀疏矩阵
            matrix = sp.coo_matrix((values, (rows, cols)), shape=(n, n), dtype=values.dtype)
        
        # 转换为CSR格式以提高计算效率
        matrix_csr = matrix.tocsr()
        print("矩阵已转换为CSR格式")
        
        # 检查矩阵是否对称/Hermitian
        is_real = np.isrealobj(matrix_csr)
        tol = 1e-10  # 对称性容差
        
        if is_real:
            # 实数矩阵 - 检查对称性
            print("检查矩阵对称性...")
            # 使用稀疏矩阵操作避免转换为稠密矩阵
            diff = matrix_csr - matrix_csr.T
            max_diff = abs(diff).max()
            print(f"最大非对称差值: {max_diff}")
            
            if max_diff > tol:
                print("警告：矩阵不对称！使用对称部分 (A + A.T)/2")
                matrix_csr = (matrix_csr + matrix_csr.T) / 2
        else:
            # 复数矩阵 - 检查Hermitian性
            print("检查矩阵Hermitian性...")
            diff = matrix_csr - matrix_csr.getH()
            max_diff = abs(diff).max()
            print(f"最大非Hermitian差值: {max_diff}")
            
            if max_diff > tol:
                print("警告：矩阵不是Hermitian！使用Hermitian部分 (A + A^dagger)/2")
                matrix_csr = (matrix_csr + matrix_csr.getH()) / 2
        
        # 计算最小本征值和本征向量
        print("开始对角化计算...")
        eigenvalues, eigenvectors = eigsh(
            matrix_csr, 
            k=1, 
            which='SA',  # 最小代数特征值
            tol=1e-6,    # 收敛容差
            maxiter=10000,  # 增大迭代次数以确保收敛
            ncv=min(100, n-1)  # 子空间大小
        )
        
        # 提取结果
        min_eigenvalue = eigenvalues[0]
        min_eigenvector = eigenvectors[:, 0]
        
        # 将本征值追加写入eng.txt（使用普通十进制格式）
        with open(output_eng_path, 'a') as eng_file:
            # 使用足够精度的十进制表示（非科学计数法）
            eng_file.write(f"P={P}, E={min_eigenvalue:.16f}\n")
        
        # 将本征向量写入HDF5文件（按子空间分类）
        with h5py.File(output_vec_path, 'a') as vec_file:
            dset_name = f"subspace_{P}"
            vec_file.create_dataset(dset_name, data=min_eigenvector)
        
        elapsed = time.time() - start_time
        print(f"\nP={P} 计算完成！耗时: {elapsed:.2f} 秒")
        print(f"最小本征值: {min_eigenvalue:.16f}")
        print(f"本征向量范数: {np.linalg.norm(min_eigenvector):.6f}")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n处理 P={P} 时发生错误: {str(e)}")
        print(f"耗时: {elapsed:.2f} 秒")
        traceback.print_exc()
        print(f"跳过 P={P}")

print(f"\n{'='*60}")
print("所有矩阵处理完成！")
print(f"本征值结果保存到: {os.path.abspath(output_eng_path)}")
print(f"本征向量结果保存到: {os.path.abspath(output_vec_path)}")
print("="*60)

print(f"总耗时: {time.time() - t0:.2f} 秒, 约 {(time.time() - t0)/60:.2f} 分")  # 输出总耗时

# 最终结果摘要
print("\n最终结果摘要:")
if os.path.exists(output_eng_path):
    with open(output_eng_path, 'r') as f:
        for line in f:
            print(line.strip())

print("\nPython对角化程序执行完成！")