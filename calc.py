import numpy as np
import matplotlib.pyplot as plt
from openjij import SQASampler
from pyqubo import Array, Placeholder, Constraint
import openjij as oj
import pandas as pd
import japanize_matplotlib
from dwave.system import DWaveSampler, EmbeddingComposite

# パラメータの設定
genre = 5
choice = 5
num_mem = 10  # num_matrices =10 に対応するように設定
Input = np.zeros((genre, choice), dtype=int)  # genre x choice のゼロ行列を作成

# 各ジャンルにランダムに1を配置
for i in range(genre):
    col = np.random.randint(0, choice)  # 0からchoice-1のランダムな列を選択
    Input[i, col] = 1

num_matrices = 10  # 作成する行列の数

# 複数のxi行列を格納するリスト
matrices = []

# 10個のランダム行列を作成
for _ in range(num_matrices):
    matrix = np.zeros((genre, choice), dtype=int)
    # 各ジャンルにランダムに1を配置
    for i in range(genre):
        col = np.random.randint(0, choice)  # 0からchoice-1のランダムな列を選択
        matrix[i, col] = 1
    matrices.append(matrix)

# 生成した行列を表示（必要に応じてコメントを外してください）
# for idx, matrix in enumerate(matrices):
#     print(f"Matrix {idx + 1}:")
#     print(matrix)
#     print()  # 行列間のスペース

# 類似度行列の初期化
similarity_matrix = np.zeros((num_matrices, num_matrices))

# 類似度を計算 (内積のトレースを使う)
for i in range(num_matrices):
    for j in range(num_matrices):
        # 内積を計算してそのトレースを求める
        similarity_matrix[i, j] = np.trace(np.dot(matrices[i], matrices[j].T))

# 類似度行列の正規化関数
def normalize_matrix(matrix):
    # 最小値と最大値を取得
    matrix_min = np.min(matrix)
    matrix_max = np.max(matrix)
    
    # 最大値が最小値と等しい場合は、すべての要素が同じ値であるため、0で埋めます
    if matrix_max == matrix_min:
        return np.zeros(matrix.shape)
    
    # 行列の正規化
    normalized_matrix = (matrix - matrix_min) / (matrix_max - matrix_min)
    
    return normalized_matrix

# similarity_matrixの正規化
normalized_similarity_matrix = normalize_matrix(similarity_matrix)

# QUBOを作成する関数
def create_qubo(input_data, similarity_matrix, matrices):
    num_mem = similarity_matrix.shape[0]      # メンバー数（ここではnum_matrices=10）
    genre = input_data.shape[0]               # ジャンル数
    choice = input_data.shape[1]              # 選択肢数

    # バイナリ変数を定義
    x = Array.create('x', shape=(num_mem,), vartype='BINARY')
    
    # lambdas の定義を修正（genre=5 に対応するため、7つの要素に拡張）
    lambdas = [1, 2, 4, 5, 6, 5, 10]  # 必要な長さに拡張
    
    # コスト関数の定義
    cost = lambdas[0] * sum(normalized_similarity_matrix[i, j] * x[i] * x[j] 
                           for i in range(num_mem) for j in range(num_mem))
    
    # 制約の定義
    # 各ジャンルと選択肢に対して、入力データとマトリックスの関係をバイナリ変数で制約
    constraint_expr = 0
    for i in range(genre):
        for j in range(choice):
            # 制約: sum_k (Matrices[k][i, j] * x[k]) ≈ input_data[i, j]
            # これをペナルティとして追加
            constraint_expr += lambdas[i+1] * (sum(matrices[k][i, j] * x[k] for k in range(num_mem)) - input_data[i, j])**2
    
    # 新しい制約式を定義
    constraint_expr1 = lambdas[genre+1] * (sum(x) - 6)**2  # ペナルティ項
    
    # コスト関数に制約を追加
    H = cost + Constraint(constraint_expr, label="constraint") + Constraint(constraint_expr1, label="constraint1")
    
    # モデルのコンパイル
    model = H.compile()
    
    # QUBOに変換
    qubo, offset = model.to_qubo()
    return qubo, offset

# QUBOの作成
qubo, offset = create_qubo(Input, normalized_similarity_matrix, matrices)

# QUBOを表示（必要に応じてコメントを外してください）
# print("QUBO:", qubo)
# print("Offset:", offset)

# サンプラーの設定とQUBOの解決
sampler = oj.SASampler()
response = sampler.sample_qubo(qubo, num_reads=100)
solution = response.first.sample

# 結果の表示
print("最適解:", solution)
print("オフセット:", response.first.energy + offset)

# 必要に応じて、他の解も表示
# for idx, sample in enumerate(response.samples()):
#     print(f"Sample {idx + 1}: {sample} Energy: {sample.energy + offset}")

