from flask import Flask, request, render_template
import numpy as np
import logging
from quantum_annealing import create_qubo_matrix, optimize_qubo  # quantum_annealing.pyからインポート

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, template_folder='/Users/toedataiga/dev/our_favo/frontend', static_folder='../static')
people = [
    {"name": "Person 1", "image": "person1.jpg"},
    {"name": "Person 2", "image": "person2.jpg"},
    {"name": "Person 3", "image": "person3.jpg"},
    {"name": "Person 4", "image": "person4.jpg"},
    {"name": "Person 5", "image": "person5.jpg"},
]


# アンケートオプションの定義
face_options = ['いぬ', 'ねこ', 'うさぎ', 'きつね', 'たぬき']
personality_options = ['外交型', '感覚型', '思考型', '判断型', '自己主張型']
voice_options = ['かっこいい', 'かわいい', 'セクシー', 'フレッシュ', '伸び']
dance_options = ['しなやか', 'きれい', 'セクシー', 'シンクロダンス', '激しさ']
feeling_options = ['直接会いたい', '見ていたい', '知りたい', 'ストレス発散したい', 'コミュニティを作りたい']

# One-hot エンコード関数
def one_hot_encode(selected_options, all_options):
    return [1 if option in selected_options else 0 for option in all_options]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_survey():
    try:
        # 各カテゴリのデータを取得してone-hotエンコード
        face_preference = one_hot_encode(request.form.getlist('face_preference'), face_options)
        personality = one_hot_encode(request.form.getlist('personality'), personality_options)
        voice = one_hot_encode(request.form.getlist('voice'), voice_options)
        dance = one_hot_encode(request.form.getlist('dance'), dance_options)
        feeling = one_hot_encode(request.form.getlist('feeling'), feeling_options)
                # 5つのone-hotエンコードベクトルを2D配列 (input_data) にスタック
        input_data = np.vstack([face_preference, personality, voice, dance, feeling])
        matrices=[]
        ep=0.00000001
        matrix1=np.array([
    [1., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0.]])
        matrix2=np.array([
    [1., 0., 0., 0., 0.,],
    [1., 0. ,0., 0., 0.],
    [1., 0., 0., 0., 0.],
    [1.,0., 0., 0., 0.],
    [1., 0., 0., 0., 0.]])
        matrix3=np.array([[0., 1., 0., 0., 0.],
 [0., 0., 1., 0. ,0.],
 [0. ,0., 1., 0., 0.],
 [0., 0. ,1. ,0., 0.],
 [0., 0., 0., 0., 1.]])
        matrix4=np.array([[0., 0., 0., 0., 1.],
 [0., 0., 1., 0. ,0.],
 [1. ,0. ,0. ,0. ,0.],
 [0., 0., 0., 0., 1.],
 [0., 1., 0., 0., 0.]])
        matrix5=np.array([[0., 0., 0., 1., 0.],
 [0. ,1., 0. ,0. ,0.],
 [0. ,0., 0., 0., 1.],
 [0. ,0. ,0. ,1. ,0.],
 [1. ,0., 0., 0. ,0.]])
        matrices.append(matrix1)
        matrices.append(matrix2)
        matrices.append(matrix3)
        matrices.append(matrix4)
        matrices.append(matrix5)
        matrices=np.array(matrices)
        # 類似度行列の初期化
        similarity_matrix = np.zeros((input_data.shape[0], input_data.shape[0]))
        mean=np.average(matrices,axis=0)
        if ((matrices)==0).all():
            for j in range(input_data.shape[0]):
                matrices[0,j,0]+=ep
            

            
        
        matrices_new=matrices-mean
        y=np.array([1,2,3,4,5])
        for i1 in range(input_data.shape[0]):
            for i2 in range(input_data.shape[0]):
                for i3 in range(input_data.shape[0]):
                    similarity_matrix[i1,i2]+=y[i3]*(matrices_new[i1,i3,:]/(np.linalg.norm(matrices_new[i1,i3,:])+ep))@matrices_new[i2,i3,:]/(np.linalg.norm(matrices[i2,i3,:]+ep))

        similarity_matrix=similarity_matrix/sum(y)







        # QUBO行列を生成
        qubo, offset = create_qubo_matrix(input_data, similarity_matrix,matrices)

        # QUBO行列を最適化
        best_solution, best_energy = optimize_qubo(qubo)

        # 最適解に基づいて選ばれた人物を抽出
        selected_people = [people[i] for i in range(len(people)) if best_solution[f'x[{i}]'] == 1]

        # QUBOと最適解、選ばれた人物をテンプレートに渡して表示
        return render_template('result.html', qubo=qubo, offset=offset, solution=best_solution, energy=best_energy, selected_people=selected_people)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return f"An error occurred: {e}", 500


if __name__ == '__main__':
    app.run(debug=True)
