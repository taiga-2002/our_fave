from flask import Flask, request, jsonify
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod

app = Flask(__name__)

@app.route('/submit-survey', methods=['POST'])
def submit_survey():
    data = request.get_json()
    question1 = data.get('question1')
    question2 = data.get('question2')

    # ここで量子アニーリングによる組合せ最適化を行います
    result = quantum_annealing(question1, question2)

    return jsonify({'result': result})

def quantum_annealing(question1, question2):
    # ここでは例として単純な組合せ最適化問題を解きます
    sampler = EmbeddingComposite(DWaveSampler())

    # サンプルQUBO問題 (実際の問題に応じて変更)
    Q = {(0, 0): 1, (1, 1): 1, (0, 1): -2}

    # 量子アニーリングの実行
    sampleset = sampler.sample_qubo(Q, num_reads=10)
    best_sample = sampleset.first.sample

    # 結果を整形して返す
    return f'Best sample: {best_sample}'

if __name__ == '__main__':
    app.run(debug=True)
