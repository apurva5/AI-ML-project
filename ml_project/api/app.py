from flask import Flask, request, jsonify
from model.train import train_model,load_data
from model.predict import predict_solution


app = Flask(__name__)

model, vectorizer, packaging_encoder = train_model()
df = load_data()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    problem_description = data.get('problem_description')
    packaging_type = data.get('packaging_type')

    if predict_solution(model, vectorizer, packaging_encoder, problem_description, packaging_type):
        matching_solution = df.loc[df['problem_description'].str.lower() == problem_description.lower()].iloc[0]
        response = {
            "technology": matching_solution['technology'],
            "gitlab_solution": matching_solution['gitlab_solution'],
            "git": matching_solution['git'],
            "packaging_type": matching_solution['packaging_type']
        }
    else:
        response = None

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
