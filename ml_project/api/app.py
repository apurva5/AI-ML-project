from flask import Flask, request, jsonify
from model.train import train_model,load_data
from model.predict import predict_solution


app = Flask(__name__)

model, vectorizer, packaging_encoder = train_model()
df = load_data()

@app.route('/predict', methods=['POST'])
def predict():
    response = {}  # Initialize response variable
    try:
        data = request.json
        problem_description = data.get('problem_description')
        packaging_type = data.get('packaging_type')

        if not problem_description or not packaging_type:
            return jsonify({"error": "Missing required fields"}), 400

        # Perform the prediction
        result = predict_solution(model, vectorizer, packaging_encoder, problem_description, packaging_type)
        print(f"Prediction result for '{problem_description}' and '{packaging_type}': {result}")

        if result:
            # Filter DataFrame to find matching solution
            matching_solution_df = df.loc[df['problem_description'].str.lower() == problem_description.lower()]
            print(f"Matching solutions DataFrame: {matching_solution_df}")
            if matching_solution_df.empty:
                return jsonify({"message": "Matching problem description not found in the dataset"}), 404

            # Safely access the first match
            matching_solution = matching_solution_df.iloc[0]

            response = {
                "status": "success",
                "message": "Matching solution found",
                "data": {
                    "problem_description": problem_description,
                    "technology": matching_solution['technology'],
                    "gitlab_solution": matching_solution['gitlab_solution'],
                    "git": matching_solution['git'],
                    "packaging_type": matching_solution['packaging_type']
                }
            }
        else:
            response = {"message": "No matching solution found"}

        return jsonify(response),200

    except Exception as e:
        print(f"Exception occurred: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
