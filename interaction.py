from flask import Flask, request, jsonify, render_template
from backend_call_func import predict_all_val
from PlotterNN import mirror_mesh, contourPlotter

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    
    # Extract inputs from the received JSON data
    x = data.get('x')
    y = data.get('y')
    z = data.get('z')
    
    # Extract additional inputs if available
    more_inputs = {f'input{i}': data.get(f'input{i}') for i in range(4, 28)}
    
    # Call the function to calculate model output
    result = predict_all_val(float(x), float(y), float(z))
    contourPlotter(float(x), float(y), float(z), result[1], result[2], result[3], result[4], result[5], result[6])

    
    # Return the result as a JSON response
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
