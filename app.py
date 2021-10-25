from flask import Flask, render_template, request
app = Flask(__name__)
from disease import testData
@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/sub', methods= ["POST"])
def submit():
    if request.method == 'POST':
        input_dict = request.form
        input_data = tuple(input_dict.values())
        result = testData(input_data)
    return render_template('sub.html',n = result)

if __name__ == '__main__':
    app.run(debug=True)

