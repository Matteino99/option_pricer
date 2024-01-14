# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 21:18:23 2024

@author: matte
"""

# Flask Application (app.py)
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('indexsemplified.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    # Simplified version, just displaying the received data
    data = request.form.to_dict()
    return render_template('indexsemplified.html', data=data)

if __name__ == '__main__':
    app.run(debug=False, port=8002)
