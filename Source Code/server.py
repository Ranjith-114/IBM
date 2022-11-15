from flask import Flask, render_template
app = Flask(__name__)
import os
import subprocess
@app.route(r'/')
def index():
  return render_template('index.html')

@app.route('/my-link/')
def my_link():
  print("hello")
  subprocess.call('python final.py', shell=True)
  return "done"

if __name__ == '__main__':
  app.run(debug=True)