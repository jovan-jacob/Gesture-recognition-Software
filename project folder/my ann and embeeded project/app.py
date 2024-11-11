from flask import Flask, render_template, request, redirect, url_for
import subprocess
import os


app = Flask(__name__)

# Variable to check if the model is created
model_created = False
actions = []

@app.route('/')
def index():
    global model_created
    return render_template('index.html', model_created=model_created)

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        # Get the number of actions entered by the user
        num_actions = int(request.form['actions_count'])
        return render_template('actions_form.html', num_actions=num_actions)
    return render_template('train.html')

@app.route('/submit_actions', methods=['POST'])
def submit_actions():
    # Get the actions names submitted by the user
    global actions
    actions = [request.form.get(f'action_{i}') for i in range(len(request.form))]
    print(f"Actions entered: {actions}")
    return redirect(url_for('instructions'))

@app.route('/instructions', methods=['GET', 'POST'])
def instructions():
    if request.method == 'POST':
        # This will be triggered when the user clicks "Process"
        subprocess.run(['python', 'train_model.py', *actions])
        global model_created
        model_created = True  # Update the model status
        return redirect(url_for('index'))  # Redirect back to home page once model is created
    return render_template('instructions.html')

@app.route('/test')
def test():
    if not model_created:
        return "Model not created yet. Please train the model first."
    # You can run a testing script here
    print("Test mode activated.")
    subprocess.run(['python', 'Gesture Recognition.py', *actions])
    return render_template('index.html')


@app.route('/exit')
def exit():
    # Exit the application or stop a process
    subprocess.run(['python', 'Deleting the files.py'])
    return "Exiting the system."

if __name__ == '__main__':
    app.run(debug=True)
