# auth.py
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User # Import db and User model from main app


auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('diagnostics.index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role', 'Patient')
        credit_plan = request.form.get('credit_plan')

        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists. Please choose a different one.', 'danger')
        else:
            if credit_plan == 'plan_200':
                credits = 200
            else:
                credits = 1

            new_user = User(username=username, role=role, credits=credits)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('auth.login'))
    return render_template('index.html', show_signup=True)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('diagnostics.index'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('diagnostics.index'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('index.html', show_login=True)

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('auth.login'))

@auth_bp.route('/topup', methods=['GET', 'POST'])
@login_required
def topup():
    if request.method == 'POST':
        credit_plan = request.form.get('credit_plan')
        if credit_plan == 'plan_200':
            current_user.credits += 200
            flash('200 credits added successfully!', 'success')
        else:
            current_user.credits += 1
            flash('1 credit added successfully!', 'success')
        db.session.commit()
        return redirect(url_for('diagnostics.index'))
    return render_template('index.html', show_topup=True)