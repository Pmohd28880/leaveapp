from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()


class Organization(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    departments = db.relationship('Department', backref='org', lazy=True)


class Department(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    org_id = db.Column(db.Integer, db.ForeignKey('organization.id'))
    teams = db.relationship('Team', backref='dept', lazy=True)


class Team(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    dept_id = db.Column(db.Integer, db.ForeignKey('department.id'))
    members = db.relationship(
        'User', backref='user_team', lazy=True)  # Changed backref name


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='user')
    team_id = db.Column(db.Integer, db.ForeignKey('team.id'))
    leaves = db.relationship('Leave', backref='applicant',
                             lazy=True, foreign_keys='Leave.user_id')


class Leave(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    reason = db.Column(db.String(200))
    status = db.Column(db.String(20), default='Pending')
    decision_reason = db.Column(db.String(500))  # Added this field
    approved_by = db.Column(db.Integer, db.ForeignKey(
        'user.id'))  # Separate foreign key
    created_at = db.Column(
        db.DateTime, default=datetime.utcnow)  # Added timestamp
