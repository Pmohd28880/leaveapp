from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime, date
from sqlalchemy import func

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
    members = db.relationship('User', backref='user_team', lazy=True)


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='user')
    team_id = db.Column(db.Integer, db.ForeignKey('team.id'))
    hire_date = db.Column(db.Date, default=date.today)

    # Relationships
    leaves = db.relationship('Leave', backref='applicant', lazy=True,
                             foreign_keys='Leave.user_id')
    leave_balances = db.relationship(
        'LeaveBalance', backref='employee', lazy=True)

    def get_current_balance(self, leave_type):
        """Get current balance for a specific leave type"""
        current_year = datetime.now().year
        balance = LeaveBalance.query.filter_by(
            user_id=self.id,
            leave_type=leave_type,
            year=current_year
        ).first()
        return balance.remaining_days if balance else 0

    def get_all_balances(self, year=None):
        """Get all leave balances for a specific year"""
        if year is None:
            year = datetime.now().year
        return LeaveBalance.query.filter_by(user_id=self.id, year=year).all()


class LeaveType(db.Model):
    """Define different types of leaves available"""
    id = db.Column(db.Integer, primary_key=True)
    # e.g., 'Annual', 'Sick', 'Personal'
    name = db.Column(db.String(50), unique=True, nullable=False)
    code = db.Column(db.String(10), unique=True,
                     nullable=False)  # e.g., 'AL', 'SL', 'PL'
    # Default days allocated per year
    default_allocation = db.Column(db.Integer, default=0)
    # Max days that can be carried over
    max_carryover = db.Column(db.Integer, default=0)
    requires_approval = db.Column(db.Boolean, default=True)
    can_be_negative = db.Column(
        db.Boolean, default=False)  # Allow negative balance
    description = db.Column(db.String(200))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    leave_balances = db.relationship(
        'LeaveBalance', backref='leave_type_ref', lazy=True)


class LeaveBalance(db.Model):
    """Track leave balances for each user by leave type and year"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    leave_type_id = db.Column(db.Integer, db.ForeignKey(
        'leave_type.id'), nullable=False)
    year = db.Column(db.Integer, nullable=False)

    # Balance tracking
    # Total days allocated for the year
    allocated_days = db.Column(db.Float, default=0.0)
    used_days = db.Column(db.Float, default=0.0)  # Days used/taken
    pending_days = db.Column(db.Float, default=0.0)  # Days in pending requests
    # Days carried over from previous year
    carried_over = db.Column(db.Float, default=0.0)

    # Audit fields
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Unique constraint to prevent duplicate entries
    __table_args__ = (db.UniqueConstraint('user_id', 'leave_type_id', 'year'),)

    @property
    def remaining_days(self):
        """Calculate remaining leave days"""
        return self.allocated_days - self.used_days - self.pending_days

    @property
    def total_available(self):
        """Total days available (allocated + carried over)"""
        return self.allocated_days + self.carried_over

    def can_take_leave(self, days_requested):
        """Check if user can take the requested number of days"""
        return self.remaining_days >= days_requested

    def update_balance(self, days_used=0, days_pending=0):
        """Update the balance when leave is taken or status changes"""
        self.used_days += days_used
        self.pending_days += days_pending
        self.updated_at = datetime.utcnow()
        db.session.commit()

    def __repr__(self):
        return f'<LeaveBalance User:{self.user_id} Type:{self.leave_type_id} Year:{self.year} Remaining:{self.remaining_days}>'


class Leave(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    leave_type_id = db.Column(db.Integer, db.ForeignKey('leave_type.id'))
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    reason = db.Column(db.String(500))  # Increased length for detailed reasons
    # Pending, Approved, Rejected, Cancelled
    status = db.Column(db.String(20), default='Pending')
    decision_reason = db.Column(db.String(500))
    approved_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    approved_at = db.Column(db.DateTime)

    # Relationships
    leave_type_ref = db.relationship('LeaveType', backref='leaves')
    approver = db.relationship('User', foreign_keys=[
                               approved_by], backref='approved_leaves')

    @property
    def duration_days(self):
        """Calculate the number of days for this leave"""
        if self.start_date and self.end_date:
            return (self.end_date - self.start_date).days + 1
        return 0

    def update_leave_balance(self, old_status=None):
        """Update leave balance when leave status changes"""
        if not self.leave_type_id:
            return

        # Get the leave balance for this user and leave type
        year = self.start_date.year if self.start_date else datetime.now().year
        balance = LeaveBalance.query.filter_by(
            user_id=self.user_id,
            leave_type_id=self.leave_type_id,
            year=year
        ).first()

        if not balance:
            return

        days = self.duration_days

        # Handle status changes
        if old_status == 'Pending' and self.status == 'Approved':
            # Move from pending to used
            balance.pending_days -= days
            balance.used_days += days
        elif old_status == 'Pending' and self.status == 'Rejected':
            # Remove from pending
            balance.pending_days -= days
        elif old_status is None and self.status == 'Pending':
            # New pending request
            balance.pending_days += days
        elif old_status == 'Approved' and self.status == 'Cancelled':
            # Return used days
            balance.used_days -= days

        balance.updated_at = datetime.utcnow()
        db.session.commit()


class LeaveTransaction(db.Model):
    """Track all leave balance transactions for audit purposes"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    leave_type_id = db.Column(db.Integer, db.ForeignKey(
        'leave_type.id'), nullable=False)
    # Optional, for leave-related transactions
    leave_id = db.Column(db.Integer, db.ForeignKey('leave.id'))

    # 'allocation', 'usage', 'adjustment', 'carryover'
    transaction_type = db.Column(db.String(20), nullable=False)
    # Positive for additions, negative for deductions
    days_changed = db.Column(db.Float, nullable=False)
    balance_before = db.Column(db.Float, nullable=False)
    balance_after = db.Column(db.Float, nullable=False)

    description = db.Column(db.String(200))
    created_by = db.Column(db.Integer, db.ForeignKey(
        'user.id'))  # Who made this transaction
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    user = db.relationship('User', foreign_keys=[
                           user_id], backref='leave_transactions')
    leave_type = db.relationship('LeaveType', backref='transactions')
    leave = db.relationship('Leave', backref='transactions')
    creator = db.relationship('User', foreign_keys=[created_by])


# Helper functions for leave balance management
def initialize_leave_balances_for_user(user_id, year=None):
    """Initialize leave balances for a user for a given year"""
    if year is None:
        year = datetime.now().year

    leave_types = LeaveType.query.filter_by(is_active=True).all()

    for leave_type in leave_types:
        existing_balance = LeaveBalance.query.filter_by(
            user_id=user_id,
            leave_type_id=leave_type.id,
            year=year
        ).first()

        if not existing_balance:
            balance = LeaveBalance(
                user_id=user_id,
                leave_type_id=leave_type.id,
                year=year,
                allocated_days=leave_type.default_allocation
            )
            db.session.add(balance)

    db.session.commit()


def carry_over_leave_balances(user_id, from_year, to_year):
    """Carry over unused leave balances from one year to the next"""
    from_balances = LeaveBalance.query.filter_by(
        user_id=user_id,
        year=from_year
    ).all()

    for balance in from_balances:
        # Calculate carryover amount
        remaining = balance.remaining_days
        max_carryover = balance.leave_type_ref.max_carryover
        carryover_amount = min(
            remaining, max_carryover) if max_carryover > 0 else remaining

        if carryover_amount > 0:
            # Create or update balance for the new year
            new_balance = LeaveBalance.query.filter_by(
                user_id=user_id,
                leave_type_id=balance.leave_type_id,
                year=to_year
            ).first()

            if new_balance:
                new_balance.carried_over += carryover_amount
            else:
                new_balance = LeaveBalance(
                    user_id=user_id,
                    leave_type_id=balance.leave_type_id,
                    year=to_year,
                    allocated_days=balance.leave_type_ref.default_allocation,
                    carried_over=carryover_amount
                )
                db.session.add(new_balance)

    db.session.commit()


def get_leave_summary(user_id, year=None):
    """Get a summary of leave balances for a user"""
    if year is None:
        year = datetime.now().year

    balances = LeaveBalance.query.filter_by(
        user_id=user_id,
        year=year
    ).all()

    summary = {}
    for balance in balances:
        summary[balance.leave_type_ref.name] = {
            'allocated': balance.allocated_days,
            'used': balance.used_days,
            'pending': balance.pending_days,
            'carried_over': balance.carried_over,
            'remaining': balance.remaining_days,
            'total_available': balance.total_available
        }

    return summary
