from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Leave, Team
from datetime import datetime, timedelta
import os
import logging
from logging.handlers import RotatingFileHandler
import json
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict, Optional
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import text  # Add this import
import calendar

# OpenAI Agents SDK imports
from agents import Agent, Runner, function_tool
from pydantic import BaseModel, Field
import nest_asyncio

# Allow nested event loops (needed for Flask + async)
nest_asyncio.apply()

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = "IAmhere"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///leave.db'
# Added to suppress warning
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove all existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
    handler.close()

# Create file handler
file_handler = RotatingFileHandler(
    'app.log',
    maxBytes=1024*1024,  # 1MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

# Add handler to logger
logger.addHandler(file_handler)

# Also configure Flask's default logger
app.logger.handlers.clear()
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# Flask-Login setup
login_manager = LoginManager(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# =====================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# =====================================================


class LeaveDecision(BaseModel):
    """Structured output for leave decisions"""
    status: str = Field(
        description="Decision status: Approved, Denied, or Escalate")
    reason: str = Field(description="Detailed explanation for the decision")
    confidence: float = Field(
        description="Confidence level between 0.0 and 1.0", ge=0.0, le=1.0)
    escalate: bool = Field(description="Whether this requires human review")
    agent_reasoning: str = Field(description="Step-by-step reasoning process")
    business_impact: str = Field(description="Assessment of business impact")
    employee_considerations: str = Field(
        description="Employee wellbeing factors considered")
    precedent_used: Optional[str] = Field(
        description="Similar cases referenced", default=None)
    recommended_actions: List[str] = Field(
        description="Any follow-up actions needed", default_factory=list)


class ContextAnalysis(BaseModel):
    """Analysis of leave request context"""
    user_pattern: str = Field(description="Employee's leave usage pattern")
    team_impact: str = Field(description="Impact on team operations")
    business_timing: str = Field(
        description="Business calendar considerations")
    risk_assessment: str = Field(
        description="Overall risk level: low, medium, high")
    recommendations: List[str] = Field(
        description="recommended actions for decision", default_factory=list)


class RoutingDecision(BaseModel):
    """Structured output for triage agent routing decisions"""
    route_to: str = Field(
        description="Agent to route to: 'decision_agent' or 'escalation_agent'")
    reason: str = Field(description="Reason for routing decision")
    confidence: float = Field(
        description="Confidence in routing decision", ge=0.0, le=1.0)
    escalate: bool = Field(
        description="Whether the request requires immediate escalation", default=False)
    analysis: str = Field(description="Brief analysis of the request")
# =====================================================
# DATACLASS FOR LEAVE REQUESTS
# =====================================================


@dataclass
class LeaveRequest:
    reason: str
    start_date: str
    end_date: str
    user_id: int
    duration_days: int = None

    def __post_init__(self):
        if self.duration_days is None:
            start = datetime.strptime(self.start_date, '%Y-%m-%d')
            end = datetime.strptime(self.end_date, '%Y-%m-%d')
            self.duration_days = (end - start).days + 1

# =====================================================
# AGENT TOOLS (Functions agents can call)
# =====================================================


def safe_date_conversion(date_input):
    """Safely convert date input to both string and datetime objects"""
    if isinstance(date_input, str):
        # Already a string, parse to datetime for datetime operations
        dt_obj = datetime.strptime(date_input, '%Y-%m-%d').date()
        return date_input, dt_obj
    elif hasattr(date_input, 'strftime'):
        # It's a datetime object, convert to string
        str_obj = date_input.strftime('%Y-%m-%d')
        dt_obj = date_input.date() if hasattr(date_input, 'date') else date_input
        return str_obj, dt_obj
    else:
        raise ValueError(f"Unsupported date format: {type(date_input)}")


@function_tool
def check_duplicate_leave_request_tool(user_id: int, start_date: str, end_date: str) -> Dict:
    """AI Tool version - Check if user already has approved/pending leave for the requested period"""
    return check_duplicate_leave_request_impl(user_id, start_date, end_date)


def check_duplicate_leave_request_impl(user_id: int, start_date: str, end_date: str) -> Dict:
    """Check if user already has approved/pending leave for the requested period"""
    try:
        logger.info(
            f"🔍 Checking for duplicate leave requests for user {user_id}")

        with app.app_context():
            start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

            # Check for overlapping leaves that are approved or pending
            overlapping_leaves = Leave.query.filter(
                Leave.user_id == user_id,
                Leave.status.in_(['Approved', 'Pending']),
                Leave.start_date <= end_dt,
                Leave.end_date >= start_dt
            ).all()

            if overlapping_leaves:
                conflicts = []
                for leave in overlapping_leaves:
                    conflicts.append({
                        'leave_id': leave.id,
                        'start_date': leave.start_date.strftime('%Y-%m-%d'),
                        'end_date': leave.end_date.strftime('%Y-%m-%d'),
                        'status': leave.status,
                        'reason': leave.reason,
                        'duration_days': leave.duration_days
                    })

                logger.warning(
                    f"Found {len(conflicts)} conflicting leave requests for user {user_id}")

                return {
                    'has_conflict': True,
                    'conflict_type': 'overlapping_leave',
                    'total_conflicts': len(conflicts),
                    'conflicting_leaves': conflicts,
                    'message': f"User already has {len(conflicts)} approved/pending leave(s) overlapping with requested dates",
                    'should_reject': True,
                    'rejection_reason': "Cannot apply for leave during a period when you already have approved or pending leave"
                }

            # Check for leaves in close proximity (within 7 days before/after)
            proximity_start = start_dt - timedelta(days=7)
            proximity_end = end_dt + timedelta(days=7)

            nearby_leaves = Leave.query.filter(
                Leave.user_id == user_id,
                Leave.status == 'Approved',
                Leave.start_date >= proximity_start,
                Leave.end_date <= proximity_end,
                # Exclude overlapping ones already checked
                ~((Leave.start_date <= end_dt) & (Leave.end_date >= start_dt))
            ).all()

            nearby_info = []
            if nearby_leaves:
                for leave in nearby_leaves:
                    nearby_info.append({
                        'leave_id': leave.id,
                        'start_date': leave.start_date.strftime('%Y-%m-%d'),
                        'end_date': leave.end_date.strftime('%Y-%m-%d'),
                        'days_gap': min(
                            abs((leave.end_date - start_dt).days),
                            abs((end_dt - leave.start_date).days)
                        )
                    })

            logger.info(f"No conflicting leaves found for user {user_id}")

            return {
                'has_conflict': False,
                'conflict_type': None,
                'total_conflicts': 0,
                'conflicting_leaves': [],
                'nearby_leaves': nearby_info,
                'message': "No conflicting leave requests found",
                'should_reject': False,
                'rejection_reason': None
            }

    except Exception as e:
        logger.error(f"Error checking duplicate leave requests: {str(e)}")
        return {
            'has_conflict': False,
            'conflict_type': 'error',
            'total_conflicts': 0,
            'conflicting_leaves': [],
            'message': f"Error checking conflicts: {str(e)}",
            'should_reject': False,
            'rejection_reason': None,
            'error': str(e)
        }


@function_tool
def validate_leave_dates_tool(start_date: str, end_date: str, user_id: int = None) -> Dict:
    """AI Tool version - Validate leave dates for weekends, holidays, and business rules"""
    return validate_leave_dates_impl(start_date, end_date, user_id)


def validate_leave_dates_impl(start_date: str, end_date: str, user_id: int = None) -> Dict:
    """Validate leave dates for weekends, holidays, and business rules"""
    try:
        logger.info(f"📅 Validating leave dates: {start_date} to {end_date}")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

        # Basic date validation
        if start_dt > end_dt:
            return {
                'is_valid': False,
                'validation_type': 'invalid_date_range',
                'message': "Start date cannot be after end date",
                'should_reject': True,
                'rejection_reason': "Invalid date range: Start date is after end date"
            }

        if start_dt < datetime.now().date():
            return {
                'is_valid': False,
                'validation_type': 'past_date',
                'message': "Cannot apply for leave in the past",
                'should_reject': True,
                'rejection_reason': "Leave start date cannot be in the past"
            }

        # Generate list of all dates in the leave period
        leave_dates = []
        current_date = start_dt
        while current_date <= end_dt:
            leave_dates.append(current_date)
            current_date += timedelta(days=1)

        # Check for weekends
        weekend_dates = []
        working_dates = []

        for date in leave_dates:
            if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                weekend_dates.append(date.strftime('%Y-%m-%d'))
            else:
                working_dates.append(date.strftime('%Y-%m-%d'))

        # Check if all dates are weekends
        if len(weekend_dates) == len(leave_dates):
            return {
                'is_valid': False,
                'validation_type': 'weekend_only',
                'message': "Cannot apply for leave on weekends only",
                'weekend_dates': weekend_dates,
                'working_dates': working_dates,
                'total_weekend_days': len(weekend_dates),
                'total_working_days': len(working_dates),
                'should_reject': True,
                'rejection_reason': "Leave request covers only weekends. Please apply for leave on working days."
            }

        # Check for excessive future dating (more than 90 days in advance)
        days_in_advance = (start_dt - datetime.now().date()).days
        if days_in_advance > 90:
            return {
                'is_valid': False,
                'validation_type': 'too_far_future',
                'message': f"Leave requested {days_in_advance} days in advance (max 90 days allowed)",
                'days_in_advance': days_in_advance,
                'should_reject': True,
                'rejection_reason': f"Leave cannot be requested more than 90 days in advance. Current request is {days_in_advance} days ahead."
            }

        # Check for minimum notice period (less than 2 days notice for non-emergency)
        if days_in_advance < 2:
            notice_warning = {
                'validation_type': 'short_notice',
                'message': f"Only {days_in_advance} day(s) notice provided. Consider if this requires manager approval.",
                'days_in_advance': days_in_advance,
                'requires_manager_approval': True
            }
        else:
            notice_warning = None

        # Check for long duration (more than 15 consecutive days)
        total_days = len(leave_dates)
        if total_days > 15:
            duration_warning = {
                'validation_type': 'long_duration',
                'message': f"Extended leave of {total_days} days may require special approval",
                'total_days': total_days,
                'requires_special_approval': True
            }
        else:
            duration_warning = None

        # Check for common holiday periods
        holiday_periods = [
            {'start': '12-20', 'end': '01-05', 'name': 'Christmas/New Year'},
            {'start': '07-01', 'end': '08-31', 'name': 'Summer Holiday Period'},
        ]

        holiday_overlap = []
        for holiday in holiday_periods:
            start_month_day = start_dt.strftime('%m-%d')
            end_month_day = end_dt.strftime('%m-%d')

            if (start_month_day >= holiday['start'] and start_month_day <= holiday['end']) or \
               (end_month_day >= holiday['start'] and end_month_day <= holiday['end']):
                holiday_overlap.append(holiday['name'])

        logger.info(
            f"Date validation complete - Working days: {len(working_dates)}, Weekend days: {len(weekend_dates)}")

        return {
            'is_valid': True,
            'validation_type': 'valid',
            'message': "Leave dates are valid",
            'weekend_dates': weekend_dates,
            'working_dates': working_dates,
            'total_weekend_days': len(weekend_dates),
            'total_working_days': len(working_dates),
            'total_days': total_days,
            'days_in_advance': days_in_advance,
            'holiday_periods_overlap': holiday_overlap,
            'should_reject': False,
            'rejection_reason': None,
            'warnings': [w for w in [notice_warning, duration_warning] if w is not None]
        }

    except Exception as e:
        logger.error(f"Error validating leave dates: {str(e)}")
        return {
            'is_valid': False,
            'validation_type': 'error',
            'message': f"Error validating dates: {str(e)}",
            'should_reject': False,
            'rejection_reason': None,
            'error': str(e)
        }


@function_tool
def get_comprehensive_leave_validation_tool(user_id: int, start_date: str, end_date: str) -> Dict:
    """AI Tool version - Comprehensive validation combining duplicate checks and date validation"""
    return get_comprehensive_leave_validation_impl(user_id, start_date, end_date)


def get_comprehensive_leave_validation_impl(user_id: int, start_date: str, end_date: str) -> Dict:
    """Comprehensive validation combining duplicate checks and date validation"""
    try:
        logger.info(
            f"🔍 Running comprehensive leave validation for user {user_id}")

        # Run both validations using the implementation functions
        duplicate_check = check_duplicate_leave_request_impl(
            user_id, start_date, end_date)
        date_validation = validate_leave_dates_impl(
            start_date, end_date, user_id)

        # Determine overall result
        should_reject = duplicate_check.get(
            'should_reject', False) or date_validation.get('should_reject', False)

        rejection_reasons = []
        if duplicate_check.get('rejection_reason'):
            rejection_reasons.append(duplicate_check['rejection_reason'])
        if date_validation.get('rejection_reason'):
            rejection_reasons.append(date_validation['rejection_reason'])

        warnings = []
        if date_validation.get('warnings'):
            warnings.extend(date_validation['warnings'])
        if duplicate_check.get('nearby_leaves'):
            warnings.append({
                'validation_type': 'nearby_leaves',
                'message': f"You have {len(duplicate_check['nearby_leaves'])} other approved leaves within 7 days of this request",
                'nearby_leaves': duplicate_check['nearby_leaves']
            })

        return {
            'overall_valid': not should_reject,
            'should_reject': should_reject,
            'rejection_reasons': rejection_reasons,
            'warnings': warnings,
            'duplicate_check_result': duplicate_check,
            'date_validation_result': date_validation,
            'summary': {
                'has_conflicts': duplicate_check.get('has_conflict', False),
                'weekend_only': date_validation.get('validation_type') == 'weekend_only',
                'working_days': date_validation.get('total_working_days', 0),
                'total_days': date_validation.get('total_days', 0),
                'requires_special_approval': any(w.get('requires_special_approval', False) for w in warnings),
                'requires_manager_approval': any(w.get('requires_manager_approval', False) for w in warnings)
            }
        }

    except Exception as e:
        logger.error(f"Error in comprehensive leave validation: {str(e)}")
        return {
            'overall_valid': True,
            'should_reject': False,
            'rejection_reasons': [],
            'warnings': [{'validation_type': 'error', 'message': f"Validation error: {str(e)}"}],
            'error': str(e)
        }


@function_tool
def analyze_team_availability_tool(team_id: int, start_date: str, end_date: str) -> Dict:
    """AI Tool version - Analyze team availability during requested leave period"""
    return analyze_team_availability_impl(team_id, start_date, end_date)


def analyze_team_availability_impl(team_id: int, start_date: str, end_date: str) -> Dict:
    """Analyze team availability during requested leave period with enhanced debugging"""
    try:
        logger.info(f"👥 Analyzing team availability for team {team_id}")
        logger.info(f"📅 Leave period: {start_date} to {end_date}")

        with app.app_context():
            team = Team.query.get(team_id)
            if not team:
                logger.error(
                    f"❌ Team {team_id} not found - defaulting to critical impact")
                return {
                    'error': 'Team not found',
                    'total_team_members': 1,
                    'available_members': 0,
                    'impact_level': 'critical',
                    'debug_info': f'Team {team_id} not found in database'
                }

            total_members = len(team.members)
            logger.info(
                f"📊 Team '{team.name}' (ID: {team_id}) has {total_members} total members")

            # Debug: List all team members
            member_ids = [member.id for member in team.members]
            logger.info(f"👤 Team member IDs: {member_ids}")

            # Safe date conversion
            start_date_str, start_dt = safe_date_conversion(start_date)
            end_date_str, end_dt = safe_date_conversion(end_date)

            # Fixed join with explicit condition
            overlapping_leaves = Leave.query.join(
                User, User.id == Leave.user_id
            ).filter(
                User.team_id == team_id,
                Leave.status == 'Approved',
                Leave.start_date <= end_dt,
                Leave.end_date >= start_dt
            ).all()

            members_on_leave = len(
                {leave.user_id for leave in overlapping_leaves})
            available_members = total_members - members_on_leave

            logger.info(f"📈 Leave analysis:")
            logger.info(
                f"   - Overlapping approved leaves: {len(overlapping_leaves)}")
            logger.info(f"   - Unique members on leave: {members_on_leave}")
            logger.info(f"   - Available members: {available_members}")

            # Debug: List members on leave
            if overlapping_leaves:
                leave_details = [(leave.user_id, leave.start_date, leave.end_date)
                                 for leave in overlapping_leaves]
                logger.info(f"   - Leave details: {leave_details}")

            # Enhanced impact calculation with detailed logging
            if total_members == 1:
                impact_level = "critical"
                logger.warning(
                    f"🚨 CRITICAL: Single person team - any leave creates critical impact")
            elif available_members <= 0:
                impact_level = "critical"
                logger.warning(
                    f"🚨 CRITICAL: No members available ({available_members})")
            elif available_members <= 1:
                impact_level = "critical"
                logger.warning(
                    f"🚨 CRITICAL: Only {available_members} member(s) available")
            elif available_members <= total_members * 0.5:
                impact_level = "high"
                logger.warning(
                    f"⚠️ HIGH: Only {available_members}/{total_members} members available")
            elif available_members <= total_members * 0.7:
                impact_level = "medium"
                logger.info(
                    f"⚠️ MEDIUM: {available_members}/{total_members} members available")
            else:
                impact_level = "low"
                logger.info(
                    f"✅ LOW: {available_members}/{total_members} members available")

            logger.info(
                f"📊 Final team availability impact level: {impact_level}")

            return {
                'total_team_members': total_members,
                'members_on_leave': members_on_leave,
                'available_members': available_members,
                'availability_percentage': (available_members / total_members) * 100 if total_members > 0 else 0,
                'impact_level': impact_level,
                'overlapping_leaves': len(overlapping_leaves),
                'team_name': team.name,
                'debug_info': {
                    'team_member_ids': member_ids,
                    'overlapping_leave_details': [(leave.user_id, str(leave.start_date), str(leave.end_date)) for leave in overlapping_leaves],
                    'calculation_basis': f'{available_members} available out of {total_members} total'
                }
            }

    except Exception as e:
        logger.error(f"💥 Error analyzing team availability: {str(e)}")
        logger.error(f"📍 Error details: {type(e).__name__}: {str(e)}")
        return {
            'total_team_members': 1,
            'members_on_leave': 1,
            'available_members': 0,
            'availability_percentage': 0,
            'impact_level': 'critical',
            'error': str(e),
            'debug_info': f'Error occurred: {type(e).__name__}: {str(e)}'
        }


def safe_date_conversion(date_input):
    """Safely convert date input to both string and datetime objects"""
    if isinstance(date_input, str):
        # Already a string, parse to datetime for datetime operations
        dt_obj = datetime.strptime(date_input, '%Y-%m-%d').date()
        return date_input, dt_obj
    elif hasattr(date_input, 'strftime'):
        # It's a datetime object, convert to string
        str_obj = date_input.strftime('%Y-%m-%d')
        dt_obj = date_input.date() if hasattr(date_input, 'date') else date_input
        return str_obj, dt_obj
    else:
        raise ValueError(f"Unsupported date format: {type(date_input)}")


# Also update your main processing to call the team analysis
def get_team_analysis_for_agent(team_id: int, start_date: str, end_date: str) -> str:
    """Get formatted team analysis for agent input"""
    team_analysis = analyze_team_availability_impl(
        team_id, start_date, end_date)

    analysis_text = f"""
    Team Analysis:
    - Team: {team_analysis.get('team_name', 'Unknown')}
    - Total Members: {team_analysis.get('total_team_members', 0)}
    - Available Members: {team_analysis.get('available_members', 0)}
    - Availability: {team_analysis.get('availability_percentage', 0):.1f}%
    - Impact Level: {team_analysis.get('impact_level', 'unknown').upper()}
    - Members Currently on Leave: {team_analysis.get('members_on_leave', 0)}
    """

    if team_analysis.get('impact_level') == 'critical':
        analysis_text += f"\n    ⚠️ CRITICAL IMPACT: This leave would severely impact team operations!"

    return analysis_text


@function_tool
def get_user_leave_history(user_id: int) -> Dict:
    """Retrieve and analyze user's leave history patterns"""
    try:
        logger.info(f"🔍 Analyzing leave history for user {user_id}")

        with app.app_context():  # Ensure database session is active
            # Get recent leave requests
            recent_leaves = Leave.query.filter_by(user_id=user_id).order_by(
                Leave.start_date.desc()).limit(10).all()

            if recent_leaves:
                total_days = sum(
                    [(leave.end_date - leave.start_date).days + 1 for leave in recent_leaves])
                sick_count = len(
                    [leave for leave in recent_leaves if 'sick' in leave.reason.lower()])
                approved_count = len(
                    [leave for leave in recent_leaves if leave.status == 'Approved'])

                # Calculate patterns
                avg_days_per_request = total_days / \
                    len(recent_leaves) if recent_leaves else 0
                approval_rate = (approved_count / len(recent_leaves)
                                 ) * 100 if recent_leaves else 0

                # Determine pattern classification (relaxed threshold)
                if total_days > 60:  # Changed from 45 to 60
                    pattern = "high_usage"
                elif sick_count > len(recent_leaves) * 0.8:
                    pattern = "frequent_sick_leave"
                elif avg_days_per_request > 10:
                    pattern = "long_duration_requests"
                else:
                    pattern = "normal"

                return {
                    'total_recent_days': total_days,
                    'total_requests': len(recent_leaves),
                    'sick_leave_count': sick_count,
                    'approval_rate': approval_rate,
                    'average_duration': avg_days_per_request,
                    'pattern_classification': pattern,
                    'last_leave_date': recent_leaves[0].end_date.strftime('%Y-%m-%d') if recent_leaves[0].end_date else 'N/A',
                    'recent_reasons': [leave.reason for leave in recent_leaves[:3]]
                }
            else:
                return {
                    'total_recent_days': 0,
                    'total_requests': 0,
                    'pattern_classification': 'new_employee',
                    'message': 'No previous leave history found'
                }

    except Exception as e:
        logger.error(f"Error analyzing user history: {str(e)}")
        return {'error': str(e), 'pattern_classification': 'unknown'}


@function_tool
def check_duplicate_leave_request(user_id: int, start_date: str, end_date: str) -> Dict:
    """Check if user already has approved/pending leave for the requested period"""
    try:
        logger.info(
            f"🔍 Checking for duplicate leave requests for user {user_id}")

        with app.app_context():
            start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

            # Check for overlapping leaves that are approved or pending
            overlapping_leaves = Leave.query.filter(
                Leave.user_id == user_id,
                Leave.status.in_(['Approved', 'Pending']),
                Leave.start_date <= end_dt,
                Leave.end_date >= start_dt
            ).all()

            if overlapping_leaves:
                conflicts = []
                for leave in overlapping_leaves:
                    conflicts.append({
                        'leave_id': leave.id,
                        'start_date': leave.start_date.strftime('%Y-%m-%d'),
                        'end_date': leave.end_date.strftime('%Y-%m-%d'),
                        'status': leave.status,
                        'reason': leave.reason,
                        'duration_days': leave.duration_days
                    })

                logger.warning(
                    f"Found {len(conflicts)} conflicting leave requests for user {user_id}")

                return {
                    'has_conflict': True,
                    'conflict_type': 'overlapping_leave',
                    'total_conflicts': len(conflicts),
                    'conflicting_leaves': conflicts,
                    'message': f"User already has {len(conflicts)} approved/pending leave(s) overlapping with requested dates",
                    'should_reject': True,
                    'rejection_reason': "Cannot apply for leave during a period when you already have approved or pending leave"
                }

            # Check for leaves in close proximity (within 7 days before/after)
            proximity_start = start_dt - timedelta(days=7)
            proximity_end = end_dt + timedelta(days=7)

            nearby_leaves = Leave.query.filter(
                Leave.user_id == user_id,
                Leave.status == 'Approved',
                Leave.start_date >= proximity_start,
                Leave.end_date <= proximity_end,
                # Exclude overlapping ones already checked
                ~((Leave.start_date <= end_dt) & (Leave.end_date >= start_dt))
            ).all()

            nearby_info = []
            if nearby_leaves:
                for leave in nearby_leaves:
                    nearby_info.append({
                        'leave_id': leave.id,
                        'start_date': leave.start_date.strftime('%Y-%m-%d'),
                        'end_date': leave.end_date.strftime('%Y-%m-%d'),
                        'days_gap': min(
                            abs((leave.end_date - start_dt).days),
                            abs((end_dt - leave.start_date).days)
                        )
                    })

            logger.info(f"No conflicting leaves found for user {user_id}")

            return {
                'has_conflict': False,
                'conflict_type': None,
                'total_conflicts': 0,
                'conflicting_leaves': [],
                'nearby_leaves': nearby_info,
                'message': "No conflicting leave requests found",
                'should_reject': False,
                'rejection_reason': None
            }

    except Exception as e:
        logger.error(f"Error checking duplicate leave requests: {str(e)}")
        return {
            'has_conflict': False,
            'conflict_type': 'error',
            'total_conflicts': 0,
            'conflicting_leaves': [],
            'message': f"Error checking conflicts: {str(e)}",
            'should_reject': False,
            'rejection_reason': None,
            'error': str(e)
        }


@function_tool
def validate_leave_dates(start_date: str, end_date: str, user_id: int = None) -> Dict:
    """Validate leave dates for weekends, holidays, and business rules"""
    try:
        logger.info(f"📅 Validating leave dates: {start_date} to {end_date}")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

        # Basic date validation
        if start_dt > end_dt:
            return {
                'is_valid': False,
                'validation_type': 'invalid_date_range',
                'message': "Start date cannot be after end date",
                'should_reject': True,
                'rejection_reason': "Invalid date range: Start date is after end date"
            }

        if start_dt < datetime.now().date():
            return {
                'is_valid': False,
                'validation_type': 'past_date',
                'message': "Cannot apply for leave in the past",
                'should_reject': True,
                'rejection_reason': "Leave start date cannot be in the past"
            }

        # Generate list of all dates in the leave period
        leave_dates = []
        current_date = start_dt
        while current_date <= end_dt:
            leave_dates.append(current_date)
            current_date += timedelta(days=1)

        # Check for weekends
        weekend_dates = []
        working_dates = []

        for date in leave_dates:
            if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                weekend_dates.append(date.strftime('%Y-%m-%d'))
            else:
                working_dates.append(date.strftime('%Y-%m-%d'))

        # Check if all dates are weekends
        if len(weekend_dates) == len(leave_dates):
            return {
                'is_valid': False,
                'validation_type': 'weekend_only',
                'message': "Cannot apply for leave on weekends only",
                'weekend_dates': weekend_dates,
                'working_dates': working_dates,
                'total_weekend_days': len(weekend_dates),
                'total_working_days': len(working_dates),
                'should_reject': True,
                'rejection_reason': "Leave request covers only weekends. Please apply for leave on working days."
            }

        # Check for excessive future dating (more than 90 days in advance)
        days_in_advance = (start_dt - datetime.now().date()).days
        if days_in_advance > 90:
            return {
                'is_valid': False,
                'validation_type': 'too_far_future',
                'message': f"Leave requested {days_in_advance} days in advance (max 90 days allowed)",
                'days_in_advance': days_in_advance,
                'should_reject': True,
                'rejection_reason': f"Leave cannot be requested more than 90 days in advance. Current request is {days_in_advance} days ahead."
            }

        # Check for minimum notice period (less than 2 days notice for non-emergency)
        if days_in_advance < 2:
            notice_warning = {
                'validation_type': 'short_notice',
                'message': f"Only {days_in_advance} day(s) notice provided. Consider if this requires manager approval.",
                'days_in_advance': days_in_advance,
                'requires_manager_approval': True
            }
        else:
            notice_warning = None

        # Check for long duration (more than 15 consecutive days)
        total_days = len(leave_dates)
        if total_days > 15:
            duration_warning = {
                'validation_type': 'long_duration',
                'message': f"Extended leave of {total_days} days may require special approval",
                'total_days': total_days,
                'requires_special_approval': True
            }
        else:
            duration_warning = None

        # Check for common holiday periods (you can customize these dates)
        holiday_periods = [
            # Christmas/New Year
            {'start': '12-20', 'end': '01-05', 'name': 'Christmas/New Year'},
            # Summer vacation (adjust based on your region)
            {'start': '07-01', 'end': '08-31', 'name': 'Summer Holiday Period'},
            # Add more holiday periods as needed
        ]

        holiday_overlap = []
        for holiday in holiday_periods:
            # Simple check - can be made more sophisticated
            start_month_day = start_dt.strftime('%m-%d')
            end_month_day = end_dt.strftime('%m-%d')

            if (start_month_day >= holiday['start'] and start_month_day <= holiday['end']) or \
               (end_month_day >= holiday['start'] and end_month_day <= holiday['end']):
                holiday_overlap.append(holiday['name'])

        logger.info(
            f"Date validation complete - Working days: {len(working_dates)}, Weekend days: {len(weekend_dates)}")

        return {
            'is_valid': True,
            'validation_type': 'valid',
            'message': "Leave dates are valid",
            'weekend_dates': weekend_dates,
            'working_dates': working_dates,
            'total_weekend_days': len(weekend_dates),
            'total_working_days': len(working_dates),
            'total_days': total_days,
            'days_in_advance': days_in_advance,
            'holiday_periods_overlap': holiday_overlap,
            'should_reject': False,
            'rejection_reason': None,
            'warnings': [w for w in [notice_warning, duration_warning] if w is not None]
        }

    except Exception as e:
        logger.error(f"Error validating leave dates: {str(e)}")
        return {
            'is_valid': False,
            'validation_type': 'error',
            'message': f"Error validating dates: {str(e)}",
            'should_reject': False,  # Don't reject on technical errors
            'rejection_reason': None,
            'error': str(e)
        }


@function_tool
def get_comprehensive_leave_validation(user_id: int, start_date: str, end_date: str) -> Dict:
    """Comprehensive validation combining duplicate checks and date validation"""
    try:
        logger.info(
            f"🔍 Running comprehensive leave validation for user {user_id}")

        # Run both validations
        duplicate_check = check_duplicate_leave_request(
            user_id, start_date, end_date)
        date_validation = validate_leave_dates(start_date, end_date, user_id)

        # Determine overall result
        should_reject = duplicate_check.get(
            'should_reject', False) or date_validation.get('should_reject', False)

        rejection_reasons = []
        if duplicate_check.get('rejection_reason'):
            rejection_reasons.append(duplicate_check['rejection_reason'])
        if date_validation.get('rejection_reason'):
            rejection_reasons.append(date_validation['rejection_reason'])

        warnings = []
        if date_validation.get('warnings'):
            warnings.extend(date_validation['warnings'])
        if duplicate_check.get('nearby_leaves'):
            warnings.append({
                'validation_type': 'nearby_leaves',
                'message': f"You have {len(duplicate_check['nearby_leaves'])} other approved leaves within 7 days of this request",
                'nearby_leaves': duplicate_check['nearby_leaves']
            })

        return {
            'overall_valid': not should_reject,
            'should_reject': should_reject,
            'rejection_reasons': rejection_reasons,
            'warnings': warnings,
            'duplicate_check_result': duplicate_check,
            'date_validation_result': date_validation,
            'summary': {
                'has_conflicts': duplicate_check.get('has_conflict', False),
                'weekend_only': date_validation.get('validation_type') == 'weekend_only',
                'working_days': date_validation.get('total_working_days', 0),
                'total_days': date_validation.get('total_days', 0),
                'requires_special_approval': any(w.get('requires_special_approval', False) for w in warnings),
                'requires_manager_approval': any(w.get('requires_manager_approval', False) for w in warnings)
            }
        }

    except Exception as e:
        logger.error(f"Error in comprehensive leave validation: {str(e)}")
        return {
            'overall_valid': True,  # Don't block on technical errors
            'should_reject': False,
            'rejection_reasons': [],
            'warnings': [{'validation_type': 'error', 'message': f"Validation error: {str(e)}"}],
            'error': str(e)
        }


@function_tool
def analyze_team_availability(team_id: int, start_date: str, end_date: str) -> Dict:
    """Analyze team availability during requested leave period"""
    try:
        logger.info(f"👥 Analyzing team availability for team {team_id}")

        with app.app_context():
            team = Team.query.get(team_id)
            if not team:
                logger.error(
                    f"Team {team_id} not found - defaulting to critical impact")
                return {
                    'error': 'Team not found',
                    'total_team_members': 1,
                    'available_members': 0,  # ← Changed: No one available if team not found
                    'impact_level': 'critical'  # ← Changed: Critical impact
                }

            total_members = len(team.members)
            logger.info(f"Team {team.name} has {total_members} total members")

            start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

            # Fixed join with explicit condition
            overlapping_leaves = Leave.query.join(
                User, User.id == Leave.user_id
            ).filter(
                User.team_id == team_id,
                Leave.status == 'Approved',
                Leave.start_date <= end_dt,
                Leave.end_date >= start_dt
            ).all()

            members_on_leave = len(
                {leave.user_id for leave in overlapping_leaves})
            available_members = total_members - members_on_leave

            logger.info(
                f"Members on leave: {members_on_leave}, Available: {available_members}")

            # Impact calculation - be more strict for small teams
            if total_members == 1:
                impact_level = "critical"  # Single person teams are always critical
            elif available_members <= 0:
                impact_level = "critical"
            elif available_members <= 1:
                impact_level = "critical"  # Only 1 person left is critical
            elif available_members <= total_members * 0.5:
                impact_level = "high"
            elif available_members <= total_members * 0.7:
                impact_level = "medium"
            else:
                impact_level = "low"

            logger.info(f"Team availability impact level: {impact_level}")

            return {
                'total_team_members': total_members,
                'members_on_leave': members_on_leave,
                'available_members': available_members,
                'availability_percentage': (available_members / total_members) * 100 if total_members > 0 else 0,
                'impact_level': impact_level,
                'overlapping_leaves': len(overlapping_leaves),
                'team_name': team.name
            }

    except Exception as e:
        logger.error(f"Error analyzing team availability: {str(e)}")
        return {
            'total_team_members': 1,
            'members_on_leave': 1,  # ← Changed: Assume the person is taking leave
            'available_members': 0,  # ← Changed: No one available on error
            'availability_percentage': 0,  # ← Changed: 0% availability
            'impact_level': 'critical',  # ← Changed: Critical impact on error
            'error': str(e)
        }


@function_tool
def assess_business_calendar_impact(start_date: str, end_date: str) -> Dict:
    """Assess impact based on business calendar and critical periods"""
    try:
        logger.info(
            f"📅 Assessing business calendar impact for {start_date} to {end_date}")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        critical_periods = []
        impact_factors = []

        # Check for month-end (last 5 days of month)
        if start_dt.day >= 26 or end_dt.day >= 26:
            critical_periods.append("month_end_reporting")
            impact_factors.append(
                "Financial reporting and month-end processes")

        # Check for quarter-end
        quarter_end_months = [3, 6, 9, 12]
        if (start_dt.month in quarter_end_months and start_dt.day >= 25) or \
           (end_dt.month in quarter_end_months and end_dt.day >= 25):
            critical_periods.append("quarter_end")
            impact_factors.append(
                "Quarter-end financial activities and reporting")

        # Check for year-end
        if (start_dt.month == 12 and start_dt.day >= 20) or \
           (end_dt.month == 1 and end_dt.day <= 10):
            critical_periods.append("year_end")
            impact_factors.append("Year-end closing and planning activities")

        # Disabled random project deadlines for consistency
        project_risk = False  # Changed from random.choice([True, False])
        if project_risk:
            critical_periods.append("project_deadlines")
            impact_factors.append("Potential project milestone deadlines")

        # Determine overall business impact
        if len(critical_periods) >= 2:
            business_impact = "high"
        elif len(critical_periods) == 1:
            business_impact = "medium"
        else:
            business_impact = "low"

        return {
            'business_impact_level': business_impact,
            'critical_periods': critical_periods,
            'impact_factors': impact_factors,
            'recommendation': "Require additional approval" if business_impact == "high" else "Standard process",
            'start_date': start_date,
            'end_date': end_date
        }

    except Exception as e:
        logger.error(f"Error assessing business impact: {str(e)}")
        return {'error': str(e), 'business_impact_level': 'unknown'}


@function_tool
def get_similar_leave_decisions(reason: str, duration_days: int, user_id: int) -> Dict:
    """Find similar past leave decisions for precedent analysis"""
    try:
        if not reason or not isinstance(duration_days, int) or duration_days <= 0:
            return {
                'similar_cases_found': 0,
                'precedents': [],
                'historical_approval_rate': 0,
                'recommendation': 'Invalid input parameters'
            }
        logger.info(
            f"🔍 Finding similar decisions for {reason}, {duration_days} days")

        with app.app_context():  # Ensure database session
            # Find leaves with similar characteristics
            similar_leaves = Leave.query.filter(
                Leave.user_id != user_id,
                Leave.reason.contains(reason.split()[0])
            ).limit(5).all()

            if not similar_leaves:
                return {'message': 'No similar cases found', 'precedents': []}

            precedents = []
            for leave in similar_leaves:
                leave_duration = (leave.end_date - leave.start_date).days + 1
                precedents.append({
                    'duration': leave_duration,
                    'reason': leave.reason,
                    'status': leave.status,
                    'decision_reason': leave.decision_reason,
                    'similarity_score': max(0, 100 - abs(leave_duration - duration_days) * 5)
                })

            # Sort by similarity
            precedents.sort(key=lambda x: x['similarity_score'], reverse=True)

            approval_rate = len(
                [p for p in precedents if p['status'] == 'Approved']) / len(precedents) * 100

            return {
                'similar_cases_found': len(precedents),
                'precedents': precedents[:3],
                'historical_approval_rate': approval_rate,
                'recommendation': 'Favor approval' if approval_rate > 70 else 'Standard review' if approval_rate > 40 else 'Careful review'
            }

    except Exception as e:
        logger.error(f"Error finding similar decisions: {str(e)}")
        return {'error': str(e)}

# =====================================================
# SPECIALIZED AGENTS
# =====================================================


# Context Analysis Agent
context_agent = Agent(
    name="Leave Context Analyzer",
    instructions="""
    You are a specialized agent for analyzing the context around leave requests.

    Your role is to:
    1. Use available tools to gather comprehensive information
    2. Analyze patterns and identify potential concerns
    3. Provide structured assessment of the situation

    Focus on:
    - Employee leave patterns and history
    - Team impact and availability
    - Business timing considerations
    - Risk factors that need attention

    Be thorough but efficient in your analysis.
    """,
    tools=[get_user_leave_history, analyze_team_availability,
           assess_business_calendar_impact, get_similar_leave_decisions],
    output_type=ContextAnalysis
)

# Decision Making Agent
decision_agent = Agent(
    name="Leave Decision Maker",
    instructions="""
    You are the primary decision-making agent for employee leave requests.

    Core principles:
    1. Employee wellbeing is paramount
    2. Business continuity must be maintained
    3. Fairness and consistency in decisions
    4. Transparency in reasoning
    5. Escalate when human judgment is needed

    Decision guidelines:
    - Approve when impact is manageable and request is reasonable
    - Deny only when business impact is severe and no alternatives exist
    - Escalate for complex situations, policy exceptions, or high-risk scenarios

    Consider:
    - Employee's history and circumstances
    - Team capacity and availability
    - Business timing and critical periods
    - Precedent from similar cases
    - Company values of flexibility and employee support

    Always provide clear, empathetic reasoning for your decisions.

    Output your decision in the following JSON format:
    {
        "status": "Approved|Denied|Escalate",
        "reason": "<short summary>",
        "confidence": <float between 0 and 1>,
        "escalate": <true|false>,
        "agent_reasoning": "<detailed explanation>",
        "business_impact": "<low|medium|high|critical>",
        "employee_considerations": "<employee wellbeing factors>",
        "precedent_used": "<reference if any>",
        "recommended_actions": ["<action1>", "<action2>"]
    }
    """,
    tools=[get_user_leave_history, analyze_team_availability,
           assess_business_calendar_impact, get_similar_leave_decisions],
    output_type=LeaveDecision
)
# Escalation Agent
escalation_agent = Agent(
    name="Leave Escalation Specialist",
    instructions="""
    You handle complex and sensitive leave requests that require special consideration.

    Your expertise covers:
    - Extended leave requests (>14 days)
    - Medical and family emergencies
    - Unusual circumstances
    - Policy exceptions
    - High business impact situations

    Approach:
    - Gather comprehensive context
    - Consider all stakeholders
    - Identify accommodation options
    - Provide detailed analysis for human reviewers
    - Suggest alternative solutions when possible

    Remember: Your goal is to find solutions that work for both employee and business.
    """,
    tools=[get_user_leave_history, analyze_team_availability,
           assess_business_calendar_impact, get_similar_leave_decisions],
    output_type=LeaveDecision
)

# Triage Agent (Entry point)
triage_agent = Agent(
    name="Leave Request Triage",
    instructions="""
    You are the first point of contact for all leave requests. Your job is to:

    1. Quickly assess the complexity and sensitivity of the request
    2. Route to the appropriate specialist agent (decision_agent or escalation_agent)
    3. Provide a brief analysis of your routing decision

    Routing guidelines:

    → Standard Decision Agent (decision_agent) for:
    - Routine vacation requests (≤14 days)
    - Standard sick leave (≤5 days)
    - Personal leave with clear reasons
    - Low team/business impact situations

    → Escalation Specialist (escalation_agent) for:
    - Extended leave requests (>14 days)
    - Medical emergencies or serious illness (>7 days sick leave)
    - Family crisis situations (>5 days)
    - Requests during critical business periods with high team impact
    - Employees with concerning leave patterns (>90 days in 12 months)
    - Policy exception requests
    - High + critical business impact scenarios (both high impact AND critical timing)

    Output your decision in the following JSON format:
    {
        "route_to": "<decision_agent|escalation_agent>",
        "reason": "<brief reason for routing>",
        "confidence": <float between 0 and 1>,
        "escalate": <true|false>,
        "analysis": "<brief analysis of the request>"
    }
    """,
    handoffs=[decision_agent, escalation_agent],
    output_type=RoutingDecision
)


@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

# =====================================================
# AGENTS SDK INTEGRATION CLASS
# =====================================================


class AgenticLeaveSystemSDK:
    def __init__(self):
        self.triage_agent = triage_agent
        self.context_agent = context_agent
        self.executor = ThreadPoolExecutor(max_workers=3)
        with app.app_context():  # Ensure database session is initialized
            if not db.session:
                db.session = db.create_scoped_session()
                logger.info("Initialized new database session")

    def make_intelligent_decision(self, leave_request: LeaveRequest, team_members: List, additional_context: Dict = None) -> Dict:
        """
        Process leave request using OpenAI Agents SDK
        """
        try:
            logger.info(
                f"🤖 Starting Agents SDK analysis for user {leave_request.user_id}")

            # Run the async processing in the thread executor
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._async_process_request(
                        leave_request, team_members, additional_context)
                )
                # --- FIX: Ensure result is always a dict ---
                if isinstance(result, str):
                    logger.warning(f"Unexpected output type: {type(result)}")
                    result = {
                        'status': 'Escalate',
                        'reason': f"Unexpected AI response format: {result}",
                        'confidence': 0.5,
                        'escalate': True,
                        'agent_reasoning': f"Output: {result}",
                        'business_impact': 'unknown',
                        'employee_considerations': 'format_error'
                    }
                elif not isinstance(result, dict):
                    logger.warning(
                        f"Completely unexpected output type: {type(result)}")
                    result = {
                        'status': 'Escalate',
                        'reason': f"AI returned unsupported type: {type(result)}",
                        'confidence': 0.0,
                        'escalate': True,
                        'agent_reasoning': f"Output: {result}",
                        'business_impact': 'unknown',
                        'employee_considerations': 'format_error'
                    }
                return result

            finally:
                loop.close()

        except Exception as e:
            logger.exception(f"Error in Agents SDK processing: {str(e)}")
            return {
                'status': 'Escalate',
                'reason': f'System error during AI analysis: {str(e)}',
                'confidence': 0.0,
                'escalate': True,
                'agent_reasoning': f'SDK Error: {str(e)}',
                'business_impact': 'unknown',
                'employee_considerations': 'system_error'
            }

    async def _async_process_request(self, leave_request: LeaveRequest, team_members: List, additional_context: Dict = None) -> Dict:
        """Async processing of leave request with comprehensive validation"""

        # STEP 1: Run comprehensive validation first
        logger.info("🔍 Running comprehensive leave validation...")

        # Handle both string and datetime objects for dates
        if isinstance(leave_request.start_date, str):
            start_date_str = leave_request.start_date
            end_date_str = leave_request.end_date
        else:
            start_date_str = leave_request.start_date.strftime('%Y-%m-%d')
            end_date_str = leave_request.end_date.strftime('%Y-%m-%d')

        validation_result = get_comprehensive_leave_validation_impl(
            leave_request.user_id,
            start_date_str,
            end_date_str
        )

        # If validation fails, reject immediately
        if validation_result.get('should_reject', False):
            logger.warning(
                f"Leave request rejected due to validation failures: {validation_result.get('rejection_reasons', [])}")
            return {
                'status': 'Rejected',
                'reason': '; '.join(validation_result.get('rejection_reasons', ['Validation failed'])),
                'confidence': 1.0,  # High confidence in validation-based rejections
                'escalate': False,
                'agent_reasoning': f"Automated rejection due to validation failures: {validation_result}",
                'business_impact': 'none',
                'employee_considerations': 'validation_failure',
                'precedent_used': 'validation_rules',
                'recommended_actions': ['Review leave policy and reapply with valid dates'],
                'validation_details': validation_result
            }

        # STEP 2: Get user's team for context
        with app.app_context():
            user = User.query.get(leave_request.user_id)
            if not user:
                logger.error(f"User {leave_request.user_id} not found")
                raise ValueError(f"User {leave_request.user_id} not found")
            team_id = user.team_id if user else None

        # STEP 3: Get team analysis
        logger.info(f"👥 Getting detailed team analysis for team {team_id}")
        team_analysis_text = get_team_analysis_for_agent(
            team_id, start_date_str, end_date_str)

        # STEP 4: Format comprehensive input for the triage agent
        agent_input = f"""
        NEW LEAVE REQUEST FOR ANALYSIS:

        Employee Details:
        - User ID: {leave_request.user_id}
        - Team ID: {team_id}

        Leave Request:
        - Reason: {leave_request.reason}
        - Duration: {leave_request.duration_days} days
        - Start Date: {start_date_str}
        - End Date: {end_date_str}

        Team Context:
        - Team Size: {len(team_members)} members
        {team_analysis_text}

        Validation Results:
        - Overall Valid: {validation_result.get('overall_valid', True)}
        - Working Days: {validation_result.get('summary', {}).get('working_days', 0)}
        - Weekend Days: {validation_result.get('date_validation_result', {}).get('total_weekend_days', 0)}
        - Warnings: {len(validation_result.get('warnings', []))} warnings found
        - Special Approval Required: {validation_result.get('summary', {}).get('requires_special_approval', False)}
        - Manager Approval Required: {validation_result.get('summary', {}).get('requires_manager_approval', False)}

        Additional Context: {json.dumps(additional_context or {}, indent=2)}

        Please analyze this request and route to the appropriate agent.
        """

        logger.info("🚀 Sending request to Agents SDK triage system...")

        # STEP 5: Run the triage agent
        result = await Runner.run(self.triage_agent, agent_input)
        logger.info(f"Raw agent output: {result.final_output}")
        logger.info(f"🤖 Agents SDK completed processing")
        logger.info(f"📊 Result type: {type(result.final_output)}")

        # STEP 6: Handle the triage output
        if isinstance(result.final_output, RoutingDecision):
            routing_decision = result.final_output
        elif isinstance(result.final_output, str):
            try:
                # Parse JSON string to dict
                decision_dict = json.loads(result.final_output)
                # Convert dict to RoutingDecision object
                routing_decision = RoutingDecision(**decision_dict)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse JSON output: {str(e)}")
                return {
                    'status': 'Escalate',
                    'reason': f'Invalid JSON response format: {str(e)}',
                    'confidence': 0.5,
                    'escalate': True,
                    'agent_reasoning': f'Output: {result.final_output}',
                    'business_impact': 'unknown',
                    'employee_considerations': 'format_error',
                    'validation_details': validation_result
                }
        else:
            logger.warning(
                f"Unexpected output type: {type(result.final_output)}")
            return {
                'status': 'Escalate',
                'reason': f'Unexpected AI response type: {type(result.final_output)}',
                'confidence': 0.5,
                'escalate': True,
                'agent_reasoning': f'Output: {str(result.final_output)}',
                'business_impact': 'unknown',
                'employee_considerations': 'format_error',
                'validation_details': validation_result
            }

        # STEP 7: If triage indicates immediate escalation
        if routing_decision.escalate:
            return {
                'status': 'Escalate',
                'reason': routing_decision.reason,
                'confidence': routing_decision.confidence,
                'escalate': True,
                'agent_reasoning': routing_decision.analysis,
                'business_impact': 'unknown',
                'employee_considerations': 'triage_escalation',
                'precedent_used': None,
                'recommended_actions': [],
                'validation_details': validation_result
            }

        # STEP 8: Route to the appropriate agent
        target_agent = decision_agent if routing_decision.route_to == 'decision_agent' else escalation_agent
        logger.info(f"🚀 Routing to {routing_decision.route_to}...")

        # STEP 9: Run the target agent
        result = await Runner.run(target_agent, agent_input)
        logger.info(
            f"Raw {routing_decision.route_to} output: {result.final_output}")

        # STEP 10: Handle the final decision
        if isinstance(result.final_output, LeaveDecision):
            decision = result.final_output
        elif isinstance(result.final_output, str):
            try:
                decision_dict = json.loads(result.final_output)
                decision = LeaveDecision(**decision_dict)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(
                    f"Failed to parse JSON output from {routing_decision.route_to}: {str(e)}")
                return {
                    'status': 'Escalate',
                    'reason': f'Invalid JSON response format from {routing_decision.route_to}: {str(e)}',
                    'confidence': 0.5,
                    'escalate': True,
                    'agent_reasoning': f'Output: {result.final_output}',
                    'business_impact': 'unknown',
                    'employee_considerations': 'format_error',
                    'validation_details': validation_result
                }
        else:
            logger.warning(
                f"Unexpected output type from {routing_decision.route_to}: {type(result.final_output)}")
            return {
                'status': 'Escalate',
                'reason': f'Unexpected AI response type from {routing_decision.route_to}: {type(result.final_output)}',
                'confidence': 0.5,
                'escalate': True,
                'agent_reasoning': f'Output: {str(result.final_output)}',
                'business_impact': 'unknown',
                'employee_considerations': 'format_error',
                'validation_details': validation_result
            }

        # STEP 11: Convert LeaveDecision to dict for consistency
        final_result = {
            'status': decision.status,
            'reason': decision.reason,
            'confidence': decision.confidence,
            'escalate': decision.escalate,
            'agent_reasoning': decision.agent_reasoning,
            'business_impact': decision.business_impact,
            'employee_considerations': decision.employee_considerations,
            'precedent_used': decision.precedent_used,
            'recommended_actions': decision.recommended_actions,
            'validation_details': validation_result
        }

        # Add validation warnings to the final result if any
        if validation_result.get('warnings'):
            final_result['validation_warnings'] = validation_result['warnings']

        return final_result

# =====================================================
# FLASK ROUTES
# =====================================================


@app.route('/test-log')
def test_log():
    logger.info("This is a test log message from Agents SDK version")
    return "Check your log files - Agents SDK version active"


@app.cli.command("init-db")
def init_db():
    """Initialize the database with sample data"""
    with app.app_context():
        db.create_all()
        if not Team.query.first():
            # Create teams
            dev_team = Team(name="Development Team")
            hr_team = Team(name="HR Team")
            db.session.add_all([dev_team, hr_team])
            db.session.commit()

            # Create users
            admin = User(
                username='admin',
                password=generate_password_hash('admin123'),
                role='admin',
                team_id=dev_team.id
            )
            employee1 = User(
                username='employee',
                password=generate_password_hash('employee123'),
                role='user',
                team_id=dev_team.id
            )
            employee2 = User(
                username='john',
                password=generate_password_hash('john123'),
                role='user',
                team_id=dev_team.id
            )
            hr_user = User(
                username='hr_manager',
                password=generate_password_hash('hr123'),
                role='admin',
                team_id=hr_team.id
            )

            db.session.add_all([admin, employee1, employee2, hr_user])
            db.session.commit()

            # Add minimal sample leave history
            sample_leave = Leave(
                user_id=employee1.id,
                start_date=datetime(2024, 5, 1).date(),
                end_date=datetime(2024, 5, 2).date(),  # Reduced to 2 days
                reason="Sick leave - flu",
                status="Approved",
                decision_reason="Standard sick leave approval"
            )
            db.session.add(sample_leave)
            db.session.commit()

        print("Database initialized with teams, users, and minimal sample data.")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash(
                f'Welcome back! Now using AI Agents SDK for intelligent leave decisions.', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')


@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'admin':
        leaves = db.session.query(Leave).order_by(
            Leave.created_at.desc()).all()
    else:
        leaves = db.session.query(Leave).filter_by(user_id=current_user.id).order_by(
            Leave.created_at.desc()).all()
    return render_template('dashboard.html', leaves=leaves)


@app.route('/apply', methods=['GET', 'POST'])
@login_required
def apply_leave():
    if request.method == 'POST':
        try:
            logger.info(
                f"🚀 New Agents SDK leave request from user {current_user.id}")

            with app.app_context():
                # Validate team assignment
                if not current_user.team_id:
                    flash(
                        'You are not assigned to any team. Please contact HR.', 'error')
                    return redirect(url_for('dashboard'))

                team = Team.query.get(current_user.team_id)
                if not team:
                    flash(
                        'Your team information is invalid. Please contact HR.', 'error')
                    return redirect(url_for('dashboard'))

                logger.info(
                    f"Team context: {team.name} ({team.id}) with {len(team.members)} members")

                # Create leave request object
                leave_request = LeaveRequest(
                    reason=request.form['reason'],
                    start_date=request.form['start_date'],
                    end_date=request.form['end_date'],
                    user_id=current_user.id
                )

                # Process with Agents SDK
                agent_system = AgenticLeaveSystemSDK()
                decision = agent_system.make_intelligent_decision(
                    leave_request,
                    team.members,
                    {
                        'request_source': 'web_form',
                        'user_role': current_user.role,
                        'team_name': team.name
                    }
                )

                logger.info(f"🤖 Agents SDK Decision: {decision}")

                # Save to database
                new_leave = Leave(
                    user_id=current_user.id,
                    start_date=datetime.strptime(
                        request.form['start_date'], '%Y-%m-%d').date(),
                    end_date=datetime.strptime(
                        request.form['end_date'], '%Y-%m-%d').date(),
                    reason=request.form['reason'],
                    status=decision['status'],
                    decision_reason=decision.get(
                        'reason', 'No reason provided')
                )

                db.session.add(new_leave)
                db.session.commit()
                logger.info("✅ Leave saved to database")

                # Enhanced flash messages with Agents SDK insights
                confidence_text = f" (AI Confidence: {decision.get('confidence', 0):.0%})"

                flash_message = f"🤖 Leave {decision['status']}! {decision.get('reason', '')}{confidence_text}"

                # Add business impact info if available
                if decision.get('business_impact') and decision['business_impact'] != 'unknown':
                    flash_message += f" | Business Impact: {decision['business_impact']}"

                # Add employee considerations
                if decision.get('employee_considerations') and decision['employee_considerations'] != 'system_error':
                    flash_message += f" | Employee factors considered: {decision['employee_considerations']}"

                flash_type = 'success' if decision['status'] == 'Approved' else 'warning' if decision['status'] == 'Escalate' else 'info'
                flash(flash_message, flash_type)

                # Show reasoning in a separate message for transparency
                if decision.get('agent_reasoning'):
                    reasoning_msg = f"🧠 AI Reasoning: {decision['agent_reasoning'][:200]}..."
                    flash(reasoning_msg, 'info')

                return redirect(url_for('dashboard'))

        except ValueError as e:
            logger.error(f"Date format error: {str(e)}")
            flash('Invalid date format. Please use the date picker.', 'error')
        except Exception as e:
            db.session.rollback()
            logger.exception(f"Error processing leave request: {str(e)}")
            flash(
                f'Error processing your leave request: {str(e)}. Please contact HR.', 'error')
            return redirect(url_for('apply_leave'))

    return render_template('apply_leave.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))


@app.route('/test-agents-sdk')
@login_required
def test_agents_sdk():
    """Test the Agents SDK with various scenarios"""

    test_scenarios = [
        {
            'name': 'Standard Vacation Request',
            'reason': 'Annual vacation with family',
            'start_date': '2025-07-01',
            'end_date': '2025-07-05',
        },
        {
            'name': 'Extended Sick Leave',
            'reason': 'Medical procedure and recovery time needed',
            'start_date': '2025-06-20',
            'end_date': '2025-07-10',
        },
        {
            'name': 'Emergency Family Situation',
            'reason': 'Family emergency requiring immediate attention',
            'start_date': '2025-06-15',
            'end_date': '2025-06-22',
        },
        {
            'name': 'Mental Health Leave',
            'reason': 'Personal mental health and wellbeing break',
            'start_date': '2025-06-25',
            'end_date': '2025-07-02',
        }
    ]

    results = []
    with app.app_context():
        team = Team.query.get(
            current_user.team_id) if current_user.team_id else None

        if not team:
            return "Error: User not assigned to a team", 400

        agent_system = AgenticLeaveSystemSDK()

        for scenario in test_scenarios:
            try:
                leave_req = LeaveRequest(
                    reason=scenario['reason'],
                    start_date=scenario['start_date'],
                    end_date=scenario['end_date'],
                    user_id=current_user.id
                )

                decision = agent_system.make_intelligent_decision(
                    leave_req,
                    team.members,
                    {'test_scenario': scenario['name']}
                )

                results.append({
                    'scenario': scenario['name'],
                    'request': scenario,
                    'decision': decision
                })

            except Exception as e:
                results.append({
                    'scenario': scenario['name'],
                    'request': scenario,
                    'error': str(e)
                })

    # Format results as HTML
    html_output = "<h2>🤖 Agents SDK Test Results</h2>"
    for result in results:
        html_output += f"<h3>📋 {result['scenario']}</h3>"
        html_output += f"<pre>{json.dumps(result, indent=2, default=str)}</pre><hr>"

    return html_output


@app.route('/test-escalation')
@login_required
def test_escalation():
    """Test different scenarios that might trigger escalation"""
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('dashboard'))

    test_scenarios = [
        {
            'name': 'Extended Family Emergency',
            'reason': 'Emergency family situation requiring extended time',
            'start_date': '2025-06-15',
            'end_date': '2025-07-15',
            'expected_outcome': 'Escalate - Extended duration'
        },
        {
            'name': 'Chronic Medical Leave',
            'reason': 'Sick leave - chronic condition flare-up',
            'start_date': '2025-06-20',
            'end_date': '2025-06-27',
            'expected_outcome': 'Escalate or Approve - Medical consideration'
        },
        {
            'name': 'Mental Health Personal Leave',
            'reason': 'Personal leave for mental health and wellbeing',
            'start_date': '2025-06-25',
            'end_date': '2025-07-02',
            'expected_outcome': 'Context dependent - Employee wellbeing priority'
        },
        {
            'name': 'Medical Procedure Extended Leave',
            'reason': 'Scheduled medical procedure requiring recovery time',
            'start_date': '2025-07-01',
            'end_date': '2025-07-28',
            'expected_outcome': 'Escalate - Extended medical leave'
        },
        {
            'name': 'Bereavement Leave Extended',
            'reason': 'Bereavement leave for close family member',
            'start_date': '2025-06-18',
            'end_date': '2025-06-30',
            'expected_outcome': 'Escalate - Extended bereavement consideration'
        },
        {
            'name': 'Quarter-End Critical Period Request',
            'reason': 'Personal vacation during busy period',
            'start_date': '2025-09-25',
            'end_date': '2025-09-30',
            'expected_outcome': 'Escalate - Business critical timing'
        }
    ]

    results = []

    with app.app_context():
        team = Team.query.filter_by(id=current_user.team_id).first()
        if not team:
            return "Error: User not assigned to a team.", 400

        agent_system = AgenticLeaveSystemSDK()

        logger.info(
            f"🧪 Running escalation tests for {len(test_scenarios)} scenarios")

        for scenario in test_scenarios:
            try:
                logger.info(f"Testing scenario: {scenario['name']}")

                leave_request = LeaveRequest(
                    reason=scenario['reason'],
                    start_date=scenario['start_date'],
                    end_date=scenario['end_date'],
                    user_id=current_user.id
                )

                decision = agent_system.make_intelligent_decision(
                    leave_request,
                    team.members,
                    {
                        'test_scenario': scenario['name'],
                        'escalation_test': True,
                        'expected_outcome': scenario['expected_outcome']
                    }
                )

                escalation_triggered = decision.get(
                    'escalate', False) or decision.get('status') == 'Escalate'

                results.append({
                    'scenario_name': scenario['name'],
                    'request_details': {
                        'reason': scenario['reason'],
                        'duration_days': leave_request.duration_days,
                        'start_date': scenario['start_date'],
                        'end_date': scenario['end_date']
                    },
                    'expected_outcome': scenario['expected_outcome'],
                    'actual_decision': {
                        'status': decision.get('status'),
                        'escalate': decision.get('escalate', False),
                        'confidence': decision.get('confidence', 0),
                        'reason': decision.get('reason', 'No reason provided'),
                        'business_impact': decision.get('business_impact', 'unknown'),
                        'employee_considerations': decision.get('employee_considerations', 'none')
                    },
                    'escalation_triggered': escalation_triggered,
                    'agent_reasoning': decision.get('agent_reasoning', 'No reasoning provided'),
                    'test_result': 'PASS' if escalation_triggered else 'REVIEW_NEEDED'
                })

            except Exception as e:
                logger.error(
                    f"Error testing scenario {scenario['name']}: {str(e)}")
                results.append({
                    'scenario_name': scenario['name'],
                    'request_details': scenario,
                    'error': str(e),
                    'test_result': 'ERROR'
                })

    # Generate summary statistics
    total_tests = len(results)
    escalations_triggered = len(
        [r for r in results if r.get('escalation_triggered', False)])
    errors = len([r for r in results if 'error' in r])

    summary = {
        'total_scenarios_tested': total_tests,
        'escalations_triggered': escalations_triggered,
        'escalation_rate': f"{(escalations_triggered/total_tests)*100:.1f}%" if total_tests > 0 else "0%",
        'errors_encountered': errors,
        'test_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tester_user_id': current_user.id,
        'team_context': team.name
    }

    # Format results as HTML
    html_output = f"""
    <div style="font-family: Arial, sans-serif; margin: 20px;">
        <h2>🚨 Escalation Testing Results</h2>

        <div style="background: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <h3>📊 Test Summary</h3>
            <ul>
                <li><strong>Total Scenarios:</strong> {summary['total_scenarios_tested']}</li>
                <li><strong>Escalations Triggered:</strong> {summary['escalations_triggered']}</li>
                <li><strong>Escalation Rate:</strong> {summary['escalation_rate']}</li>
                <li><strong>Errors:</strong> {summary['errors_encountered']}</li>
                <li><strong>Test Time:</strong> {summary['test_timestamp']}</li>
                <li><strong>Team Context:</strong> {summary['team_context']}</li>
            </ul>
        </div>
    """

    for i, result in enumerate(results, 1):
        status_color = '#28a745' if result.get(
            'escalation_triggered') else '#ffc107'
        if 'error' in result:
            status_color = '#dc3545'

        html_output += f"""
        <div style="border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px;">
            <h3 style="color: {status_color};">#{i}. {result['scenario_name']} 
                <span style="font-size: 0.8em; background: {status_color}; color: white; padding: 2px 8px; border-radius: 3px;">
                    {result.get('test_result', 'UNKNOWN')}
                </span>
            </h3>

            <div style="display: flex; gap: 20px;">
                <div style="flex: 1;">
                    <h4>📋 Request Details:</h4>
                    <ul>
                        <li><strong>Reason:</strong> {result.get('request_details', {}).get('reason', 'N/A')}</li>
                        <li><strong>Duration:</strong> {result.get('request_details', {}).get('duration_days', 'N/A')} days</li>
                        <li><strong>Dates:</strong> {result.get('request_details', {}).get('start_date', 'N/A')} to {result.get('request_details', {}).get('end_date', 'N/A')}</li>
                    </ul>
                </div>

                <div style="flex: 1;">
                    <h4>🤖 AI Decision:</h4>
                    <ul>
                        <li><strong>Status:</strong> {result.get('actual_decision', {}).get('status', 'N/A')}</li>
                        <li><strong>Escalate:</strong> {result.get('actual_decision', {}).get('escalate', False)}</li>
                        <li><strong>Confidence:</strong> {result.get('actual_decision', {}).get('confidence', 0):.1%}</li>
                        <li><strong>Business Impact:</strong> {result.get('actual_decision', {}).get('business_impact', 'N/A')}</li>
                    </ul>
                </div>
            </div>

            <div style="margin-top: 10px;">
                <h4>💭 Decision Reasoning:</h4>
                <p style="background: #f8f9fa; padding: 10px; border-radius: 3px; font-style: italic;">
                    "{result.get('agent_reasoning', 'No reasoning provided')}"
                </p>
            </div>

            <div style="margin-top: 10px;">
                <h4>🎯 Expected vs Actual:</h4>
                <p><strong>Expected:</strong> {result.get('expected_outcome', 'N/A')}</p>
                <p><strong>Actual Decision Reason:</strong> {result.get('actual_decision', {}).get('reason', 'N/A')}</p>
            </div>

            {f'<div style="color: red; margin-top: 10px;"><strong>❌ Error:</strong> {result["error"]}</div>' if 'error' in result else ''}
        </div>
        """

    html_output += """
        <div style="margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 5px;">
            <h3>📝 Notes:</h3>
            <ul>
                <li>Escalation scenarios test the AI's ability to identify complex situations requiring human review</li>
                <li>High confidence escalations indicate the AI is correctly identifying edge cases</li>
                <li>Review scenarios marked as "REVIEW_NEEDED" to ensure appropriate escalation triggers</li>
            </ul>
        </div>
    </div>
    """

    logger.info(f"🧪 Escalation test completed: {summary}")
    return html_output


@app.route('/agent-analytics')
@login_required
def agent_analytics():
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('dashboard'))

    with app.app_context():
        recent_leaves = Leave.query.order_by(
            Leave.created_at.desc()).limit(50).all()

        analytics = {
            'total_requests': len(recent_leaves),
            'approved_count': len([l for l in recent_leaves if l.status == 'Approved']),
            'denied_count': len([l for l in recent_leaves if l.status == 'Denied']),
            'escalated_count': len([l for l in recent_leaves if l.status == 'Escalate']),
            'pending_count': len([l for l in recent_leaves if l.status == 'Pending']),
        }

        if analytics['total_requests'] > 0:
            analytics['approval_rate'] = (
                analytics['approved_count'] / analytics['total_requests']) * 100
            analytics['escalation_rate'] = (
                analytics['escalated_count'] / analytics['total_requests']) * 100
            analytics['denial_rate'] = (
                analytics['denied_count'] / analytics['total_requests']) * 100
        else:
            analytics['approval_rate'] = 0
            analytics['escalation_rate'] = 0
            analytics['denial_rate'] = 0

        reason_patterns = {}
        for leave in recent_leaves:
            reason_key = leave.reason.lower()
            if 'sick' in reason_key:
                category = 'Sick Leave'
            elif 'vacation' in reason_key or 'holiday' in reason_key:
                category = 'Vacation'
            elif 'personal' in reason_key:
                category = 'Personal'
            elif 'emergency' in reason_key or 'family' in reason_key:
                category = 'Emergency/Family'
            elif 'medical' in reason_key:
                category = 'Medical'
            else:
                category = 'Other'

            if category not in reason_patterns:
                reason_patterns[category] = {
                    'count': 0, 'approved': 0, 'denied': 0, 'escalated': 0}

            reason_patterns[category]['count'] += 1
            if leave.status == 'Approved':
                reason_patterns[category]['approved'] += 1
            elif leave.status == 'Denied':
                reason_patterns[category]['denied'] += 1
            elif leave.status == 'Escalate':
                reason_patterns[category]['escalated'] += 1

        team_analytics = {}
        teams = Team.query.all()
        for team in teams:
            team_leaves = [
                l for l in recent_leaves if l.user.team_id == team.id]
            if team_leaves:
                team_analytics[team.name] = {
                    'total_requests': len(team_leaves),
                    'approved': len([l for l in team_leaves if l.status == 'Approved']),
                    'denied': len([l for l in team_leaves if l.status == 'Denied']),
                    'escalated': len([l for l in team_leaves if l.status == 'Escalate']),
                    'avg_duration': sum([(l.end_date - l.start_date).days + 1 for l in team_leaves]) / len(team_leaves)
                }

        duration_analytics = {
            'short_term': len([l for l in recent_leaves if (l.end_date - l.start_date).days <= 2]),
            'medium_term': len([l for l in recent_leaves if 3 <= (l.end_date - l.start_date).days <= 7]),
            'long_term': len([l for l in recent_leaves if (l.end_date - l.start_date).days > 7]),
        }

        thirty_days_ago = datetime.now().date() - timedelta(days=30)
        sixty_days_ago = datetime.now().date() - timedelta(days=60)

        recent_month = [l for l in recent_leaves if l.created_at.date()
                        >= thirty_days_ago]
        previous_month = [l for l in recent_leaves if sixty_days_ago <=
                          l.created_at.date() < thirty_days_ago]

        trends = {
            'current_month_requests': len(recent_month),
            'previous_month_requests': len(previous_month),
            'trend_direction': 'up' if len(recent_month) > len(previous_month) else 'down' if len(recent_month) < len(previous_month) else 'stable'
        }

        if len(previous_month) > 0:
            trends['percentage_change'] = (
                (len(recent_month) - len(previous_month)) / len(previous_month)) * 100
        else:
            trends['percentage_change'] = 0

        decision_reasons = {}
        for leave in recent_leaves:
            if leave.decision_reason:
                reason = leave.decision_reason[:50] + "..." if len(
                    leave.decision_reason) > 50 else leave.decision_reason
                decision_reasons[reason] = decision_reasons.get(reason, 0) + 1

        top_decision_reasons = sorted(
            decision_reasons.items(), key=lambda x: x[1], reverse=True)[:5]

        return render_template('analytics.html',
                               analytics=analytics,
                               reason_patterns=reason_patterns,
                               team_analytics=team_analytics,
                               duration_analytics=duration_analytics,
                               trends=trends,
                               top_decision_reasons=top_decision_reasons,
                               recent_leaves=recent_leaves[:10])


@app.route('/ai-performance')
@login_required
def ai_performance():
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('dashboard'))

    with app.app_context():
        ai_processed_leaves = Leave.query.filter(
            Leave.decision_reason.like('%AI%') |
            Leave.decision_reason.apply('%agent%') |
            Leave.decision_reason.apply('%confidence%')
        ).order_by(Leave.created_at.desc()).limit(100).all()

        performance_metrics = {
            'total_ai_decisions': len(ai_processed_leaves),
            'avg_processing_time': 'N/A',
            'escalation_accuracy': 0,
            'user_satisfaction': 'N/A',
        }

        ai_decision_patterns = {
            'auto_approved': len([l for l in ai_processed_leaves if l.status == 'Approved']),
            'auto_denied': len([l for l in ai_processed_leaves if l.status == 'Denied']),
            'escalated_to_human': len([l for l in ai_processed_leaves if l.status == 'Escalate']),
        }

        confidence_scores = []
        for leave in ai_processed_leaves:
            if 'confidence' in leave.decision_reason.lower():
                import re
                match = re.search(r'(\d+)%', leave.decision_reason)
                if match:
                    confidence_scores.append(int(match.group(1)))

        avg_confidence = sum(confidence_scores) / \
            len(confidence_scores) if confidence_scores else 0

        return render_template('ai_performance.html',
                               performance_metrics=performance_metrics,
                               ai_decision_patterns=ai_decision_patterns,
                               avg_confidence=avg_confidence,
                               confidence_scores=confidence_scores,
                               ai_processed_leaves=ai_processed_leaves[:20])


@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    logger.error(f"Internal server error: {error}")
    return render_template('500.html'), 500


@app.route('/health')
def health_check():
    try:
        with app.app_context():
            db.session.execute(text('SELECT 1'))  # Wrap SQL in text()
            agent_system = AgenticLeaveSystemSDK()

        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': 'connected',
            'agents_sdk': 'initialized',
            'version': 'agents-sdk-v1.0'
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, 500


@app.route('/dev/reset-db')
def reset_database():
    if not app.debug:
        return "This endpoint is only available in debug mode", 403

    try:
        with app.app_context():
            db.drop_all()
            db.create_all()

            dev_team = Team(name="Development Team")
            hr_team = Team(name="HR Team")
            qa_team = Team(name="QA Team")
            db.session.add_all([dev_team, hr_team, qa_team])
            db.session.commit()

            users = [
                User(username='admin', password=generate_password_hash(
                    'admin123'), role='admin', team_id=dev_team.id),
                User(username='employee', password=generate_password_hash(
                    'employee123'), role='user', team_id=dev_team.id),
                User(username='john', password=generate_password_hash(
                    'john123'), role='user', team_id=dev_team.id),
                User(username='jane', password=generate_password_hash(
                    'jane123'), role='user', team_id=qa_team.id),
                User(username='hr_manager', password=generate_password_hash(
                    'hr123'), role='admin', team_id=hr_team.id),
            ]
            db.session.add_all(users)
            db.session.commit()

            sample_leaves = [
                Leave(user_id=2, start_date=datetime(2024, 5, 1).date(), end_date=datetime(2024, 5, 2).date(),
                      reason="Sick leave - flu", status="Approved", decision_reason="Standard sick leave approval"),
                Leave(user_id=2, start_date=datetime(2024, 4, 15).date(), end_date=datetime(2024, 4, 17).date(),
                      reason="Vacation with family", status="Approved", decision_reason="AI Decision: Low business impact"),
                Leave(user_id=3, start_date=datetime(2024, 5, 10).date(), end_date=datetime(2024, 5, 15).date(),
                      reason="Medical procedure", status="Escalate", decision_reason="AI Escalation: Extended medical leave"),
                Leave(user_id=4, start_date=datetime(2024, 5, 5).date(), end_date=datetime(2024, 5, 6).date(),
                      reason="Personal emergency", status="Approved", decision_reason="AI Decision: Emergency situation"),
            ]
            db.session.add_all(sample_leaves)
            db.session.commit()

        return "Database reset successfully with sample data"
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        return f"Database reset failed: {str(e)}", 500


def initialize_app():
    try:
        with app.app_context():
            db.create_all()
            logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


if __name__ == '__main__':
    log_dir = os.path.dirname(os.path.abspath('app.log'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    try:
        initialize_app()
    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        print(f"❌ Application initialization failed: {e}")
        exit(1)

    with app.app_context():
        try:
            if not User.query.first():
                logger.info(
                    "📊 No existing data found, creating sample data...")

                dev_team = Team(name="Development Team")
                hr_team = Team(name="HR Team")
                qa_team = Team(name="QA Team")
                db.session.add_all([dev_team, hr_team, qa_team])
                db.session.commit()

                users = [
                    User(username='admin', password=generate_password_hash('admin123'),
                         role='admin', team_id=dev_team.id),
                    User(username='employee', password=generate_password_hash('employee123'),
                         role='user', team_id=dev_team.id),
                    User(username='john', password=generate_password_hash('john123'),
                         role='user', team_id=dev_team.id),
                    User(username='jane', password=generate_password_hash('jane123'),
                         role='user', team_id=qa_team.id),
                    User(username='hr_manager', password=generate_password_hash('hr123'),
                         role='admin', team_id=hr_team.id),
                ]
                db.session.add_all(users)
                db.session.commit()

                logger.info("✅ Sample data created successfully")
                print("\n" + "="*60)
                print("🎉 LEAVE MANAGEMENT SYSTEM - AGENTS SDK VERSION")
                print("="*60)
                print("📋 Sample Login Credentials:")
                print("   Admin: admin/admin123")
                print("   Employee: employee/employee123")
                print("   HR Manager: hr_manager/hr123")
                print("="*60)
                print("🤖 AI Features Available:")
                print("   • Intelligent leave decision making")
                print("   • Automatic escalation detection")
                print("   • Business impact analysis")
                print("   • Team availability assessment")
                print("   • Historical pattern analysis")
                print("="*60)
                print("🔲 Test Endpoints:")
                print("   • /test-agents-sdk - Test AI scenarios")
                print("   • /test-escalation - Test escalation logic")
                print("   • /agent-analytics - View AI analytics")
                print("   • /ai-performance - AI performance metrics")
                print("   • /health - System health check")
                print("="*60)
            else:
                logger.info(
                    "📊 Existing data found, skipping sample data creation")

        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            print(f"❌ Database initialization failed: {e}")

    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

    try:
        print(f"\n🌐 Starting server on http://127.0.0.1:5000")
        print("🛑 Press Ctrl+C to stop the server")

        app.run(
            debug=True,
            host='127.0.0.1',
            port=5000,
            threaded=True,
            use_reloader=True,
            use_debugger=True
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        logger.error(f"Server startup error: {e}")
        print(f"❌ Server failed to start: {e}")
    finally:
        logger.info("🧹 Cleaning up resources...")
        db.session.close()
