from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Leave, Team, LeaveTransaction, LeaveType, LeaveBalance, Organization, Department
from datetime import datetime, timedelta, date
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
from difflib import SequenceMatcher
import re
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
    """Retrieve and analyze user's leave history patterns with enhanced sick leave detection"""
    try:
        logger.info(f"🔍 Analyzing leave history for user {user_id}")

        with app.app_context():
            # Get ALL leave requests in the last 12 months (not just 10)
            cutoff_date = datetime.utcnow() - timedelta(days=365)
            # Convert to date for comparison if needed
            cutoff_date_only = cutoff_date.date()

            all_leaves = Leave.query.filter(
                Leave.user_id == user_id,
                Leave.start_date >= cutoff_date_only
            ).order_by(Leave.start_date.desc()).all()

            # Get recent leave requests (last 3 months for pattern detection)
            recent_cutoff = datetime.utcnow() - timedelta(days=90)
            recent_cutoff_date_only = recent_cutoff.date()

            recent_leaves = [
                leave for leave in all_leaves if leave.start_date >= recent_cutoff_date_only]

            if all_leaves:
                # Calculate overall statistics
                total_days = sum(
                    [(leave.end_date - leave.start_date).days + 1 for leave in all_leaves])
                total_requests = len(all_leaves)
                approved_count = len(
                    [leave for leave in all_leaves if leave.status == 'Approved'])

                # Sick leave analysis - Enhanced detection
                sick_leaves = [
                    leave for leave in all_leaves if 'sick' in leave.reason.lower()]
                recent_sick_leaves = [
                    leave for leave in recent_leaves if 'sick' in leave.reason.lower()]

                # Calculate sick leave patterns
                sick_count_total = len(sick_leaves)
                sick_count_recent = len(recent_sick_leaves)
                sick_days_total = sum(
                    [(leave.end_date - leave.start_date).days + 1 for leave in sick_leaves])
                sick_days_recent = sum(
                    [(leave.end_date - leave.start_date).days + 1 for leave in recent_sick_leaves])

                # Frequency analysis
                if recent_sick_leaves:
                    days_between_sick_leaves = []
                    for i in range(len(recent_sick_leaves) - 1):
                        # Ensure we're working with date objects for calculation
                        start_date = recent_sick_leaves[i].start_date
                        end_date = recent_sick_leaves[i+1].end_date

                        # Convert to date objects if they're datetime objects
                        if hasattr(start_date, 'date'):
                            start_date = start_date.date()
                        if hasattr(end_date, 'date'):
                            end_date = end_date.date()

                        diff = (start_date - end_date).days
                        days_between_sick_leaves.append(diff)
                    avg_gap_between_sick = sum(days_between_sick_leaves) / len(
                        days_between_sick_leaves) if days_between_sick_leaves else 0
                else:
                    avg_gap_between_sick = 0

                # Pattern classification with stricter thresholds
                pattern_flags = []
                risk_score = 0

                # Flag 1: Excessive sick leave frequency (>4 in 3 months)
                if sick_count_recent > 4:
                    pattern_flags.append("excessive_sick_frequency")
                    risk_score += 30

                # Flag 2: High sick leave percentage (>60% of all leaves are sick)
                sick_percentage = (
                    sick_count_total / total_requests) * 100 if total_requests > 0 else 0
                if sick_percentage > 60:
                    pattern_flags.append("high_sick_percentage")
                    risk_score += 25

                # Flag 3: Short gaps between sick leaves (<7 days average)
                if recent_sick_leaves and avg_gap_between_sick < 7:
                    pattern_flags.append("frequent_sick_pattern")
                    risk_score += 20

                # Flag 4: Excessive total sick days (>15 days in 3 months)
                if sick_days_recent > 15:
                    pattern_flags.append("excessive_sick_days")
                    risk_score += 25

                # Flag 5: Monday/Friday pattern detection
                monday_friday_sick = []
                for leave in recent_sick_leaves:
                    start_date = leave.start_date
                    # Convert to date object if it's a datetime object
                    if hasattr(start_date, 'date'):
                        start_date = start_date.date()
                    # 0=Monday, 4=Friday
                    if start_date.weekday() in [0, 4]:
                        monday_friday_sick.append(leave)

                if len(monday_friday_sick) >= 3:
                    pattern_flags.append("weekend_adjacent_pattern")
                    risk_score += 15

                # Flag 6: Single day sick leaves (potential abuse)
                single_day_sick = [leave for leave in recent_sick_leaves
                                   if (leave.end_date - leave.start_date).days == 0]
                if len(single_day_sick) >= 4:
                    pattern_flags.append("frequent_single_day_sick")
                    risk_score += 20

                # Determine overall pattern classification
                if risk_score >= 50:
                    pattern = "high_risk_abuse_pattern"
                elif risk_score >= 30:
                    pattern = "concerning_sick_leave_pattern"
                elif risk_score >= 15:
                    pattern = "elevated_sick_leave_usage"
                elif sick_count_recent > 2:
                    pattern = "moderate_sick_leave_usage"
                else:
                    pattern = "normal"

                # Generate recommendations based on pattern
                recommendations = []
                if "excessive_sick_frequency" in pattern_flags:
                    recommendations.append("require_medical_certification")
                if "frequent_sick_pattern" in pattern_flags:
                    recommendations.append("manager_review_required")
                if "weekend_adjacent_pattern" in pattern_flags:
                    recommendations.append("pattern_verification_needed")
                if risk_score >= 50:
                    recommendations.append("escalate_to_hr")
                    recommendations.append("possible_disciplinary_review")

                # Handle last_leave_date safely
                last_leave_date = 'N/A'
                if all_leaves and all_leaves[0].end_date:
                    end_date = all_leaves[0].end_date
                    if hasattr(end_date, 'strftime'):
                        last_leave_date = end_date.strftime('%Y-%m-%d')
                    else:
                        last_leave_date = str(end_date)

                return {
                    'total_recent_days': sum([(leave.end_date - leave.start_date).days + 1 for leave in recent_leaves]),
                    'total_requests': len(recent_leaves),
                    'total_annual_requests': total_requests,
                    'sick_leave_count_recent': sick_count_recent,
                    'sick_leave_count_total': sick_count_total,
                    'sick_days_recent': sick_days_recent,
                    'sick_days_total': sick_days_total,
                    'sick_leave_percentage': round(sick_percentage, 2),
                    'approval_rate': (approved_count / total_requests) * 100 if total_requests else 0,
                    'average_gap_between_sick_leaves': round(avg_gap_between_sick, 1),
                    'pattern_classification': pattern,
                    'pattern_flags': pattern_flags,
                    'risk_score': risk_score,
                    'recommendations': recommendations,
                    'last_leave_date': last_leave_date,
                    'recent_sick_reasons': [leave.reason for leave in recent_sick_leaves[:5]],
                    'single_day_sick_count': len(single_day_sick),
                    'monday_friday_sick_count': len(monday_friday_sick)
                }
            else:
                return {
                    'total_recent_days': 0,
                    'total_requests': 0,
                    'pattern_classification': 'new_employee',
                    'risk_score': 0,
                    'message': 'No previous leave history found'
                }

    except Exception as e:
        logger.error(f"Error analyzing user history: {str(e)}")
        return {'error': str(e), 'pattern_classification': 'unknown', 'risk_score': 100}


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
def analyze_leave_balance(user_id: int, leave_type_id: int, days_requested: float, year: int = None) -> Dict:
    """Analyze leave balance for a specific user and leave type"""
    try:
        logger.info(
            f"💰 Analyzing leave balance for user {user_id}, leave type {leave_type_id}")

        with app.app_context():
            # Get current year if not specified
            if year is None:
                year = datetime.now().year

            # Get user
            user = User.query.get(user_id)
            if not user:
                logger.error(f"User {user_id} not found")
                return {
                    'error': 'User not found',
                    'can_take_leave': False,
                    'balance_status': 'error'
                }

            # Get leave type
            leave_type = LeaveType.query.get(leave_type_id)
            if not leave_type:
                logger.error(f"Leave type {leave_type_id} not found")
                return {
                    'error': 'Leave type not found',
                    'can_take_leave': False,
                    'balance_status': 'error'
                }

            # Get or create leave balance for this user, leave type, and year
            leave_balance = LeaveBalance.query.filter_by(
                user_id=user_id,
                leave_type_id=leave_type_id,
                year=year
            ).first()

            if not leave_balance:
                # Initialize leave balance if it doesn't exist
                logger.info(
                    f"Creating new leave balance for user {user_id}, leave type {leave_type_id}")
                leave_balance = LeaveBalance(
                    user_id=user_id,
                    leave_type_id=leave_type_id,
                    year=year,
                    allocated_days=leave_type.default_allocation
                )
                db.session.add(leave_balance)
                db.session.commit()

            # Calculate current balance
            current_balance = leave_balance.remaining_days
            balance_after_request = current_balance - days_requested

            logger.info(
                f"Current balance: {current_balance}, After request: {balance_after_request}")

            # Determine if leave can be taken
            can_take_leave = False
            balance_status = ""
            warning_message = ""

            if balance_after_request >= 0:
                # Sufficient balance
                can_take_leave = True
                balance_status = "sufficient"
            elif leave_type.can_be_negative:
                # Negative balance allowed
                can_take_leave = True
                balance_status = "negative_allowed"
                warning_message = f"This will result in a negative balance of {abs(balance_after_request)} days"
            else:
                # Insufficient balance and negative not allowed
                can_take_leave = False
                balance_status = "insufficient"
                warning_message = f"Insufficient balance. You need {abs(balance_after_request)} more days"

            # Calculate usage statistics
            usage_percentage = (leave_balance.used_days / leave_balance.total_available) * \
                100 if leave_balance.total_available > 0 else 0
            pending_percentage = (leave_balance.pending_days / leave_balance.total_available) * \
                100 if leave_balance.total_available > 0 else 0

            # Determine risk level
            risk_level = "low"
            if balance_after_request < 0 and not leave_type.can_be_negative:
                risk_level = "critical"
            elif balance_after_request < 0:
                risk_level = "high"  # Negative but allowed
            elif balance_after_request <= leave_balance.total_available * 0.1:
                risk_level = "medium"  # Less than 10% remaining

            logger.info(
                f"Balance analysis complete - Can take leave: {can_take_leave}, Status: {balance_status}")

            return {
                'user_id': user_id,
                'username': user.username,
                'leave_type_name': leave_type.name,
                'leave_type_code': leave_type.code,
                'year': year,
                'days_requested': days_requested,
                'can_take_leave': can_take_leave,
                'balance_status': balance_status,
                'warning_message': warning_message,
                'risk_level': risk_level,
                'balance_details': {
                    'current_balance': current_balance,
                    'balance_after_request': balance_after_request,
                    'allocated_days': leave_balance.allocated_days,
                    'used_days': leave_balance.used_days,
                    'pending_days': leave_balance.pending_days,
                    'carried_over': leave_balance.carried_over,
                    'total_available': leave_balance.total_available
                },
                'leave_type_settings': {
                    'negative_allowed': leave_type.can_be_negative,
                    'max_carryover': leave_type.max_carryover,
                    'requires_approval': leave_type.requires_approval
                },
                'usage_statistics': {
                    'usage_percentage': round(usage_percentage, 2),
                    'pending_percentage': round(pending_percentage, 2),
                    'remaining_percentage': round((current_balance / leave_balance.total_available) * 100, 2) if leave_balance.total_available > 0 else 0
                }
            }

    except Exception as e:
        logger.error(f"Error analyzing leave balance: {str(e)}")
        return {
            'error': str(e),
            'can_take_leave': False,
            'balance_status': 'error',
            'risk_level': 'critical'
        }


@function_tool
def get_comprehensive_leave_analysis(user_id: int, leave_type_id: int, days_requested: float,
                                     start_date: str, end_date: str, year: int = None) -> Dict:
    """Get comprehensive leave analysis including balance and team impact"""
    try:
        logger.info(
            f"🔍 Performing comprehensive leave analysis for user {user_id}")

        with app.app_context():
            # Get user to determine team
            user = User.query.get(user_id)
            if not user:
                return {
                    'error': 'User not found',
                    'analysis_complete': False
                }

            # Analyze leave balance
            balance_analysis = analyze_leave_balance(
                user_id, leave_type_id, days_requested, year)

            # Analyze team availability
            team_analysis = analyze_team_availability(
                user.team_id, start_date, end_date)

            # Combine analyses for overall recommendation
            overall_recommendation = "approved"
            blocking_factors = []

            # Check balance issues
            if not balance_analysis.get('can_take_leave', False):
                overall_recommendation = "rejected"
                blocking_factors.append("Insufficient leave balance")
            elif balance_analysis.get('risk_level') == 'high':
                overall_recommendation = "conditional"
                blocking_factors.append(
                    "Leave request results in negative balance")

            # Check team impact
            if team_analysis.get('impact_level') == 'critical':
                if overall_recommendation == "approved":
                    overall_recommendation = "conditional"
                blocking_factors.append("Critical impact on team availability")
            elif team_analysis.get('impact_level') == 'high':
                if overall_recommendation == "approved":
                    overall_recommendation = "review_required"
                blocking_factors.append("High impact on team availability")

            # Calculate overall risk score (1-10)
            risk_score = 1
            if balance_analysis.get('risk_level') == 'critical':
                risk_score += 4
            elif balance_analysis.get('risk_level') == 'high':
                risk_score += 3
            elif balance_analysis.get('risk_level') == 'medium':
                risk_score += 2

            if team_analysis.get('impact_level') == 'critical':
                risk_score += 4
            elif team_analysis.get('impact_level') == 'high':
                risk_score += 2
            elif team_analysis.get('impact_level') == 'medium':
                risk_score += 1

            risk_score = min(risk_score, 10)  # Cap at 10

            logger.info(
                f"Comprehensive analysis complete - Recommendation: {overall_recommendation}")

            return {
                'analysis_complete': True,
                'user_info': {
                    'user_id': user_id,
                    'username': user.username,
                    'team_name': user.user_team.name if user.user_team else 'No Team'
                },
                'leave_request': {
                    'days_requested': days_requested,
                    'start_date': start_date,
                    'end_date': end_date
                },
                'balance_analysis': balance_analysis,
                'team_analysis': team_analysis,
                'overall_assessment': {
                    'recommendation': overall_recommendation,
                    'risk_score': risk_score,
                    'blocking_factors': blocking_factors,
                    'approval_likelihood': self._calculate_approval_likelihood(overall_recommendation, risk_score)
                }
            }

    except Exception as e:
        logger.error(f"Error in comprehensive leave analysis: {str(e)}")
        return {
            'error': str(e),
            'analysis_complete': False
        }


def _calculate_approval_likelihood(recommendation: str, risk_score: int) -> str:
    """Calculate likelihood of approval based on recommendation and risk score"""
    if recommendation == "approved" and risk_score <= 3:
        return "very_high"
    elif recommendation == "approved" and risk_score <= 5:
        return "high"
    elif recommendation in ["conditional", "review_required"] and risk_score <= 6:
        return "medium"
    elif recommendation in ["conditional", "review_required"]:
        return "low"
    else:
        return "very_low"


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
def get_similar_leave_decisions(reason: str, duration_days: int, user_id: int,
                                team_id: int = None, leave_start_date: str = None) -> Dict:
    """Find similar past leave decisions with enhanced similarity analysis"""
    try:
        # Enhanced input validation
        if not reason or not isinstance(duration_days, int) or duration_days <= 0:
            return {
                'similar_cases_found': 0,
                'precedents': [],
                'historical_approval_rate': 0,
                'recommendation': 'Invalid input parameters',
                'confidence_score': 0
            }

        logger.info(
            f"🔍 Enhanced precedent analysis for: {reason}, {duration_days} days, user {user_id}")

        with app.app_context():
            # Get user details for better matching
            current_user = User.query.get(user_id)
            user_role = getattr(current_user, 'role',
                                'employee') if current_user else 'employee'
            user_team_id = getattr(
                current_user, 'team_id', team_id) if current_user else team_id

            # Build more sophisticated query
            query = Leave.query.filter(Leave.user_id != user_id)

            # Add team context if available
            if user_team_id:
                team_members = User.query.filter_by(team_id=user_team_id).all()
                team_user_ids = [u.id for u in team_members]
                # Include both team members and similar roles
                query = query.filter(
                    db.or_(
                        Leave.user_id.in_(team_user_ids),
                        User.role == user_role
                    )
                ).join(User, Leave.user_id == User.id)

            # Get broader set for analysis
            all_leaves = query.limit(50).all()

            if not all_leaves:
                return {
                    'similar_cases_found': 0,
                    'precedents': [],
                    'historical_approval_rate': 0,
                    'recommendation': 'No historical data available',
                    'confidence_score': 0
                }

            # Enhanced similarity analysis
            precedents = []
            reason_keywords = extract_keywords(reason)

            for leave in all_leaves:
                similarity_score = calculate_comprehensive_similarity(
                    leave, reason, reason_keywords, duration_days,
                    leave_start_date, user_team_id, user_role
                )

                if similarity_score > 20:  # Only include reasonably similar cases
                    leave_duration = (
                        leave.end_date - leave.start_date).days + 1
                    leave_user = User.query.get(leave.user_id)

                    precedent = {
                        'duration': leave_duration,
                        'reason': leave.reason,
                        'status': leave.status,
                        'decision_reason': leave.decision_reason or 'No reason provided',
                        'similarity_score': round(similarity_score, 1),
                        'user_role': getattr(leave_user, 'role', 'unknown') if leave_user else 'unknown',
                        'same_team': getattr(leave_user, 'team_id', None) == user_team_id if leave_user else False,
                        'created_at': leave.created_at.strftime('%Y-%m-%d') if leave.created_at else 'unknown',
                        'seasonal_match': is_seasonal_match(leave.start_date, leave_start_date),
                        'match_factors': get_match_factors(leave, reason, reason_keywords, duration_days)
                    }
                    precedents.append(precedent)

            # Sort by similarity score
            precedents.sort(key=lambda x: x['similarity_score'], reverse=True)

            # Calculate weighted approval rate (more recent and similar cases have higher weight)
            approval_rate = calculate_weighted_approval_rate(precedents)

            # Generate enhanced recommendation
            recommendation_data = generate_enhanced_recommendation(
                precedents, approval_rate, duration_days, reason
            )

            return {
                'similar_cases_found': len(precedents),
                'precedents': precedents[:5],  # Return top 5 most similar
                'historical_approval_rate': round(approval_rate, 1),
                'recommendation': recommendation_data['recommendation'],
                'confidence_score': recommendation_data['confidence'],
                'analysis_summary': recommendation_data['summary'],
                'risk_factors': recommendation_data['risk_factors'],
                'supporting_factors': recommendation_data['supporting_factors']
            }

    except Exception as e:
        logger.error(f"Error in enhanced precedent analysis: {str(e)}")
        return {
            'error': str(e),
            'similar_cases_found': 0,
            'precedents': [],
            'historical_approval_rate': 0,
            'recommendation': 'Error in analysis',
            'confidence_score': 0
        }


def extract_keywords(reason: str) -> List[str]:
    """Extract meaningful keywords from leave reason"""
    # Common leave reason keywords
    important_keywords = [
        'medical', 'surgery', 'hospital', 'emergency', 'family', 'wedding',
        'vacation', 'personal', 'sick', 'maternity', 'paternity', 'bereavement',
        'conference', 'training', 'relocation', 'honeymoon', 'graduation'
    ]

    reason_lower = reason.lower()
    keywords = []

    # Extract important keywords
    for keyword in important_keywords:
        if keyword in reason_lower:
            keywords.append(keyword)

    # Extract other meaningful words (length > 3, not common words)
    common_words = {'the', 'and', 'for', 'with',
                    'have', 'will', 'need', 'want', 'going'}
    words = re.findall(r'\b\w{4,}\b', reason_lower)
    for word in words:
        if word not in common_words and word not in keywords:
            keywords.append(word)

    return keywords[:5]  # Return top 5 keywords


def calculate_comprehensive_similarity(leave, reason, reason_keywords, duration_days,
                                       leave_start_date, user_team_id, user_role) -> float:
    """Calculate comprehensive similarity score"""
    score = 0
    max_score = 0

    # 1. Reason similarity (40% weight)
    reason_sim = calculate_reason_similarity(
        leave.reason, reason, reason_keywords)
    score += reason_sim * 0.4
    max_score += 40

    # 2. Duration similarity (25% weight)
    leave_duration = (leave.end_date - leave.start_date).days + 1
    duration_diff = abs(leave_duration - duration_days)
    # Penalty increases with difference
    duration_sim = max(0, 100 - duration_diff * 10)
    score += duration_sim * 0.25
    max_score += 25

    # 3. Team context (20% weight)
    leave_user = User.query.get(leave.user_id)
    if leave_user:
        if getattr(leave_user, 'team_id', None) == user_team_id:
            score += 20 * 0.2  # Same team
        elif getattr(leave_user, 'role', None) == user_role:
            score += 15 * 0.2  # Same role
        else:
            score += 5 * 0.2   # Different context
    max_score += 20

    # 4. Recency (15% weight) - more recent decisions are more relevant
    if leave.created_at:
        days_ago = (datetime.now() - leave.created_at).days
        recency_score = max(0, 100 - days_ago / 10)  # Decays over time
        score += recency_score * 0.15
    max_score += 15

    return (score / max_score) * 100 if max_score > 0 else 0


def calculate_reason_similarity(leave_reason: str, current_reason: str, keywords: List[str]) -> float:
    """Calculate similarity between leave reasons"""
    # Exact match
    if leave_reason.lower() == current_reason.lower():
        return 100

    # Sequence similarity
    seq_sim = SequenceMatcher(
        None, leave_reason.lower(), current_reason.lower()).ratio() * 100

    # Keyword overlap
    leave_reason_lower = leave_reason.lower()
    keyword_matches = sum(
        1 for keyword in keywords if keyword in leave_reason_lower)
    keyword_sim = (keyword_matches / len(keywords)) * 100 if keywords else 0

    # Combined similarity (weighted average)
    return (seq_sim * 0.6 + keyword_sim * 0.4)


def is_seasonal_match(leave_date, current_date_str) -> bool:
    """Check if leaves are in similar time periods"""
    if not current_date_str or not leave_date:
        return False

    try:
        current_date = datetime.strptime(current_date_str, '%Y-%m-%d').date()
        # Consider same month or adjacent months as seasonal match
        month_diff = abs(leave_date.month - current_date.month)
        return month_diff <= 1 or month_diff >= 11  # Handle Dec-Jan case
    except:
        return False


def get_match_factors(leave, reason, keywords, duration_days) -> List[str]:
    """Get list of factors that contributed to the match"""
    factors = []

    leave_duration = (leave.end_date - leave.start_date).days + 1

    # Duration factors
    if abs(leave_duration - duration_days) <= 1:
        factors.append("Exact duration match")
    elif abs(leave_duration - duration_days) <= 3:
        factors.append("Similar duration")

    # Reason factors
    leave_reason_lower = leave.reason.lower()
    reason_lower = reason.lower()

    if any(keyword in leave_reason_lower for keyword in keywords):
        factors.append("Keyword match in reason")

    if SequenceMatcher(None, leave_reason_lower, reason_lower).ratio() > 0.7:
        factors.append("Similar reason description")

    return factors


def calculate_weighted_approval_rate(precedents: List[Dict]) -> float:
    """Calculate approval rate with weights based on similarity"""
    if not precedents:
        return 0

    total_weight = 0
    weighted_approvals = 0

    for precedent in precedents:
        weight = precedent['similarity_score'] / 100  # Normalize to 0-1
        total_weight += weight

        if precedent['status'] == 'Approved':
            weighted_approvals += weight

    return (weighted_approvals / total_weight) * 100 if total_weight > 0 else 0


def generate_enhanced_recommendation(precedents: List[Dict], approval_rate: float,
                                     duration_days: int, reason: str) -> Dict:
    """Generate comprehensive recommendation with analysis"""

    risk_factors = []
    supporting_factors = []

    # Analyze precedents for patterns
    if precedents:
        avg_similarity = sum(p['similarity_score']
                             for p in precedents) / len(precedents)
        recent_cases = [p for p in precedents if p.get(
            'seasonal_match', False)]
        same_team_cases = [p for p in precedents if p.get('same_team', False)]

        # Risk factors
        if approval_rate < 30:
            risk_factors.append(
                "Low historical approval rate for similar requests")

        rejected_cases = [p for p in precedents if p['status'] == 'Rejected']
        if len(rejected_cases) > len(precedents) / 2:
            risk_factors.append("Majority of similar cases were rejected")

        if duration_days > 10 and any('long duration' in p.get('decision_reason', '').lower()
                                      for p in rejected_cases):
            risk_factors.append(
                "Long duration requests historically face scrutiny")

        # Supporting factors
        if approval_rate > 70:
            supporting_factors.append(
                "High historical approval rate for similar requests")

        if same_team_cases and all(p['status'] == 'Approved' for p in same_team_cases):
            supporting_factors.append(
                "Similar requests from team members were approved")

        if recent_cases and all(p['status'] == 'Approved' for p in recent_cases):
            supporting_factors.append("Recent similar requests were approved")

    # Generate recommendation
    if approval_rate >= 80:
        recommendation = "Strong approval recommendation"
        confidence = min(95, approval_rate)
    elif approval_rate >= 60:
        recommendation = "Favor approval with standard review"
        confidence = min(80, approval_rate)
    elif approval_rate >= 40:
        recommendation = "Standard review process"
        confidence = 60
    elif approval_rate >= 20:
        recommendation = "Careful review recommended"
        confidence = 40
    else:
        recommendation = "High scrutiny recommended"
        confidence = 20

    # Adjust confidence based on data quality
    if len(precedents) < 3:
        # Lower confidence with less data
        confidence = max(30, confidence - 20)

    summary = f"Based on {len(precedents)} similar cases with {approval_rate:.1f}% approval rate"

    return {
        'recommendation': recommendation,
        'confidence': round(confidence, 1),
        'summary': summary,
        'risk_factors': risk_factors,
        'supporting_factors': supporting_factors
    }

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
    4. Evaluate leave balance sufficiency and impact

    Focus on:
    - Employee leave patterns and history
    - Leave balance adequacy and negative balance implications
    - Team impact and availability
    - Business timing considerations
    - Risk factors that need attention

    Be thorough but efficient in your analysis. Always check leave balances
    before making recommendations.
    """,
    tools=[
        get_user_leave_history,
        analyze_team_availability,
        analyze_leave_balance,  # NEW: Essential for balance checking
        get_comprehensive_leave_analysis,  # NEW: Complete analysis
        assess_business_calendar_impact,
        get_similar_leave_decisions
    ],
    output_type=ContextAnalysis
)

# Decision Making Agent - Enhanced with balance analysis
decision_agent = Agent(
    name="Leave Decision Maker",
    instructions="""
    You are the primary decision-making agent for employee leave requests.

    Core principles:
    1. Employee wellbeing is paramount
    2. Business continuity must be maintained
    3. Leave balance sufficiency must be verified
    4. Fairness and consistency in decisions
    5. STRICT monitoring of sick leave abuse patterns
    6. Transparency in reasoning
    7. Escalate when human judgment is needed

    SICK LEAVE SPECIFIC GUIDELINES:
    - AUTOMATIC DENIAL for risk_score >= 50 (high abuse pattern)
    - MANDATORY ESCALATION for risk_score >= 30 (concerning pattern)
    - REQUIRE MEDICAL CERTIFICATE for risk_score >= 20
    - FLAG for manager review if risk_score >= 15

    Sick Leave Red Flags (MUST escalate or deny):
    - More than 4 sick leave requests in 3 months
    - More than 15 sick days in 3 months
    - Average gap between sick leaves < 7 days
    - >60% of all leave requests are sick leave
    - Frequent Monday/Friday sick leave pattern (3+ times)
    - Frequent single-day sick leaves (4+ times in 3 months)

    Decision guidelines:
    - ALWAYS check leave balance first using analyze_leave_balance
    - ALWAYS analyze leave history patterns for sick leave abuse
    - For sick leave: Apply stricter scrutiny based on pattern analysis
    - Approve when balance is sufficient, impact is manageable, request is reasonable AND no abuse patterns
    - Consider negative balances only if leave type allows it AND no concerning patterns
    - DENY when balance is insufficient AND negative not allowed
    - DENY when abuse patterns are detected (risk_score >= 50)
    - ESCALATE when concerning patterns detected (risk_score >= 30)
    - ESCALATE when business impact is severe and no alternatives exist
    - REQUIRE DOCUMENTATION for repeated sick leave (>3 in 3 months)

    Balance considerations remain the same as before...

    Always provide clear, empathetic reasoning for your decisions, especially for sick leave denials.

    Output your decision in the following JSON format:
    {
        "status": "Approved|Denied|Escalate",
        "reason": "<short summary>",
        "confidence": <float between 0 and 1>,
        "escalate": <true|false>,
        "agent_reasoning": "<detailed explanation>",
        "business_impact": "<low|medium|high|critical>",
        "balance_impact": "<sufficient|negative_allowed|insufficient|critical>",
        "employee_considerations": "<employee wellbeing factors>",
        "precedent_used": "<reference if any>",
        "recommended_actions": ["<action1>", "<action2>"],
        "pattern_analysis": "<sick leave pattern concerns if any>",
        "documentation_required": <true|false>
    }
    """,
    tools=[
        get_user_leave_history,
        analyze_team_availability,
        analyze_leave_balance,
        get_comprehensive_leave_analysis,
        assess_business_calendar_impact,
        get_similar_leave_decisions
    ],
    output_type=LeaveDecision
)
# Escalation Agent - Enhanced with balance analysis
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
    - Complex leave balance scenarios (negative balances, carryovers, etc.)

    Approach:
    - Gather comprehensive context including detailed balance analysis
    - Consider all stakeholders and balance implications
    - Identify accommodation options for balance shortfalls
    - Evaluate if negative balance approval is appropriate
    - Provide detailed analysis for human reviewers
    - Suggest alternative solutions when possible (partial leave, unpaid leave, etc.)

    Balance escalation scenarios:
    - Requests exceeding available balance when negative not allowed
    - Large negative balance requests requiring special approval
    - Complex multi-year balance calculations
    - Unusual leave type combinations

    Remember: Your goal is to find solutions that work for both employee and business,
    including creative balance management solutions.
    """,
    tools=[
        get_user_leave_history,
        analyze_team_availability,
        analyze_leave_balance,  # NEW: Essential for complex scenarios
        get_comprehensive_leave_analysis,  # NEW: Complete analysis
        assess_business_calendar_impact,
        get_similar_leave_decisions
    ],
    output_type=LeaveDecision
)

# Triage Agent - Enhanced routing logic with balance considerations
triage_agent = Agent(
    name="Leave Request Triage",
    instructions="""
    You are the first point of contact for all leave requests. Your job is to:

    1. Quickly assess the complexity and sensitivity of the request
    2. Perform initial balance check to understand constraints
    3. CRITICALLY IMPORTANT: Analyze sick leave patterns for potential abuse
    4. Route to the appropriate specialist agent (decision_agent or escalation_agent)
    5. Provide a brief analysis of your routing decision

    SICK LEAVE ROUTING RULES (CRITICAL):
    - Risk Score >= 50: → AUTOMATIC escalation_agent (high abuse risk)
    - Risk Score >= 30: → escalation_agent (concerning pattern)
    - Risk Score >= 20: → decision_agent with medical certificate requirement flag
    - Risk Score >= 15: → decision_agent with manager review flag
    - 4+ sick leaves in 3 months: → escalation_agent (excessive frequency)
    - Frequent Monday/Friday pattern: → escalation_agent (suspicious timing)

    Standard routing guidelines:
    → Standard Decision Agent (decision_agent) for:
    - Routine vacation requests (≤14 days) with sufficient balance
    - Standard sick leave (≤5 days) with LOW risk score (<15) and adequate balance
    - Personal leave with clear reasons and available balance
    - Low team/business impact situations
    - Negative balance requests where policy clearly allows it

    → Escalation Specialist (escalation_agent) for:
    - Extended leave requests (>14 days)
    - Medical emergencies or serious illness (>7 days sick leave)
    - Family crisis situations (>5 days)
    - Requests during critical business periods with high team impact
    - Employees with concerning leave patterns (>90 days in 12 months)
    - SICK LEAVE ABUSE PATTERNS (risk score >= 30)
    - Policy exception requests
    - High + critical business impact scenarios
    - Balance-related escalations (same as before)

    Always use analyze_leave_balance AND get_user_leave_history to inform your routing decision.

    Output your decision in the following JSON format:
    {
        "route_to": "<decision_agent|escalation_agent>",
        "reason": "<brief reason for routing>",
        "confidence": <float between 0 and 1>,
        "escalate": <true|false>,
        "balance_status": "<sufficient|negative_allowed|insufficient|needs_review>",
        "pattern_risk_level": "<low|moderate|high|critical>",
        "analysis": "<brief analysis of the request including balance and pattern considerations>"
    }
    """,
    tools=[
        analyze_leave_balance,
        get_user_leave_history,  # CRITICAL: Must analyze patterns
        get_comprehensive_leave_analysis
    ],
    handoffs=[decision_agent, escalation_agent],
    output_type=RoutingDecision
)

# Enhanced RoutingDecision output type to include balance information


@dataclass
class EnhancedRoutingDecision:
    route_to: str
    reason: str
    confidence: float
    escalate: bool
    balance_status: str  # NEW: Balance status indicator
    analysis: str


@dataclass
class EnhancedLeaveDecision:
    status: str
    reason: str
    confidence: float
    escalate: bool
    agent_reasoning: str
    business_impact: str
    balance_impact: str  # NEW: Balance impact indicator
    employee_considerations: str
    precedent_used: str
    recommended_actions: List[str]


@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


class AgenticLeaveSystemSDK:
    def __init__(self):
        self.triage_agent = triage_agent
        self.context_agent = context_agent
        self.executor = ThreadPoolExecutor(max_workers=3)
        with app.app_context():
            if not db.session:
                db.session = db.create_scoped_session()
                logger.info("Initialized new database session")

    def process_leave_decision(self, leave_record, ai_decision: Dict) -> Dict:
        """
        Process the AI decision and update the leave request status and balance

        Args:
            leave_record: Either a Leave object or leave_id (int)
            ai_decision: Dictionary containing AI decision
        """
        try:
            with app.app_context():
                # Handle different input types
                if isinstance(leave_record, int):
                    # It's a leave_id
                    leave = Leave.query.get(leave_record)
                    if not leave:
                        logger.error(
                            f"Leave record with ID {leave_record} not found")
                        return {"error": f"Leave record {leave_record} not found in database"}
                    logger.info(f"Found Leave object with ID: {leave.id}")

                elif hasattr(leave_record, 'id') and hasattr(leave_record, 'status'):
                    # It's already a Leave object from database
                    leave = leave_record
                    logger.info(
                        f"Using existing Leave object with ID: {leave.id}")

                else:
                    # It's a LeaveRequest object, need to find the corresponding Leave
                    leave = self._find_leave_record(leave_record)
                    if not leave:
                        logger.error(
                            "Could not find Leave record for LeaveRequest")
                        return {"error": "Leave request not found in database"}

                # Get balance info before processing (MOVED INSIDE app_context)
                balance_before = self.get_current_balance_info(
                    leave.user_id,
                    leave.leave_type_id,
                    leave.start_date.year
                )
                logger.info(f"🔍 BALANCE BEFORE PROCESSING: {balance_before}")

                old_status = leave.status
                logger.info(
                    f"Processing leave {leave.id} - Old status: {old_status}, New status: {ai_decision['status']}")

                # Calculate leave days for balance update
                leave_days = (leave.end_date - leave.start_date).days + 1
                logger.info(f"Leave duration: {leave_days} days")

                # Update leave status based on AI decision
                if ai_decision['status'] == 'Approved':
                    leave.status = 'Approved'
                    leave.approved_at = datetime.utcnow()
                    leave.decision_reason = ai_decision['reason']

                    # Update the leave balance manually
                    self._update_leave_balance_directly(
                        leave, old_status, 'Approved', leave_days)
                    logger.info(
                        f"✅ Leave request {leave.id} approved and balance updated")

                elif ai_decision['status'] in ['Denied', 'Rejected']:
                    leave.status = 'Rejected'
                    leave.decision_reason = ai_decision['reason']

                    # Update balance (removes from pending)
                    self._update_leave_balance_directly(
                        leave, old_status, 'Rejected', leave_days)
                    logger.info(
                        f"❌ Leave request {leave.id} rejected and balance updated")

                elif ai_decision['status'] == 'Escalate':
                    # Keep as pending but add escalation flag
                    leave.status = 'Pending'
                    leave.decision_reason = f"Escalated: {ai_decision['reason']}"
                    logger.info(
                        f"⬆️ Leave request {leave.id} escalated for human review")

                # Save changes
                db.session.commit()

                # Get balance info after processing (ADDED)
                balance_after = self.get_current_balance_info(
                    leave.user_id,
                    leave.leave_type_id,
                    leave.start_date.year
                )
                logger.info(f"🔍 BALANCE AFTER PROCESSING: {balance_after}")

                return {
                    "success": True,
                    "leave_id": leave.id,
                    "new_status": leave.status,
                    "balance_updated": ai_decision['status'] in ['Approved', 'Rejected'],
                    "balance_before": balance_before,  # ADDED
                    "balance_after": balance_after    # ADDED
                }

        except Exception as e:
            db.session.rollback()
            logger.exception(f"Error processing leave decision: {str(e)}")
            return {"error": f"Failed to process decision: {str(e)}"}

    def _find_leave_record(self, leave_request):
        """Helper method to find Leave record from LeaveRequest"""
        leave = None

        # Strategy 1: By leave_id
        if hasattr(leave_request, 'leave_id') and leave_request.leave_id:
            leave = Leave.query.get(leave_request.leave_id)
            if leave:
                logger.info(
                    f"Found leave by leave_id: {leave_request.leave_id}")
                return leave

        # Strategy 2: By user_id, dates, and status
        if hasattr(leave_request, 'user_id'):
            leave = Leave.query.filter_by(
                user_id=leave_request.user_id,
                start_date=leave_request.start_date,
                end_date=leave_request.end_date,
                status='Pending'
            ).order_by(Leave.created_at.desc()).first()

            if leave:
                logger.info(f"Found leave by user_id and dates: {leave.id}")
                return leave

        # Strategy 3: Most recent pending leave for user
        if hasattr(leave_request, 'user_id'):
            leave = Leave.query.filter_by(
                user_id=leave_request.user_id,
                status='Pending'
            ).order_by(Leave.created_at.desc()).first()

            if leave:
                logger.info(f"Found most recent pending leave: {leave.id}")
                return leave

        return None

    def _update_leave_balance_directly(self, leave, old_status: str, new_status: str, leave_days: int):
        """
        Directly update the leave balance in the LeaveBalance table
        """
        try:
            # Get the leave balance record
            leave_balance = LeaveBalance.query.filter_by(
                user_id=leave.user_id,
                leave_type_id=leave.leave_type_id,
                year=leave.start_date.year if hasattr(
                    leave.start_date, 'year') else datetime.now().year
            ).first()

            if not leave_balance:
                logger.error(
                    f"No leave balance found for user {leave.user_id}, leave type {leave.leave_type_id}")
                return

            # Calculate current remaining balance BEFORE update
            remaining_before = leave_balance.allocated_days - \
                leave_balance.used_days - leave_balance.pending_days

            logger.info(f"BEFORE UPDATE:")
            logger.info(f"  - Allocated: {leave_balance.allocated_days}")
            logger.info(f"  - Used: {leave_balance.used_days}")
            logger.info(f"  - Pending: {leave_balance.pending_days}")
            logger.info(f"  - Remaining: {remaining_before}")
            logger.info(f"  - Status Transition: {old_status} → {new_status}")
            logger.info(f"  - Leave Days: {leave_days}")

            # Handle status transitions
            if old_status == 'Pending' and new_status == 'Approved':
                # Move from pending to used
                leave_balance.pending_days = max(
                    0, leave_balance.pending_days - leave_days)
                leave_balance.used_days += leave_days
                logger.info(
                    f"  - Action: Moved {leave_days} days from pending to used")

            elif old_status == 'Pending' and new_status == 'Rejected':
                # Remove from pending (no change to used)
                leave_balance.pending_days = max(
                    0, leave_balance.pending_days - leave_days)
                logger.info(
                    f"  - Action: Removed {leave_days} days from pending")

            elif old_status in ['Approved', 'Rejected'] and new_status == 'Approved':
                # Direct approval (not from pending) - this should be rare now
                logger.warning(
                    f"  - UNUSUAL: Direct approval without pending status")
                leave_balance.used_days += leave_days
                logger.info(
                    f"  - Action: Added {leave_days} days directly to used")

            elif old_status == 'Approved' and new_status == 'Rejected':
                # Reverse approval - remove from used days
                leave_balance.used_days = max(
                    0, leave_balance.used_days - leave_days)
                logger.info(
                    f"  - Action: Removed {leave_days} days from used (reversal)")

            # Calculate remaining balance AFTER update
            remaining_after = leave_balance.allocated_days - \
                leave_balance.used_days - leave_balance.pending_days

            logger.info(f"AFTER UPDATE:")
            logger.info(f"  - Allocated: {leave_balance.allocated_days}")
            logger.info(f"  - Used: {leave_balance.used_days}")
            logger.info(f"  - Pending: {leave_balance.pending_days}")
            logger.info(f"  - Remaining: {remaining_after}")
            logger.info(
                f"  - Balance Change: {remaining_before} → {remaining_after}")

            # Save the updated balance
            db.session.add(leave_balance)
            db.session.commit()  # Commit immediately to ensure changes persist

            logger.info(f"✅ Balance update completed for user {leave.user_id}")

        except Exception as e:
            logger.error(f"Error updating leave balance directly: {str(e)}")
            raise

    def get_current_balance_info(self, user_id: int, leave_type_id: int, year: int = None) -> Dict:
        """
        Get current balance information with consistent calculation
        """
        if year is None:
            year = datetime.now().year

        leave_balance = LeaveBalance.query.filter_by(
            user_id=user_id,
            leave_type_id=leave_type_id,
            year=year
        ).first()

        if not leave_balance:
            return {
                'allocated': 0,
                'used': 0,
                'pending': 0,
                'remaining': 0,
                'found': False
            }

        remaining = leave_balance.allocated_days - \
            leave_balance.used_days - leave_balance.pending_days

        return {
            'allocated': leave_balance.allocated_days,
            'used': leave_balance.used_days,
            'pending': leave_balance.pending_days,
            'remaining': remaining,
            'found': True,
            'balance_record_id': leave_balance.id
        }

    @staticmethod
    def detect_leave_type(reason: str) -> int:
        """
        Detect appropriate leave type based on the reason provided
        Returns the leave_type_id
        """
        reason_lower = reason.lower().strip()

        # Define keywords for different leave types
        leave_type_keywords = {
            'sick': ['sick', 'illness', 'medical', 'doctor', 'hospital', 'surgery',
                     'health', 'fever', 'flu', 'cold', 'unwell', 'treatment'],
            'annual': ['vacation', 'holiday', 'annual', 'rest', 'break', 'travel',
                       'family time', 'personal break', 'time off'],
            'personal': ['personal', 'family', 'emergency', 'bereavement', 'funeral',
                         'wedding', 'graduation', 'appointment', 'personal matter'],
            'maternity': ['maternity', 'pregnancy', 'childbirth', 'newborn', 'baby'],
            'paternity': ['paternity', 'father', 'newborn baby', 'child birth'],
            'study': ['study', 'education', 'training', 'course', 'exam', 'learning'],
            'compassionate': ['death', 'funeral', 'bereavement', 'family emergency', 'compassionate']
        }

        # Check each leave type for matching keywords
        for leave_type_code, keywords in leave_type_keywords.items():
            if any(keyword in reason_lower for keyword in keywords):
                # Query database to get the leave type ID
                leave_type = LeaveType.query.filter_by(
                    # e.g., 'SL', 'AL', 'PL'
                    code=leave_type_code.upper()[:2] + 'L',
                    is_active=True
                ).first()

                if not leave_type:
                    # Try with different code format
                    leave_type = LeaveType.query.filter(
                        LeaveType.name.ilike(f'%{leave_type_code}%'),
                        LeaveType.is_active == True
                    ).first()

                if leave_type:
                    logger.info(
                        f"Detected leave type: {leave_type.name} (ID: {leave_type.id}) for reason: {reason}")
                    return leave_type.id

        # Default to Annual Leave if no specific type detected
        default_leave_type = LeaveType.query.filter_by(
            code='AL', is_active=True).first()
        if not default_leave_type:
            default_leave_type = LeaveType.query.filter_by(
                is_active=True).first()

        if default_leave_type:
            logger.info(
                f"Using default leave type: {default_leave_type.name} (ID: {default_leave_type.id}) for reason: {reason}")
            return default_leave_type.id

        # If no leave types exist, raise an error
        raise ValueError("No active leave types found in the system")

    def make_intelligent_decision(self, leave_request: LeaveRequest, team_members: List, additional_context: Dict = None) -> Dict:
        """
        Process leave request using OpenAI Agents SDK and apply the decision

        NOTE: This method now returns the AI decision WITHOUT processing it.
        The caller should save the Leave record first, then call process_leave_decision separately.
        """
        try:
            logger.info(
                f"🤖 Starting Agents SDK analysis for user {leave_request.user_id}")

            # Get AI decision
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ai_decision = loop.run_until_complete(
                    self._async_process_request(
                        leave_request, team_members, additional_context)
                )

                # Validate AI decision format
                if isinstance(ai_decision, str):
                    logger.warning(
                        f"Unexpected output type: {type(ai_decision)}")
                    ai_decision = {
                        'status': 'Escalate',
                        'reason': f"Unexpected AI response format: {ai_decision}",
                        'confidence': 0.5,
                        'escalate': True,
                        'agent_reasoning': f"Output: {ai_decision}",
                        'business_impact': 'unknown',
                        'employee_considerations': 'format_error'
                    }
                elif not isinstance(ai_decision, dict):
                    logger.warning(
                        f"Completely unexpected output type: {type(ai_decision)}")
                    ai_decision = {
                        'status': 'Escalate',
                        'reason': f"AI returned unsupported type: {type(ai_decision)}",
                        'confidence': 0.0,
                        'escalate': True,
                        'agent_reasoning': f"Output: {ai_decision}",
                        'business_impact': 'unknown',
                        'employee_considerations': 'format_error'
                    }

                # Return AI decision without processing
                # The caller should save the Leave record and then call process_leave_decision
                logger.info("🤖 Agents SDK Decision: %s", ai_decision)
                return ai_decision

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
                'employee_considerations': 'system_error',
                'error': str(e)
            }

    def process_leave_request_complete(self, leave_request: LeaveRequest, team_members: List, additional_context: Dict = None) -> Dict:
        """
        Complete workflow: Get AI decision, save Leave record, then process the decision
        """
        try:
            # Step 1: Get AI decision (without processing) - this calls make_intelligent_decision
            ai_decision = self.make_intelligent_decision(
                leave_request, team_members, additional_context)

            logger.info(f"🤖 AI Decision received: {ai_decision}")

            # Step 2: Create and save the Leave record
            with app.app_context():
                # Detect leave type if not provided
                leave_type_id = getattr(leave_request, 'leave_type_id', None)
                if not leave_type_id:
                    leave_type_id = self.detect_leave_type(
                        leave_request.reason)

                # Ensure we have proper date objects
                if isinstance(leave_request.start_date, str):
                    start_date = datetime.strptime(
                        leave_request.start_date, '%Y-%m-%d').date()
                else:
                    start_date = leave_request.start_date

                if isinstance(leave_request.end_date, str):
                    end_date = datetime.strptime(
                        leave_request.end_date, '%Y-%m-%d').date()
                else:
                    end_date = leave_request.end_date

                # Calculate duration if not provided (for validation purposes)
                duration_days = getattr(leave_request, 'duration_days', None)
                if not duration_days:
                    duration_days = (end_date - start_date).days + 1

                logger.info(f"📅 Leave duration will be: {duration_days} days")

                # Create Leave record (don't set duration_days - it's calculated automatically)
                leave_record = Leave(
                    user_id=leave_request.user_id,
                    leave_type_id=leave_type_id,
                    start_date=start_date,
                    end_date=end_date,
                    reason=leave_request.reason,
                    status='Pending',  # Always start as Pending
                    created_at=datetime.utcnow()
                )

                # Add to session and commit to get the ID
                db.session.add(leave_record)
                db.session.commit()
                db.session.refresh(leave_record)
                logger.info(
                    f"✅ Leave saved to database with ID: {leave_record.id}")

                # CRITICAL FIX: Add pending days to balance when leave is created
                self._add_pending_days_to_balance(leave_record)

                # Step 3: Process the AI decision
                processing_result = self.process_leave_decision(
                    leave_record, ai_decision)

                # Step 4: Combine results
                final_result = {
                    **ai_decision,  # Include original AI decision
                    **processing_result,  # Include processing results
                    'leave_id': leave_record.id,  # Ensure leave_id is included
                }

                logger.info(f"🎯 Final result: {final_result}")
                return final_result

        except Exception as e:
            db.session.rollback()
            logger.exception(
                f"Error in complete leave request processing: {str(e)}")
            return {
                'status': 'Escalate',
                'new_status': 'Pending',  # Add this for consistency
                'reason': f'System error during complete processing: {str(e)}',
                'confidence': 0.0,
                'escalate': True,
                'error': str(e)
            }

    def _add_pending_days_to_balance(self, leave):
        """
        Add pending days to the balance when a leave request is created
        """
        try:
            leave_days = (leave.end_date - leave.start_date).days + 1

            # Get the leave balance record
            leave_balance = LeaveBalance.query.filter_by(
                user_id=leave.user_id,
                leave_type_id=leave.leave_type_id,
                year=leave.start_date.year
            ).first()

            if not leave_balance:
                logger.error(
                    f"No leave balance found for user {leave.user_id}, leave type {leave.leave_type_id}")
                return

            logger.info(f"ADDING PENDING DAYS:")
            logger.info(f"  - User: {leave.user_id}")
            logger.info(f"  - Leave Days: {leave_days}")
            logger.info(f"  - Pending Before: {leave_balance.pending_days}")

            # Add to pending days
            leave_balance.pending_days += leave_days

            logger.info(f"  - Pending After: {leave_balance.pending_days}")
            logger.info(
                f"  - Remaining Balance: {leave_balance.remaining_days}")

            # Save the updated balance
            db.session.add(leave_balance)
            db.session.commit()

            logger.info(
                f"✅ Added {leave_days} pending days to balance for user {leave.user_id}")

        except Exception as e:
            logger.error(f"Error adding pending days to balance: {str(e)}")
            raise

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

        if validation_result.get('warnings'):
            final_result['validation_warnings'] = validation_result['warnings']

        return final_result


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

                # Parse dates properly
                start_date = datetime.strptime(
                    request.form['start_date'], '%Y-%m-%d').date()
                end_date = datetime.strptime(
                    request.form['end_date'], '%Y-%m-%d').date()
                duration_days = (end_date - start_date).days + 1

                # Create leave request object (don't save to DB yet)
                leave_request = LeaveRequest(
                    reason=request.form['reason'],
                    start_date=start_date,
                    end_date=end_date,
                    duration_days=duration_days,
                    user_id=current_user.id
                )

                # Use the complete workflow method
                agent_system = AgenticLeaveSystemSDK()
                result = agent_system.process_leave_request_complete(
                    leave_request,
                    team.members,
                    {
                        'request_source': 'web_form',
                        'user_role': current_user.role,
                        'team_name': team.name
                    }
                )

                logger.info(f"🤖 Complete processing result: {result}")

                if result.get('error'):
                    flash(
                        f'Error processing leave request: {result["error"]}', 'error')
                    return redirect(url_for('apply_leave'))

                # Enhanced flash messages with Agents SDK insights
                confidence_text = f" (AI Confidence: {result.get('confidence', 0):.0%})"
                flash_message = f"🤖 Leave {result['new_status']}! {result.get('reason', '')}{confidence_text}"

                # Add balance information
                if result.get('balance_after'):
                    balance_info = result['balance_after']
                    balance_text = f" | Remaining balance: {balance_info.get('remaining', 'N/A')} days"
                    flash_message += balance_text

                # Add business impact info if available
                if result.get('business_impact') and result['business_impact'] != 'unknown':
                    flash_message += f" | Business Impact: {result['business_impact']}"

                flash_type = 'success' if result['new_status'] == 'Approved' else 'warning' if result[
                    'new_status'] == 'Pending' else 'info'
                flash(flash_message, flash_type)

                # Show reasoning in a separate message for transparency
                if result.get('agent_reasoning'):
                    reasoning_msg = f"🧠 AI Reasoning: {result['agent_reasoning'][:200]}..."
                    flash(reasoning_msg, 'info')

                # Show balance change details
                if result.get('balance_before') and result.get('balance_after'):
                    before = result['balance_before']
                    after = result['balance_after']
                    balance_change_msg = f"📊 Balance Update: Pending {before.get('pending', 0)} → {after.get('pending', 0)}, Remaining {before.get('remaining', 0)} → {after.get('remaining', 0)}"
                    flash(balance_change_msg, 'info')

                return redirect(url_for('dashboard'))

        except ValueError as e:
            logger.error(f"Date format error: {str(e)}")
            flash('Invalid date format. Please use the date picker.', 'error')
        except Exception as e:
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


def create_sample_data():
    """Create comprehensive sample data across all tables"""
    try:
        # 1. Create Organizations
        tech_org = Organization(name="TechCorp Solutions")
        db.session.add(tech_org)
        db.session.commit()

        # 2. Create Departments
        engineering_dept = Department(name="Engineering", org_id=tech_org.id)
        hr_dept = Department(name="Human Resources", org_id=tech_org.id)
        qa_dept = Department(name="Quality Assurance", org_id=tech_org.id)
        finance_dept = Department(name="Finance", org_id=tech_org.id)

        db.session.add_all([engineering_dept, hr_dept, qa_dept, finance_dept])
        db.session.commit()

        # 3. Create Teams
        dev_team = Team(name="Development Team", dept_id=engineering_dept.id)
        frontend_team = Team(name="Frontend Team", dept_id=engineering_dept.id)
        backend_team = Team(name="Backend Team", dept_id=engineering_dept.id)
        hr_team = Team(name="HR Team", dept_id=hr_dept.id)
        qa_team = Team(name="QA Team", dept_id=qa_dept.id)
        finance_team = Team(name="Finance Team", dept_id=finance_dept.id)

        db.session.add_all(
            [dev_team, frontend_team, backend_team, hr_team, qa_team, finance_team])
        db.session.commit()

        # 4. Create Leave Types
        leave_types = [
            LeaveType(
                name="Annual Leave",
                code="AL",
                default_allocation=25,
                max_carryover=5,
                requires_approval=True,
                can_be_negative=False,
                description="Annual vacation leave"
            ),
            LeaveType(
                name="Sick Leave",
                code="SL",
                default_allocation=10,
                max_carryover=2,
                requires_approval=True,
                can_be_negative=True,
                description="Medical and health-related leave"
            ),
            LeaveType(
                name="Personal Leave",
                code="PL",
                default_allocation=5,
                max_carryover=0,
                requires_approval=True,
                can_be_negative=False,
                description="Personal matters and emergencies"
            ),
            LeaveType(
                name="Maternity Leave",
                code="ML",
                default_allocation=90,
                max_carryover=0,
                requires_approval=True,
                can_be_negative=False,
                description="Maternity and parental leave"
            ),
            LeaveType(
                name="Compassionate Leave",
                code="CL",
                default_allocation=3,
                max_carryover=0,
                requires_approval=True,
                can_be_negative=False,
                description="Bereavement and family emergency leave"
            )
        ]

        db.session.add_all(leave_types)
        db.session.commit()

        # 5. Create Users
        users = [
            User(username='admin', password=generate_password_hash('admin123'),
                 role='admin', team_id=dev_team.id, hire_date=date(2020, 1, 15)),
            User(username='employee', password=generate_password_hash('employee123'),
                 role='user', team_id=dev_team.id, hire_date=date(2021, 3, 10)),
            User(username='john', password=generate_password_hash('john123'),
                 role='user', team_id=frontend_team.id, hire_date=date(2022, 5, 20)),
            User(username='jane', password=generate_password_hash('jane123'),
                 role='user', team_id=qa_team.id, hire_date=date(2021, 8, 5)),
            User(username='hr_manager', password=generate_password_hash('hr123'),
                 role='admin', team_id=hr_team.id, hire_date=date(2019, 11, 1)),
            User(username='mike', password=generate_password_hash('mike123'),
                 role='user', team_id=backend_team.id, hire_date=date(2023, 2, 14)),
            User(username='sarah', password=generate_password_hash('sarah123'),
                 role='user', team_id=finance_team.id, hire_date=date(2022, 7, 8)),
            User(username='david', password=generate_password_hash('david123'),
                 role='manager', team_id=dev_team.id, hire_date=date(2020, 9, 12))
        ]

        db.session.add_all(users)
        db.session.commit()

        # 6. Initialize Leave Balances for all users
        current_year = datetime.now().year

        for user in users:
            for leave_type in leave_types:
                # Create balance for current year
                balance = LeaveBalance(
                    user_id=user.id,
                    leave_type_id=leave_type.id,
                    year=current_year,
                    allocated_days=leave_type.default_allocation,
                    used_days=0,
                    pending_days=0,
                    carried_over=0
                )
                db.session.add(balance)

                # Create some carried over balance from previous year for demonstration
                if leave_type.max_carryover > 0:
                    # Simulate some carryover
                    balance.carried_over = min(3, leave_type.max_carryover)

        db.session.commit()

        # 7. Create Sample Leave Applications
        sample_leaves = [
            Leave(
                user_id=2,  # employee
                leave_type_id=1,  # Annual Leave
                start_date=date(2024, 5, 1),
                end_date=date(2024, 5, 2),
                reason="Family vacation",
                status="Approved",
                decision_reason="Standard vacation approval",
                approved_by=1,  # admin
                approved_at=datetime(2024, 4, 25)
            ),
            Leave(
                user_id=2,  # employee
                leave_type_id=2,  # Sick Leave
                start_date=date(2024, 4, 15),
                end_date=date(2024, 4, 17),
                reason="Flu symptoms",
                status="Approved",
                decision_reason="Medical leave approved",
                approved_by=5,  # hr_manager
                approved_at=datetime(2024, 4, 14)
            ),
            Leave(
                user_id=3,  # john
                leave_type_id=1,  # Annual Leave
                start_date=date(2024, 6, 10),
                end_date=date(2024, 6, 15),
                reason="Wedding ceremony",
                status="Pending",
                decision_reason=None
            ),
            Leave(
                user_id=4,  # jane
                leave_type_id=3,  # Personal Leave
                start_date=date(2024, 5, 5),
                end_date=date(2024, 5, 6),
                reason="Personal emergency",
                status="Approved",
                decision_reason="Emergency situation approved",
                approved_by=5,  # hr_manager
                approved_at=datetime(2024, 5, 4)
            ),
            Leave(
                user_id=6,  # mike
                leave_type_id=2,  # Sick Leave
                start_date=date(2024, 5, 20),
                end_date=date(2024, 5, 22),
                reason="Medical procedure",
                status="Escalate",
                decision_reason="Extended medical leave requires review"
            ),
            Leave(
                user_id=7,  # sarah
                leave_type_id=1,  # Annual Leave
                start_date=date(2024, 7, 1),
                end_date=date(2024, 7, 10),
                reason="Summer vacation",
                status="Pending",
                decision_reason=None
            )
        ]

        db.session.add_all(sample_leaves)
        db.session.commit()

        # 8. Update leave balances based on approved leaves
        for leave in sample_leaves:
            if leave.status == 'Approved':
                leave.update_leave_balance()
            elif leave.status == 'Pending':
                leave.update_leave_balance()

        # 9. Create some Leave Transactions for audit trail
        sample_transactions = [
            LeaveTransaction(
                user_id=2,
                leave_type_id=1,
                leave_id=1,
                transaction_type='usage',
                days_changed=-2,
                balance_before=25,
                balance_after=23,
                description='Annual leave taken - Family vacation',
                created_by=1
            ),
            LeaveTransaction(
                user_id=2,
                leave_type_id=2,
                leave_id=2,
                transaction_type='usage',
                days_changed=-3,
                balance_before=10,
                balance_after=7,
                description='Sick leave taken - Flu symptoms',
                created_by=5
            ),
            LeaveTransaction(
                user_id=4,
                leave_type_id=3,
                leave_id=4,
                transaction_type='usage',
                days_changed=-2,
                balance_before=5,
                balance_after=3,
                description='Personal leave taken - Emergency',
                created_by=5
            )
        ]

        db.session.add_all(sample_transactions)
        db.session.commit()

        logger.info(
            "✅ Complete sample data created successfully across all tables")
        return True

    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        db.session.rollback()
        return False


# Updated reset_database function
@app.route('/dev/reset-db')
def reset_database():
    if not app.debug:
        return "This endpoint is only available in debug mode", 403

    try:
        with app.app_context():
            db.drop_all()
            db.create_all()

            if create_sample_data():
                return "Database reset successfully with comprehensive sample data"
            else:
                return "Database reset failed during sample data creation", 500

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


# Updated initialization in the main section
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
                    "📊 No existing data found, creating comprehensive sample data...")

                if create_sample_data():
                    logger.info("✅ Complete sample data created successfully")
                else:
                    logger.error("❌ Failed to create sample data")
                    print("❌ Failed to create sample data")
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
